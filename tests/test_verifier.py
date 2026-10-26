import json

import pytest

from conftest import ScriptedLLM
from hermes.algebra import Explorer, Plan, parse_action
from hermes.verifier import Goal, Schema, Verifier
from hermes.systems.prover import Prover

EX = "http://example.org/olympics#"
XSD = "http://www.w3.org/2001/XMLSchema#"
PREFIXES = [("ex", EX), ("xsd", XSD)]


@pytest.fixture
def verifier():
    schema = Schema(
        classes={EX + "City", EX + "Place", EX + "OlympicGames", EX + "Country"},
        properties={EX + "hosted", EX + "country", EX + "season", EX + "year"},
        domains={EX + "hosted": {EX + "City"}, EX + "season": {EX + "OlympicGames"}},
        ranges={EX + "hosted": {EX + "OlympicGames"}, EX + "country": {EX + "Country"},
                EX + "season": {XSD + "string"}, EX + "year": {XSD + "integer"}},
        parents={EX + "City": {EX + "Place"}},
    )
    return Verifier(schema, PREFIXES)


def plan(*actions):
    result = Plan()
    for data in actions:
        result = parse_action(data).apply(result)
    return result


def errors(verifier, *actions, goal=None):
    return verifier.check(plan(*actions), goal).errors


def triple(text):
    return {"op": "triple", "pattern": text}


def test_typing_context_comes_from_domains_ranges_and_types(verifier):
    judgment = verifier.check(plan(triple("?c ex:hosted ?g"), triple("?g ex:season ?s")))
    assert judgment.ok
    assert judgment.context["?c"] == {EX + "City"}
    assert judgment.context["?g"] == {EX + "OlympicGames"}
    assert judgment.context["?s"] == {XSD + "string"}


def test_unknown_identifiers_get_a_suggestion(verifier):
    assert errors(verifier, triple("?c ex:hosts ?g")) == [
        "unknown property ex:hosts (did you mean ex:hosted?)"]
    assert "unknown class ex:Cty" in errors(verifier, triple("?c a ex:Cty"))[0]


def test_type_mismatches(verifier):
    assert "got the literal" in errors(verifier, triple('?c ex:hosted "Winter"'))[0]
    found = errors(verifier, triple("?g ex:season ?s"), triple("?s ex:hosted ?x"))
    assert any("is a literal but" in e for e in found)
    assert verifier.check(plan(triple("?c a ex:City"), triple("?c a ex:Country"))).warnings


def test_scoping_and_aggregation_rules(verifier):
    assert errors(verifier, triple("?c ex:hosted ?g"), {"op": "filter", "expr": "?y > 1900"}) == [
        "unbound variable ?y in FILTER"]
    found = errors(verifier, triple("?c ex:hosted ?g"), {"op": "group", "vars": ["?c"]},
                   {"op": "aggregate", "expr": "COUNT(?g)", "alias": "?n"},
                   {"op": "project", "vars": ["?c", "?g", "?n"]})
    assert found == ["?g is neither grouped nor aggregated; add it to GROUP BY or aggregate it"]
    assert errors(verifier, triple("?c ex:hosted ?g"), {"op": "bind", "expr": "1", "var": "?g"})


def test_goals_close_when_the_plan_proves_them(verifier):
    goal = Goal(answer=("?c",), kind="entity", type="ex:Place")
    open_goals = verifier.check(plan(triple("?c ex:hosted ?g"), {"op": "project", "vars": ["?c"]}),
                              goal).goals
    assert open_goals == []
    assert verifier.check(plan(triple("?g ex:season ?c")), goal).goals
    count = Goal(answer=("?n",), kind="number")
    assert verifier.check(plan(triple("?c ex:hosted ?g"),
                             {"op": "aggregate", "expr": "COUNT(?g)", "alias": "?n"},
                             {"op": "project", "vars": ["?n"]}), count).accomplished
    assert verifier.check(plan(triple("?c ex:hosted ?g")), Goal(kind="boolean")).goals


def test_explorer_rejects_ill_typed_steps_before_running_them(olympics):
    explorer = Explorer(olympics, verifier=lambda e: Verifier(Schema(
        classes={EX + "City"}, properties={EX + "hosted"}), e.prefixes))
    before = olympics.executions
    observation = explorer.run(parse_action(triple("?c ex:hostd ?g")))
    assert not observation.ok and observation.error.startswith("kernel: unknown property")
    assert olympics.executions == before
    assert explorer.rejected == 1


def test_prover_refuses_to_finish_with_open_goals(olympics):
    steps = [
        {"op": "goal", "answer": ["?city"], "type": "ex:City"},
        triple("?city ex:hosted ?g"),
        {"op": "finish"},
        triple("?city a ex:City"),
        {"op": "project", "vars": ["?city"]},
        {"op": "finish"},
    ]
    llm = ScriptedLLM([json.dumps(step) for step in steps])
    result = Prover(llm=llm)("Which cities hosted the Olympics?", olympics)
    assert result["trace"][2]["observation"].startswith("refused: open goals remain")
    assert result["trace"][-1]["observation"] == "finished"
    assert "⊢ ?city : ex:City" in llm.prompts[1]
    assert "open goals:\n  goals accomplished" in llm.prompts[-1]


def test_minus_binds_no_variables(verifier):
    found = errors(verifier, triple("?c ex:hosted ?g"), {"op": "minus", "patterns": ["?g ex:year ?y"]},
                   {"op": "filter", "expr": "?y > 1900"})
    assert found == ["unbound variable ?y in FILTER"]


def test_select_star_projects_extra_variables(verifier):
    goal = Goal(answer=("?c",), kind="entity")
    open_goals = verifier.check(plan(triple("?c ex:hosted ?g")), goal).goals
    assert open_goals == ["⊢ project only the answer; drop ?g"]


def test_entities_where_values_belong_and_literals_read_backwards(verifier):
    assert "expects a literal, got the entity ex:Paris" in errors(
        verifier, triple("?g ex:season ex:Paris"))[0]
    assert "would be a subject" in errors(verifier, triple('?g ^ex:hosted "Paris"'))[0]
    assert not errors(verifier, triple('?g ex:season "Winter"'))


def test_ranges_the_ontology_omits_are_read_from_the_data(olympics):
    from hermes.pricing import statistics_for

    bare = Verifier(Schema(properties={EX + "hosted", EX + "season"}), PREFIXES,
                    kind_of=lambda iri: statistics_for(olympics).predicate(iri).kind)
    assert "holds only literals" in errors(bare, triple("?g ex:season ex:Paris"))[0]
    assert "holds only entities" in errors(bare, triple('?c ex:hosted "Winter"'))[0]
    assert not errors(bare, triple('?g ex:season "Winter"'))
