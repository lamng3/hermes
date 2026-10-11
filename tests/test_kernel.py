import json

import pytest

from conftest import ScriptedLLM
from hermes.algebra import Explorer, Plan, parse_action
from hermes.kernel import Goal, Kernel, Schema
from hermes.systems.prover import Prover

EX = "http://example.org/olympics#"
XSD = "http://www.w3.org/2001/XMLSchema#"
PREFIXES = [("ex", EX), ("xsd", XSD)]


@pytest.fixture
def kernel():
    schema = Schema(
        classes={EX + "City", EX + "Place", EX + "OlympicGames", EX + "Country"},
        properties={EX + "hosted", EX + "country", EX + "season", EX + "year"},
        domains={EX + "hosted": {EX + "City"}, EX + "season": {EX + "OlympicGames"}},
        ranges={EX + "hosted": {EX + "OlympicGames"}, EX + "country": {EX + "Country"},
                EX + "season": {XSD + "string"}, EX + "year": {XSD + "integer"}},
        parents={EX + "City": {EX + "Place"}},
    )
    return Kernel(schema, PREFIXES)


def plan(*actions):
    result = Plan()
    for data in actions:
        result = parse_action(data).apply(result)
    return result


def errors(kernel, *actions, goal=None):
    return kernel.check(plan(*actions), goal).errors


def triple(text):
    return {"op": "triple", "pattern": text}


def test_typing_context_comes_from_domains_ranges_and_types(kernel):
    judgment = kernel.check(plan(triple("?c ex:hosted ?g"), triple("?g ex:season ?s")))
    assert judgment.ok
    assert judgment.context["?c"] == {EX + "City"}
    assert judgment.context["?g"] == {EX + "OlympicGames"}
    assert judgment.context["?s"] == {XSD + "string"}


def test_unknown_identifiers_get_a_suggestion(kernel):
    assert errors(kernel, triple("?c ex:hosts ?g")) == [
        "unknown property ex:hosts (did you mean ex:hosted?)"]
    assert "unknown class ex:Cty" in errors(kernel, triple("?c a ex:Cty"))[0]


def test_type_mismatches(kernel):
    assert "got the literal" in errors(kernel, triple('?c ex:hosted "Winter"'))[0]
    found = errors(kernel, triple("?g ex:season ?s"), triple("?s ex:hosted ?x"))
    assert any("is a literal but" in e for e in found)
    assert kernel.check(plan(triple("?c a ex:City"), triple("?c a ex:Country"))).warnings


def test_scoping_and_aggregation_rules(kernel):
    assert errors(kernel, triple("?c ex:hosted ?g"), {"op": "filter", "expr": "?y > 1900"}) == [
        "unbound variable ?y in FILTER"]
    found = errors(kernel, triple("?c ex:hosted ?g"), {"op": "group", "vars": ["?c"]},
                   {"op": "aggregate", "expr": "COUNT(?g)", "alias": "?n"},
                   {"op": "project", "vars": ["?c", "?g", "?n"]})
    assert found == ["?g is neither grouped nor aggregated; add it to GROUP BY or aggregate it"]
    assert errors(kernel, triple("?c ex:hosted ?g"), {"op": "bind", "expr": "1", "var": "?g"})


def test_goals_close_when_the_plan_proves_them(kernel):
    goal = Goal(answer=("?c",), kind="entity", type="ex:Place")
    open_goals = kernel.check(plan(triple("?c ex:hosted ?g"), {"op": "project", "vars": ["?c"]}),
                              goal).goals
    assert open_goals == []
    assert kernel.check(plan(triple("?g ex:season ?c")), goal).goals
    count = Goal(answer=("?n",), kind="number")
    assert kernel.check(plan(triple("?c ex:hosted ?g"),
                             {"op": "aggregate", "expr": "COUNT(?g)", "alias": "?n"},
                             {"op": "project", "vars": ["?n"]}), count).accomplished
    assert kernel.check(plan(triple("?c ex:hosted ?g")), Goal(kind="boolean")).goals


def test_explorer_rejects_ill_typed_steps_before_running_them(olympics):
    explorer = Explorer(olympics, kernel=lambda e: Kernel(Schema(
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


def test_minus_binds_no_variables(kernel):
    found = errors(kernel, triple("?c ex:hosted ?g"), {"op": "minus", "patterns": ["?g ex:year ?y"]},
                   {"op": "filter", "expr": "?y > 1900"})
    assert found == ["unbound variable ?y in FILTER"]


def test_select_star_projects_extra_variables(kernel):
    goal = Goal(answer=("?c",), kind="entity")
    open_goals = kernel.check(plan(triple("?c ex:hosted ?g")), goal).goals
    assert open_goals == ["⊢ project only the answer; drop ?g"]
