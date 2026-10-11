import json

from conftest import ScriptedLLM
from hermes.algebra import Explorer, Plan, parse_action, render
from hermes.systems.algebra_agent import AlgebraAgent


def act(explorer, **data):
    return explorer.run(parse_action(data))


def test_render_covers_the_operators():
    plan = Plan()
    for data in [
        {"op": "triple", "pattern": "?c ex:hosted ?g"},
        {"op": "optional", "patterns": ["?c ex:country ?k"]},
        {"op": "union", "branches": [["?g ex:season \"Winter\""], ["?g ex:season \"Summer\""]]},
        {"op": "minus", "patterns": ["?c ex:country ex:USA"]},
        {"op": "bind", "expr": "STR(?c)", "var": "s"},
        {"op": "filter", "expr": "FILTER(?s != \"\")"},
        {"op": "group", "vars": ["?c"]},
        {"op": "aggregate", "expr": "COUNT(?g)", "alias": "?n"},
        {"op": "having", "expr": "?n > 1"},
        {"op": "order", "expr": "DESC(?n)"},
        {"op": "limit", "n": 3},
        {"op": "offset", "n": 1},
        {"op": "distinct"},
    ]:
        plan = parse_action(data).apply(plan)
    text = render(plan)
    for fragment in ["OPTIONAL { ?c ex:country ?k . }", "UNION", "MINUS {", "BIND(STR(?c) AS ?s)",
                     "FILTER(?s != \"\")", "SELECT DISTINCT ?c (COUNT(?g) AS ?n)", "GROUP BY ?c",
                     "HAVING((COUNT(?g) > 1))", "ORDER BY DESC(?n)", "LIMIT 3", "OFFSET 1"]:
        assert fragment in text


def test_explorer_reports_how_each_action_changes_the_result(olympics):
    explorer = Explorer(olympics)
    assert act(explorer, op="triple", pattern="?city ex:hosted ?g").count == 19
    observation = act(explorer, op="triple", pattern='?g ex:season "Winter"')
    assert (observation.previous, observation.count) == (19, 7)
    assert "19 -> 7 solutions" in observation.summary()


def test_preview_does_not_commit_and_undo_restores(olympics):
    explorer = Explorer(olympics)
    act(explorer, op="triple", pattern="?city ex:hosted ?g")
    preview = explorer.preview(parse_action({"op": "triple", "pattern": "?city ex:country ex:USA"}))
    assert preview.count == 8
    assert len(explorer.plan.triples) == 1
    act(explorer, op="triple", pattern="?city ex:country ex:USA")
    assert act(explorer, op="undo").count == 19


def test_invalid_actions_are_rejected_without_changing_the_plan(olympics):
    explorer = Explorer(olympics)
    act(explorer, op="triple", pattern="?city ex:hosted ?g")
    observation = act(explorer, op="filter", expr="?g >")
    assert not observation.ok
    assert explorer.plan.filters == ()


def test_lookup_and_inspect_read_the_graph(olympics):
    explorer = Explorer(olympics)
    found = act(explorer, op="lookup", text="lake placid")
    assert found.sample[0]["entity"] == "ex:LakePlacid"
    inspected = act(explorer, op="inspect", iri="ex:OlympicGames")
    assert {"ex:season", "ex:year"} <= {row["predicate"] for row in inspected.sample}


def test_agent_follows_a_scripted_policy(olympics):
    steps = [
        {"op": "lookup", "text": "USA"},
        {"op": "triple", "pattern": "?city ex:hosted ?g"},
        {"op": "triple", "pattern": "?city ex:country ex:USA"},
        {"op": "triple", "pattern": '?g ex:season "Winter"'},
        {"op": "project", "vars": ["?city"]},
        {"op": "distinct"},
        {"op": "finish"},
    ]
    llm = ScriptedLLM([json.dumps(step) for step in steps])
    result = AlgebraAgent(llm=llm)("Which US cities hosted the Winter Olympics?", olympics)
    rows = olympics.query(result["sparql"])["results"]["bindings"]
    assert {row["city"]["value"].rsplit("#")[1] for row in rows} == {
        "LakePlacid", "SquawValley", "SaltLakeCity"}
    assert result["steps"] == 7
    assert "8 -> 4 solutions" in llm.prompts[-1]
    assert "entity=ex:USA" in llm.prompts[1]


def test_agent_recovers_from_bad_replies(olympics):
    llm = ScriptedLLM(["no json here", json.dumps({"op": "triple", "pattern": "?c a ex:City"}),
                       json.dumps({"op": "finish"})])
    result = AlgebraAgent(llm=llm)("List cities", olympics)
    assert "invalid" in result["trace"][0]["observation"]
    assert "?c a ex:City ." in result["sparql"]


def test_lookup_falls_back_to_partial_names(olympics):
    explorer = Explorer(olympics)
    found = act(explorer, op="lookup", text="Mr. Placid")
    assert found.sample[0]["entity"] == "ex:LakePlacid"


def test_group_alone_and_hidden_aggregates_render_valid_sparql(olympics):
    explorer = Explorer(olympics)
    for data in [
        {"op": "triple", "pattern": "?city ex:hosted ?g"},
        {"op": "triple", "pattern": '?g ex:season "Winter"'},
        {"op": "group", "vars": ["?city"]},
    ]:
        assert act(explorer, **data).ok
    assert explorer.plan.group_by == ("?city",)
    for data in [
        {"op": "aggregate", "expr": "COUNT(?g)", "alias": "?n"},
        {"op": "order", "expr": "DESC(?n)"},
        {"op": "limit", "n": 1},
        {"op": "project", "vars": ["?city"]},
    ]:
        assert act(explorer, **data).ok
    assert "ORDER BY DESC(COUNT(?g))" in explorer.sparql
    assert "SELECT ?city WHERE" in explorer.sparql
    rows = olympics.query(explorer.sparql)["results"]["bindings"]
    assert [row["city"]["value"].rsplit("#")[1] for row in rows] == ["LakePlacid"]


def test_malformed_actions_are_invalid_not_crashes():
    import pytest

    for bad in [{"op": "triple", "pattern": None}, {"op": "goal", "answer": [["?x"]]},
                {"op": "limit", "n": "many"}, {"op": "project", "vars": [1, 2]}]:
        with pytest.raises(ValueError):
            parse_action(bad)
    assert parse_action({"op": "limit", "n": "3"}).n == "3"
