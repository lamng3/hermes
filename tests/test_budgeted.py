import json

from conftest import ScriptedLLM
from hermes.algebra import Explorer
from hermes.binding import Catalog, Slot, assignments, similarity, slots, substitute
from hermes.pricing import statistics_for
from hermes.selection import Scorer
from hermes.systems.budgeted import Budgeted

WINTER = {
    "goal": {"answer": ["?c"], "kind": "entity", "type": "[city]", "many": True},
    "actions": [
        {"op": "triple", "pattern": "?c [hosted] ?g"},
        {"op": "triple", "pattern": "?g [season] [Winter]"},
        {"op": "project", "vars": ["?c"]},
        {"op": "distinct"},
    ],
}
WINTER_CITIES = {"LakePlacid", "SquawValley", "SaltLakeCity", "Chamonix", "Oslo", "Sapporo"}


def catalog(kg):
    explorer = Explorer(kg)
    return Catalog(kg, explorer, statistics_for(kg))


def cities(kg, sparql):
    rows = kg.query(sparql)["results"]["bindings"]
    return {row["c"]["value"].rsplit("#", 1)[-1] for row in rows}


def test_slots_take_their_kind_from_their_position():
    assert slots(WINTER) == [Slot("city", "class"), Slot("hosted", "property"),
                             Slot("season", "property"), Slot("Winter", "entity")]
    typed = {"actions": [{"op": "triple", "pattern": "?x a [Olympic Games] ; [year] [1980]"}]}
    assert slots(typed) == [Slot("Olympic Games", "class"), Slot("year", "property"),
                            Slot("1980", "entity")]


def test_phrases_bind_to_ontology_terms_and_entities(olympics):
    found = catalog(olympics)
    assert found.rank(Slot("hosted", "property"))[0].term == "ex:hosted"
    assert found.rank(Slot("host", "property"))[0].term == "ex:hosted"
    assert found.rank(Slot("cities", "class"))[0].term == "ex:City"
    assert found.rank(Slot("Lake Placid", "entity"))[0].term == "ex:LakePlacid"
    # A phrase that names no entity can still be a value.
    assert found.rank(Slot("Winter", "entity"))[-1].term == '"Winter"'
    assert found.rank(Slot("1980", "entity"))[-1].term == "1980"
    assert similarity("hosted", "hosted") == 1.0 > similarity("hosted", "country")


def test_assignments_come_best_first(olympics):
    found = catalog(olympics)
    choices = {slot: found.rank(slot) for slot in slots(WINTER)}
    ranked = assignments(choices, 5)
    weights = [weight for _assignment, weight in ranked]
    assert weights == sorted(weights, reverse=True)
    bound = substitute(WINTER, ranked[0][0])
    assert bound["goal"]["type"] == "ex:City"
    assert bound["actions"][0]["pattern"] == "?c ex:hosted ?g"


def test_the_planner_binds_checks_prices_and_runs_the_best_candidate(olympics):
    llm = ScriptedLLM([json.dumps(WINTER)])
    output = Budgeted(llm=llm)("Which cities hosted the Winter Olympics?", olympics)
    assert cities(olympics, output["sparql"]) == WINTER_CITIES
    assert output["spent"]["llm_calls"] == 1
    assert output["confidence"] >= 0.75 and not output["abstain"]
    executed = [c for c in output["candidates"] if c["rows"] is not None]
    assert executed and all("estimate" in c and "features" in c for c in executed)


def test_a_failed_reading_is_parsed_again_with_what_was_learned(olympics):
    wrong = {"goal": {"answer": ["?c"]},
             "actions": [{"op": "triple", "pattern": "?c [hosted] [Nowhere Games]"}]}
    llm = ScriptedLLM([json.dumps(wrong), json.dumps(WINTER)])
    output = Budgeted(llm=llm, llm_calls=2)("Which cities hosted the Winter Olympics?", olympics)
    assert output["readings"] == 2
    assert "Earlier readings" in llm.prompts[1] and "ran: 0 rows" in llm.prompts[1]
    assert cities(olympics, output["sparql"]) == WINTER_CITIES


def test_ill_typed_candidates_never_run(olympics):
    reading = {"goal": {"answer": ["?g"]},
               "actions": [{"op": "triple", "pattern": "[Lake Placid] ex:hosted ?g"}]}
    planner = Budgeted(llm=ScriptedLLM([json.dumps(reading)]), llm_calls=1)
    output = planner("What did Lake Placid host?", olympics)
    # The phrase as a literal would be a subject: rejected before it runs.
    assert output["verifier_rejections"] == 1
    assert [c["sparql"].splitlines()[-2].strip() for c in output["candidates"]] == [
        "ex:LakePlacid ex:hosted ?g ."]


def test_the_budget_caps_executions_and_expensive_queries(olympics):
    llm = ScriptedLLM([json.dumps(WINTER)] * 3)
    output = Budgeted(llm=llm, max_query_seconds=0.0)("Which cities hosted the Winter Olympics?",
                                                      olympics)
    assert output["spent"]["executions"] == 0  # nothing was affordable; the best guess stands
    assert output["sparql"]
    llm = ScriptedLLM([json.dumps(WINTER)] * 3)
    output = Budgeted(llm=llm, executions=1, accept=1.1)("Which cities hosted the Winter Olympics?",
                                                         olympics)
    assert output["spent"]["executions"] == 1


def test_the_scorer_learns_from_labelled_candidates():
    examples = [({"executed": 1, "empty": 1}, 0)] * 20 + [({"executed": 1, "empty": 0}, 1)] * 20
    scorer = Scorer({name: 0.0 for name in Scorer().weights}).fit(examples, epochs=2000, rate=0.5)
    assert scorer.weights["empty"] < -1
    assert scorer.probability({"executed": 1}) > 0.8 > 0.2 > scorer.probability(
        {"executed": 1, "empty": 1})


def test_a_property_can_be_read_backwards(olympics):
    found = catalog(olympics)
    hosted = found.rank(Slot("hosted", "property"))
    assert "^ex:hosted" in [c.term for c in hosted]
    reading = {"actions": [{"op": "triple", "pattern": "?g [hosted] ?c"}]}
    backwards = {Slot("hosted", "property"): next(c for c in hosted if c.term == "^ex:hosted")}
    assert substitute(reading, backwards)["actions"][0]["pattern"] == "?c ex:hosted ?g"


def test_repairs_for_what_small_models_write(olympics):
    from hermes.binding import collapse_labels, drop_unbound_type, open_unknown
    from hermes.binding import mentioned as exists

    explorer = Explorer(olympics)
    guessed = {"actions": [{"op": "triple", "pattern": "ex:LakePlacidCity [hosted] ?g"}]}
    repaired = open_unknown(guessed, lambda iri: exists(olympics, iri), explorer.expand)
    assert repaired["actions"][0]["pattern"] == "[Lake Placid City] [hosted] ?g"
    known = {"actions": [{"op": "triple", "pattern": "ex:LakePlacid ex:hosted ?g"}]}
    assert open_unknown(known, lambda iri: exists(olympics, iri), explorer.expand) == known

    labelled = {"goal": {"answer": ["?g"]},
                "actions": [{"op": "triple", "pattern": "?c ex:hosted ?g"},
                            {"op": "triple", "pattern": '?c rdfs:label "Lake Placid"'}]}
    [(variant, score)] = collapse_labels(labelled, catalog(olympics).resolve)
    assert variant["actions"] == [{"op": "triple", "pattern": "ex:LakePlacid ex:hosted ?g"}]
    assert score == 1.0

    quoted = {"actions": [{"op": "triple", "pattern": '?c ex:country "Norway"'}]}
    opened = open_unknown(quoted, lambda iri: True, explorer.expand)
    assert opened["actions"][0]["pattern"] == "?c ex:country [=Norway]"
    assert slots(opened) == [Slot("=Norway", "value")]
    values = [c.term for c in catalog(olympics).rank(Slot("=Norway", "value"))]
    assert values[:2] == ['"Norway"', "ex:Norway"]
    assert catalog(olympics).rank(Slot("ex:hosted", "property"))[0].term == "ex:hosted"
    assert catalog(olympics).rank(Slot("rdf:type", "property"))[0].term == "a"
    assert catalog(olympics).rank(Slot("instance_of", "property"))[0].term == "a"

    typed = {"goal": {"answer": ["?c"], "type": "[telephone]"}, "actions": []}
    assert drop_unbound_type(typed, catalog(olympics))["goal"]["type"] == ""


def test_a_session_explains_every_phase(olympics):
    session = Budgeted().start("Which cities hosted the Winter Olympics?", olympics, explain=True)
    text = session.next_prompt()
    assert "Question: Which cities hosted the Winter Olympics?" in text
    session.take(json.dumps(WINTER))
    result = session.result()
    phases = [event["phase"] for event in session.events]
    assert phases[:4] == ["parse", "repair", "bind", "check"] and phases[-1] == "answer"
    bind = next(e for e in session.events if e["phase"] == "bind")
    assert {"phrase": "hosted", "kind": "property"}.items() <= bind["slots"][1].items()
    check = next(e for e in session.events if e["phase"] == "check")
    assert all({"ok", "sparql", "errors", "context"} <= set(c) for c in check["candidates"])
    run = next(e for e in session.events if e["phase"] == "run")
    assert run["accepted"] and run["rows"] == 6 and run["p"] > run["p_before"]
    assert session.done and session.next_prompt() is None
    assert cities(olympics, result["sparql"]) == WINTER_CITIES


def test_label_phrases_bind_to_the_label_properties_the_data_uses(olympics):
    found = catalog(olympics)
    assert found.rank(Slot("label", "property"))[0].term == "rdfs:label"
    assert "skos:prefLabel" not in [c.term for c in found.rank(Slot("label", "property"))]


def test_ill_typed_assignments_do_not_use_up_the_candidate_cap(olympics):
    # [Lake Placid] as a subject: the entity is well-typed, the literal is not.
    reading = {"goal": {"answer": ["?g"]},
               "actions": [{"op": "triple", "pattern": "[Lake Placid] [hosted] ?g"}]}
    output = Budgeted(llm=ScriptedLLM([json.dumps(reading)]), llm_calls=1, candidates=1)(
        "What did Lake Placid host?", olympics)
    assert output["candidates"] and "ex:LakePlacid ex:hosted ?g" in output["sparql"]
