import json
from pathlib import Path

import pytest

import hermes
from conftest import ScriptedLLM
from hermes.systems import uniqgen
from hermes.systems.uniqgen import UniQGen

OLYMPICS = Path(__file__).parent / "fixtures" / "olympics.ttl"
QUESTION = "Which cities in the USA have hosted the Olympics in February?"
WINTER_HOSTS = "Lake Placid; Salt Lake City; Squaw Valley"
# The CTable of the paper's Appendix III, written as SPARQL. The last row has
# no match in the graph and should be pruned.
CTABLE = {
    "answer": "?city",
    "constraints": [
        "?city ex:hosted ?games",
        "?city ex:country ex:USA",
        "?games ex:year ?year",
        "FILTER(?year > 1896)",
        '?games ex:season "Winter"',
        '?games ex:season "Summer"',
        '?games ex:season "Spring"',
    ],
}


@pytest.fixture
def llm():
    def reply(prompt, system):
        if "constraint table" in prompt:
            return json.dumps(CTABLE)
        return WINTER_HOSTS

    return ScriptedLLM(reply)


def rows(sparql):
    return [line.strip() for line in sparql.split("WHERE {", 1)[1].splitlines()[1:-1]]


def test_entity_linking_finds_the_country_and_the_class(llm):
    UniQGen(llm=llm)(QUESTION, OLYMPICS)
    assert "- ex:USA" in llm.prompts[0]
    assert "- ex:OlympicGames" in llm.prompts[0]
    assert "ex:LakePlacid ex:hosted ex:Winter1932 ." in llm.prompts[0]


def test_ctable_is_scored_and_pruned(llm):
    result = UniQGen(llm=llm)(QUESTION, OLYMPICS)
    table = {row["constraint"]: row["u"] for row in result["ctable"]}
    assert '?games ex:season "Spring"' not in table
    assert table["?city ex:hosted ?games"] == 1.0
    assert table['?games ex:season "Winter"'] < table['?games ex:season "Summer"']


def test_chase_covers_and_backchase_minimizes(llm):
    result = UniQGen(llm=llm)(QUESTION, OLYMPICS)
    assert result["grounded"] == ["Lake Placid", "Salt Lake City", "Squaw Valley"]
    assert '?games ex:season "Summer" .' not in rows(result["universal"])
    assert sorted(rows(result["minimal"])) == sorted(
        [
            "?city ex:hosted ?games .",
            "?city ex:country ex:USA .",
            '?games ex:season "Winter" .',
        ]
    )
    assert result["sparql"] == result["minimal"]


def test_gold_oracle_replaces_the_llm_oracle(llm):
    system = UniQGen(
        oracle=lambda question, context: ["Atlanta", "Los Angeles", "St. Louis"], llm=llm
    )
    result = system(QUESTION, OLYMPICS)
    assert len(llm.prompts) == 1
    assert '?games ex:season "Summer" .' in rows(result["minimal"])


def test_ungrounded_reference_skips_the_search(llm):
    result = UniQGen(oracle=lambda question, context: ["Atlantis"], llm=llm)(QUESTION, OLYMPICS)
    assert "note" in result
    assert "minimal" not in result


def test_gold_oracle_reads_iris_from_the_context(llm):
    context = hermes.Context(answers=[
        "http://example.org/olympics#LakePlacid",
        "http://example.org/olympics#SaltLakeCity",
        "http://example.org/olympics#SquawValley",
    ])
    result = UniQGen(oracle="gold", llm=llm)(QUESTION, OLYMPICS, context)
    assert len(llm.prompts) == 1
    assert '?games ex:season "Winter" .' in rows(result["minimal"])


def test_runs_on_rdflib_too(llm):
    kg = hermes.KnowledgeGraph.from_files(OLYMPICS, engine="rdflib")
    result = UniQGen(llm=llm)(QUESTION, kg)
    assert '?games ex:season "Winter" .' in rows(result["minimal"])


def test_answers_are_parsed_from_a_list():
    assert uniqgen.parse_answers("1. Oslo\n- Paris; UNKNOWN") == ["Oslo", "Paris"]


def test_registered_as_a_system():
    assert "uniqgen" in hermes.systems.available()
