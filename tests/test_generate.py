from pathlib import Path

from hermes.generate import Context, Example, build_prompt
from hermes.ontology import load_vocabulary
from hermes.stages import apply_stages
from hermes.validate import lift_modifiers, parse_error

FIXTURES = Path(__file__).parent / "fixtures"


def test_declared_prefix_is_kept():
    vocabulary = load_vocabulary(FIXTURES / "city.ttl")
    assert ("ex", "http://example.org/demo#") in vocabulary["prefixes"]
    assert 'ex:City "City"' in vocabulary["classes"]
    assert 'ex:inCountry "in country"' in vocabulary["properties"]


def test_prefix_is_derived_when_the_file_declares_none():
    vocabulary = load_vocabulary(FIXTURES / "bare.ttl")
    assert vocabulary["prefixes"] == [("outage", "https://example.org/outage#")]
    assert vocabulary["classes"] == ['outage:Record "Outage record"']


def test_solution_modifiers_move_out_of_where():
    query = """SELECT ?city WHERE {
  ?s a ex:City .
  ORDER BY ?city
  LIMIT 5
}"""
    lifted = lift_modifiers(query)
    assert "ORDER BY ?city" not in lifted.split("}", 1)[0]
    assert lifted.strip().endswith("ORDER BY ?city\nLIMIT 5")
    assert parse_error("PREFIX ex: <http://example.org/demo#>\n" + lifted) is None


def test_modifier_inside_a_subquery_stays_put():
    query = """SELECT ?city WHERE {
  {
    SELECT ?city WHERE { ?s a ex:City }
    GROUP BY ?city
  }
}"""
    assert lift_modifiers(query) == query


def test_prompt_includes_context_and_examples():
    vocabulary = load_vocabulary(FIXTURES / "city.ttl")
    prompt = build_prompt(
        "Which cities are listed?",
        vocabulary,
        Context(
            text="Prefer countries in Europe.",
            examples=[
                Example(
                    nl="List cities.",
                    sparql="SELECT ?city WHERE { ?city a ex:City }",
                )
            ],
        ),
    )
    assert "Prefer countries in Europe." in prompt
    assert "SELECT ?city WHERE { ?city a ex:City }" in prompt
    assert prompt.endswith("Question: Which cities are listed?")


def test_research_stages_leave_the_draft_unchanged():
    draft = "SELECT ?city WHERE { ?city a ex:City }"
    assert apply_stages(draft, {"classes": []}, Context()) == draft


def test_registered_system_runs_through_ask():
    import hermes

    @hermes.systems.register("echo-test")
    def echo(question, ontology, context):
        return {"sparql": f"# {question}"}

    assert "echo-test" in hermes.systems.available()
    assert hermes.ask("hi", "x.ttl", system="echo-test") == {"sparql": "# hi"}
