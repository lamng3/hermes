from rdflib import Variable

from conftest import ScriptedLLM
from hermes.pipeline import Hermes
from hermes.pipeline.catalog import catalog_for
from hermes.pipeline.events import Recorder
from hermes.pipeline.fewshot import Example, FewShotSelector, FewShotStore
from hermes.pipeline.names import Prefixes
from hermes.pipeline.prompt import PromptBuilder
from hermes.pipeline.refiner import OntologyRefiner, mentions
from hermes.pipeline.repair import QueryRefiner
from hermes.pricing.budget import Budget, Price, Spend, llm_price
from hermes.retrieval import Document, Lexical, words

EX = "http://example.org/olympics#"
WINTER = """```sparql
PREFIX ex: <http://example.org/olympics#>
SELECT DISTINCT ?c WHERE { ?c ex:hosted ?g . ?g ex:season "Winter" }
```"""


def cities(rows):
    return {str(row[Variable("c")]).rsplit("#", 1)[-1] for row in rows}


def test_lexical_retrieval_ranks_by_shared_words():
    assert words("hostedCities of the Olympics") == ["host", "city", "olympic"]
    ranked = Lexical().fit([Document("hosted: a city hosted games"), Document("country"),
                            Document("season of the games")]).rank("Which city hosted them?")
    assert ranked[0].document.text.startswith("hosted")


def test_the_catalog_reads_the_ontology_and_the_data(olympics):
    catalog = catalog_for(olympics)
    hosted, season = catalog.terms[EX + "hosted"], catalog.terms[EX + "season"]
    assert hosted.holds == "entity" and season.holds == "literal"
    assert hosted.used_by and EX + "City" in hosted.used_by
    assert dict(season.values) == {"Summer": 12, "Winter": 7}
    assert catalog_for(olympics) is catalog


def test_the_refiner_keeps_what_the_question_needs(olympics):
    catalog = catalog_for(olympics)
    refined = OntologyRefiner(catalog)("Which cities in Norway hosted the Winter Olympics?")
    names = Prefixes.of(olympics)
    text = refined.render(catalog, names)
    assert "ex:hosted" in text and "ex:City" in text
    assert ("ex:season", "Winter") in [(names.short(t.iri), v) for t, v in refined.values]
    assert [e.label for e in refined.entities] == ["Norway"]
    assert mentions("Which department does Ms. Brant belong to?") == ["Ms. Brant"]


def test_the_query_refiner_repairs_what_rules_can(olympics):
    catalog = catalog_for(olympics)
    refiner = QueryRefiner(catalog, Prefixes.of(olympics))
    refined = refiner("SELECT ?c WHERE { ?c ex:host ?g . ?g ex:season \"Winter\" }")
    assert any("declared the prefixes ex" in r for r in refined.repairs)
    assert any("ex:host → ex:hosted" in r for r in refined.repairs)
    assert not refined.problems
    wrong = refiner("PREFIX ex: <%s> SELECT ?c WHERE { ?c ex:hosted \"Winter\" }" % EX)
    assert "links to entities" in wrong.problems[0]
    missing = refiner("PREFIX ex: <%s> SELECT ?g WHERE { ex:Oslo ex:hosted ?g . ex:Oslo ex:year ?y }"
                      % EX)
    assert "no match for the pattern ex:Oslo ex:year" in missing.problems[0]


def test_the_pipeline_answers_with_one_call(olympics):
    recorder = Recorder()
    hermes = Hermes(olympics, llm=ScriptedLLM([WINTER]), observers=[recorder])
    answer = hermes.ask("Which cities hosted the Winter Olympics?")
    assert answer.ok, answer.error
    assert cities(answer.rows) == {"LakePlacid", "SquawValley", "SaltLakeCity", "Chamonix", "Oslo",
                                   "Sapporo"}
    assert [e.stage for e in recorder.events if e.kind == "end"] == [
        "refine", "examples", "generate", "repair", "plan", "price", "execute", "answer"]
    assert answer.spend.total.llm_calls == 1 and answer.spend.total.queries >= 1
    assert answer.plan.rewrites  # the season pattern goes first


def test_the_model_is_asked_again_with_the_problems(olympics):
    llm = ScriptedLLM(["SELECT ?c WHERE { ?c ex:hosted \"Winter\" }", WINTER])
    answer = Hermes(olympics, llm=llm).ask("Which cities hosted the Winter Olympics?")
    assert answer.ok and len(answer.rows) == 6
    assert "links to entities" in llm.prompts[1]
    assert any("the model rewrote the query" in r for r in answer.repairs)


def test_an_empty_answer_is_revised_once(olympics):
    # Every pattern matches something, so the probes pass; together they match nothing.
    empty = "PREFIX ex: <%s> SELECT ?c WHERE { ?c ex:hosted ?g . ?c ex:season \"Winter\" }" % EX
    llm = ScriptedLLM([empty, WINTER])
    answer = Hermes(olympics, llm=llm).ask("Which cities hosted the Winter Olympics?")
    assert len(answer.rows) == 6 and "returns no rows" in llm.prompts[1]


def test_the_budget_stops_a_run_before_it_overspends(olympics):
    answer = Hermes(olympics, llm=ScriptedLLM([WINTER]), budget=Budget(llm_calls=0)).ask(
        "Which cities hosted the Winter Olympics?")
    assert not answer.ok and "over budget at generate" in answer.error
    rows = Hermes(olympics, llm=ScriptedLLM([WINTER]), budget=Budget(rows=1)).ask(
        "Which cities hosted the Winter Olympics?")
    assert "over budget at execute" in rows.error


def test_prices_come_before_spending():
    price = llm_price("claude-haiku-5-5", "x" * 4000, "")
    assert price.input_tokens == 1000 and price.usd > 0
    spend = Spend(Budget(tokens=1500))
    assert spend.affordable(price) and not spend.affordable(price + Price(input_tokens=600))


def test_few_shots_are_chosen_by_their_questions():
    store = FewShotStore([Example("Which cities hosted games?", "SELECT 1"),
                          Example("What is the season of a game?", "SELECT 2")])
    assert FewShotSelector(store, k=1)("Which city hosted the 1980 games?")[0].sparql == "SELECT 1"
    prompt = PromptBuilder().instructions().examples(store.examples).question("Q?").build()
    assert prompt.index("Examples") < prompt.index("Question: Q?")
