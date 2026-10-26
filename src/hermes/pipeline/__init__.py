"""Hermes: ontology-based natural-language querying over knowledge graphs.

    hermes = Hermes(kg, llm=..., budget=Budget(llm_calls=3, rows=50_000))
    answer = hermes.ask("Which department does Ms. Brant belong to?")

A question goes through the stages of NL2KQL (Tang et al., arXiv 2404.02933),
with the ontology as the schema and the knowledge graph as the data, then
through a query planner:

1. refine: the ontology slice the question needs (hermes.pipeline.refiner);
2. examples: the few-shot examples closest to it (hermes.pipeline.fewshot);
3. generate: one model call writes the query from the prompt (hermes.pipeline.prompt);
4. repair: the query refiner makes it parse, name known terms, and type-check
   (hermes.pipeline.repair), asking the model again only while the budget allows;
5. plan: the query is translated into relational algebra and rewritten with
   the graph's statistics (hermes.planner);
6. price: the plan's queries and rows are estimated and checked against the budget;
7. execute: the plan runs (hermes.planner.executor).

Each stage reports to the observers as it starts and ends, and every model
call and graph query is priced before it is spent (hermes.pricing.budget).
"""

import time
from dataclasses import dataclass, field

from hermes.graph.knowledge import as_knowledge_graph
from hermes.pipeline.catalog import catalog_for
from hermes.pipeline.events import Event, Recorder
from hermes.pipeline.fewshot import FewShotSelector, FewShotStore
from hermes.pipeline.names import Prefixes
from hermes.pipeline.prompt import PromptBuilder
from hermes.pipeline.refiner import OntologyRefiner
from hermes.pipeline.repair import QueryRefiner
from hermes.planner import Optimizer, UnsupportedQuery, executor_for, translate
from hermes.pricing.budget import Budget, BudgetedLLM, BudgetExceeded, Price, Spend, plan_price
from hermes.pricing.statistics import statistics_for
from hermes.retrieval import retriever as make_retriever
from hermes.sparql.validate import extract_sparql

SYSTEM = "You write SPARQL queries over a knowledge graph. Return only the query."


@dataclass
class Answer:
    question: str
    sparql: str = ""
    variables: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    boolean: bool = None
    plan: object = None  # hermes.planner.Optimized
    slice: object = None  # hermes.pipeline.refiner.OntologySlice
    examples: list = field(default_factory=list)
    repairs: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    spend: Spend = None
    error: str = ""
    seconds: float = 0.0

    @property
    def ok(self):
        return not self.error


@dataclass
class State:
    """What the stages pass along."""

    question: str
    answer: Answer
    prompt: PromptBuilder = None
    draft: str = ""
    query: object = None  # hermes.planner.Query, translated


class Stage:
    name = ""

    def run(self, state, hermes):
        raise NotImplementedError


class Refine(Stage):
    name = "refine"

    def run(self, state, hermes):
        state.answer.slice = hermes.refiner(state.question)
        return {"classes": len(state.answer.slice.classes),
                "properties": len(state.answer.slice.properties),
                "values": len(state.answer.slice.values),
                "entities": [e.label for e in state.answer.slice.entities]}


class Examples(Stage):
    name = "examples"

    def run(self, state, hermes):
        state.answer.examples = hermes.selector(state.question)
        return {"examples": [e.question for e in state.answer.examples]}


class Generate(Stage):
    name = "generate"

    def run(self, state, hermes):
        state.prompt = (PromptBuilder().instructions()
                        .prefixes(hermes.names.header())
                        .ontology(state.answer.slice.render(hermes.catalog, hermes.names))
                        .examples(state.answer.examples))
        prompt = PromptBuilder()
        prompt.parts = dict(state.prompt.parts)
        text = prompt.question(state.question).build()
        reply = hermes.model(state.answer.spend, "generate").complete(text, SYSTEM)
        state.draft = extract_sparql(reply)
        return {"prompt": text, "reply": reply, "sparql": state.draft}


class Repair(Stage):
    name = "repair"

    def run(self, state, hermes):
        base = PromptBuilder()
        base.parts = dict(state.prompt.parts)
        base.question(state.question)
        refiner = QueryRefiner(hermes.catalog, hermes.names,
                               hermes.model(state.answer.spend, "repair"), state.answer.spend,
                               hermes.max_repairs)
        refined = refiner(state.draft, base)
        state.answer.sparql = refined.sparql
        state.answer.repairs, state.answer.problems = refined.repairs, refined.problems
        return {"sparql": refined.sparql, "repairs": refined.repairs, "problems": refined.problems}


class Plan(Stage):
    name = "plan"

    def run(self, state, hermes):
        query = translate(state.answer.sparql)
        optimizer = Optimizer(hermes.statistics) if hermes.optimize else Optimizer(hermes.statistics, ())
        state.answer.plan = optimized = optimizer(query)
        return {"rewrites": [f"{r.rule}: {r.text}" for r in optimized.rewrites],
                "cost": [round(optimized.cost_before, 1), round(optimized.cost_after, 1)]}


class CheckBudget(Stage):
    name = "price"

    def run(self, state, hermes):
        price = plan_price(state.answer.plan)
        state.answer.spend.check("execute", price)
        return {"estimated": price.to_dict()}


class Execute(Stage):
    """Run the plan. An empty answer is sent back to the model once, while the budget allows."""

    name = "execute"
    EMPTY = ("it returns no rows. Check that each property is used from its subject to its "
             "object as listed, that entities are written with their listed IRIs, and that "
             "every pattern is needed")

    def run(self, state, hermes):
        answer = state.answer
        result = self.execute(answer, hermes)
        revised = 0
        while (result.form == "SELECT" and not result.rows and revised < hermes.revisions):
            if not self.revise(state, hermes):
                break
            revised += 1
            result = self.execute(answer, hermes)
        answer.variables, answer.rows, answer.boolean = result.variables, result.rows, result.boolean
        trace = result.trace
        return {"rows": len(result.rows), "queries": trace.queries, "fetched": trace.fetched,
                "seconds": round(trace.seconds, 4), "revisions": revised}

    def execute(self, answer, hermes):
        estimated = plan_price(answer.plan)
        answer.spend.check("execute", estimated)
        result = executor_for(hermes.executor, hermes.kg)(answer.plan.query)
        answer.spend.record("execute", estimated,
                            Price(queries=result.trace.queries, rows=result.trace.fetched))
        return result

    def revise(self, state, hermes):
        """Ask the model for a query that answers, then repair and plan it again."""
        answer = state.answer
        base = PromptBuilder()
        base.parts = dict(state.prompt.parts)
        base.question(state.question)
        refiner = QueryRefiner(hermes.catalog, hermes.names,
                               hermes.model(answer.spend, "revise"), answer.spend, hermes.max_repairs)
        reply = refiner.ask(answer.sparql, [self.EMPTY], base)
        if not reply:
            return False
        refined = refiner(reply, base)
        if refined.problems:
            return False
        try:
            optimized = Optimizer(hermes.statistics, () if not hermes.optimize else
                                  Optimizer(hermes.statistics).rules)(translate(refined.sparql))
        except Exception:
            return False
        answer.repairs.append("the model revised the query, which returned no rows")
        answer.repairs.extend(refined.repairs)
        answer.sparql, answer.plan = refined.sparql, optimized
        return True


STAGES = (Refine(), Examples(), Generate(), Repair(), Plan(), CheckBudget(), Execute())


class Hermes:
    """The pipeline over one knowledge graph; components are injected, with defaults."""

    def __init__(self, kg, llm=None, budget=None, fewshots=None, retriever="lexical",
                 examples=2, executor="plan", optimize=True, max_repairs=2, revisions=1,
                 observers=(), stages=STAGES):
        from hermes import llm as llms

        self.kg = as_knowledge_graph(kg)
        self.llm = llm or llms.from_env()
        self.budget = budget or Budget()
        self.catalog = catalog_for(self.kg)
        self.statistics = statistics_for(self.kg)
        self.names = Prefixes.of(self.kg)
        retrieve = (lambda: make_retriever(retriever)) if isinstance(retriever, str) else (lambda: retriever)
        self.refiner = OntologyRefiner(self.catalog, retrieve())
        store = fewshots if isinstance(fewshots, FewShotStore) else FewShotStore.load(fewshots) \
            if fewshots else FewShotStore()
        self.selector = FewShotSelector(store, examples, retrieve())
        self.executor = executor
        self.optimize = optimize
        self.max_repairs = max_repairs
        self.revisions = revisions
        self.observers = list(observers)
        self.stages = tuple(stages)

    def model(self, spend, stage):
        return BudgetedLLM(self.llm, spend, stage)

    def ask(self, question):
        """Template method: run the stages in order; a failure ends the run with its reason."""
        answer = Answer(question.strip(), spend=Spend(self.budget))
        state = State(answer.question, answer)
        started = time.perf_counter()
        for stage in self.stages:
            self.emit(Event(stage.name, "start"))
            try:
                data = stage.run(state, self)
            except BudgetExceeded as exc:
                answer.error = f"over budget at {exc.stage}: {exc.reason}"
            except UnsupportedQuery as exc:
                answer.error = f"unsupported query: {exc}"
            except Exception as exc:  # a store, model or parse failure: report it, do not crash
                answer.error = f"{stage.name} failed: {type(exc).__name__}: {exc}"
            if answer.error:
                self.emit(Event(stage.name, "error", {"error": answer.error}))
                break
            self.emit(Event(stage.name, "end", data))
        answer.seconds = time.perf_counter() - started
        self.emit(Event("answer", "end", {"ok": answer.ok, "rows": len(answer.rows),
                                          "spend": answer.spend.to_dict()}))
        return answer

    def emit(self, event):
        for observer in self.observers:
            observer.on_event(event)


__all__ = ["Answer", "Budget", "BudgetExceeded", "Event", "Hermes", "Recorder", "STAGES", "Stage"]
