"""The algebra agent: an LLM builds a query one algebra operator at a time.

Each turn the model sees the question, the vocabulary, the current plan as
SPARQL, and what every previous action did to the result (counts, samples,
errors). It answers with one action from hermes.algebra. The loop ends on
`finish` or after `max_steps`, and returns the last plan that ran.
"""

from hermes import llm as llms
from hermes.algebra import Explorer, Finish, parse_action, render
from hermes.context import as_context
from hermes.knowledge import as_knowledge_graph
from hermes.systems import register

SYSTEM = (
    "You build SPARQL queries by applying one algebra operator at a time and "
    "reading the result of each step. Reply with exactly one JSON object."
)

ACTIONS = """\
Actions. Reply with exactly one JSON object per turn.
Read the graph (the plan does not change):
  {"op": "lookup", "text": "Karen Brant"}             entities whose label contains the text
  {"op": "inspect", "iri": "ex:Employee"}             predicates around an entity or a class
Change the plan (SPARQL algebra):
  {"op": "triple", "pattern": "?e ex:memberOf ?d"}    add a triple pattern (join)
  {"op": "filter", "expr": "?price > 100"}
  {"op": "optional", "patterns": ["?e ex:phone ?p"]}
  {"op": "union", "branches": [["?x a ex:A"], ["?x a ex:B"]]}
  {"op": "minus", "patterns": ["?x ex:retired true"]}
  {"op": "bind", "expr": "YEAR(?date)", "var": "?year"}
  {"op": "group", "vars": ["?d"]}
  {"op": "aggregate", "expr": "COUNT(DISTINCT ?e)", "alias": "?n"}
  {"op": "having", "expr": "?n > 2"}
  {"op": "order", "expr": "DESC(?n)"}
  {"op": "limit", "n": 1}
  {"op": "offset", "n": 1}
  {"op": "project", "vars": ["?d"]}
  {"op": "distinct"}
  {"op": "ask"}                                       yes/no questions
  {"op": "remove", "text": "?e ex:phone ?p"}          drop a triple pattern or filter
  {"op": "undo"}
  {"op": "finish"}                                    when the query answers the question"""

RULES = """\
Rules: use only the prefixes and terms shown or found with lookup and inspect.
Look entities up before using them. Project only the variables that answer the
question. Finish as soon as the result looks right."""


@register("algebra-agent")
class AlgebraAgent:
    def __init__(self, llm=None, max_steps=15, sample_size=5, max_entities=10):
        self.llm = llm
        self.max_steps = max_steps
        self.sample_size = sample_size
        self.max_entities = max_entities

    def __call__(self, question, kg, context=None):
        question = (question or "").strip()
        if not question:
            raise ValueError("A question is required.")
        llm = self.llm or llms.from_env()
        context = as_context(context)
        kg = as_knowledge_graph(kg)
        explorer = Explorer(kg, sample_size=self.sample_size)
        entities = explorer.linker.link(question)[: self.max_entities]
        trace = []
        for _turn in range(self.max_steps):
            reply = llm.complete(
                prompt(question, kg.vocabulary, explorer, entities, trace, context), SYSTEM
            )
            try:
                action = parse_action(reply)
            except ValueError as exc:
                trace.append({"action": reply.strip()[:200], "observation": f"invalid: {exc}"})
                continue
            if isinstance(action, Finish):
                trace.append({"action": str(action), "observation": "finished"})
                break
            if trace and trace[-1]["action"] == str(action):
                note = "repeated; its result is above. Choose a different action."
                trace.append({"action": str(action), "observation": note})
                continue
            observation = explorer.run(action)
            trace.append({"action": str(action), "observation": observation.summary()})
        sparql = final_query(explorer)
        if sparql is None:
            raise ValueError("The agent did not build a query that runs.")
        return {
            "sparql": sparql,
            "trace": trace,
            "steps": len(trace),
            "model": llm.model,
            "provider": llm.provider,
        }


def final_query(explorer):
    """The current plan if it runs, else the last committed plan that did."""
    for plan in reversed(explorer.plans):
        if not plan.empty and explorer.observe(plan).ok:
            return render(plan, explorer.header)
    return None


def prompt(question, vocabulary, explorer, entities, trace, context):
    prefixes = "\n".join(f"PREFIX {p}: <{ns}>" for p, ns in explorer.prefixes)
    classes = "\n".join(f"- {item}" for item in vocabulary["classes"]) or "- (none)"
    properties = "\n".join(f"- {item}" for item in vocabulary["properties"]) or "- (none)"
    linked = "\n".join(f"- {explorer.short(iri)}" for iri in entities) or "- (none found)"
    sections = [
        ACTIONS,
        RULES,
        f"Prefixes:\n{prefixes}",
        f"Classes:\n{classes}",
        f"Properties:\n{properties}",
        f"Entities whose label appears in the question:\n{linked}",
    ]
    if context.text.strip():
        sections.append(f"Context:\n{context.text.strip()}")
    if context.examples:
        sections.append(
            "Example questions:\n\n"
            + "\n\n".join(f"Question: {e.nl}\nQuery:\n{e.sparql}" for e in context.examples)
        )
    sections.append(f"Question: {question}")
    if trace:
        steps = "\n".join(
            f"{number}. {step['action']}\n   -> {step['observation']}"
            for number, step in enumerate(trace, 1)
        )
        sections.append(f"Steps so far:\n{steps}")
    current = explorer.sparql if not explorer.plan.empty else "(empty)"
    sections.append(f"Current query:\n{current}\n\nNext action:")
    return "\n\n".join(sections)
