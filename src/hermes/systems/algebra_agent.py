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
  {"op": "lookup", "text": "Lake Placid"}             entities whose label contains the text
  {"op": "inspect", "iri": "ex:City"}                 predicates around an entity or a class
Change the plan (SPARQL algebra):
  {"op": "triple", "pattern": "?c ex:hosted ?g"}      add a triple pattern (join)
  {"op": "filter", "expr": "?year > 1900"}
  {"op": "optional", "patterns": ["?c ex:country ?k"]}
  {"op": "union", "branches": [["?x a ex:A"], ["?x a ex:B"]]}
  {"op": "minus", "patterns": ["?g ex:season \\"Summer\\""]}
  {"op": "bind", "expr": "YEAR(?date)", "var": "?y"}
  {"op": "group", "vars": ["?c"]}
  {"op": "aggregate", "expr": "COUNT(DISTINCT ?g)", "alias": "?n"}
  {"op": "having", "expr": "?n > 2"}
  {"op": "order", "expr": "DESC(?n)"}
  {"op": "limit", "n": 1}
  {"op": "offset", "n": 1}
  {"op": "project", "vars": ["?c"]}
  {"op": "distinct"}
  {"op": "ask"}                                       yes/no questions
  {"op": "remove", "text": "?c ex:country ?k"}        drop a triple pattern or filter
  {"op": "undo"}
  {"op": "finish"}                                    when the query answers the question"""

RULES = """\
Rules: use only the prefixes and terms shown or found with lookup and inspect.
Look entities up before using them. Project only the variables that answer the
question. Finish as soon as the result looks right."""


@register("algebra-agent")
class AlgebraAgent:
    """Template: subclasses change the explorer, the prompt text, or finishing."""

    actions = ACTIONS
    rules = RULES

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
        explorer = self.explorer_for(kg)
        entities = explorer.linker.link(question)[: self.max_entities]
        trace = []
        for _turn in range(self.max_steps):
            text = prompt(question, kg.vocabulary, explorer, entities, trace, context,
                          self.actions, self.rules, self.state(explorer))
            reply = llm.complete(text, SYSTEM)
            try:
                action = parse_action(reply)
            except ValueError as exc:
                trace.append({"action": reply.strip()[:200], "observation": f"invalid: {exc}"})
                continue
            if isinstance(action, Finish):
                refusal = self.refuse_finish(explorer)
                trace.append({"action": str(action), "observation": refusal or "finished"})
                if refusal:
                    continue
                break
            if trace and trace[-1]["action"] == str(action):
                note = "repeated; its result is above. Choose a different action."
                trace.append({"action": str(action), "observation": note})
                continue
            try:
                observation = explorer.run(action)
            except (TypeError, AttributeError, ValueError, KeyError) as exc:
                trace.append({"action": str(action), "observation": f"invalid: {exc}"})
                continue
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
            **self.report(explorer),
        }

    def explorer_for(self, kg):
        return Explorer(kg, sample_size=self.sample_size)

    def state(self, explorer):
        """Extra prompt text describing the current state."""
        return ""

    def refuse_finish(self, explorer):
        """A reason to keep going, or None to accept finish."""
        return None

    def report(self, explorer):
        return {}


def final_query(explorer):
    """The current plan if it runs, else the last committed plan that did."""
    for plan in reversed(explorer.plans):
        if not plan.empty and explorer.observe(plan).ok:
            return render(plan, explorer.header)
    return None


def prompt(question, vocabulary, explorer, entities, trace, context,
           actions=ACTIONS, rules=RULES, state=""):
    prefixes = "\n".join(f"PREFIX {p}: <{ns}>" for p, ns in explorer.prefixes)
    classes = "\n".join(f"- {item}" for item in vocabulary["classes"]) or "- (none)"
    properties = "\n".join(f"- {item}" for item in vocabulary["properties"]) or "- (none)"
    linked = "\n".join(f"- {explorer.short(iri)}" for iri in entities) or "- (none found)"
    sections = [
        actions,
        rules,
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
    if state:
        sections.append(state)
    current = explorer.sparql if not explorer.plan.empty else "(empty)"
    sections.append(f"Current query:\n{current}\n\nNext action:")
    return "\n\n".join(sections)
