"""The prover: the algebra agent with the verifier.

The model first states the goal (what the question asks for), then applies
algebra actions. Before any plan runs, hermes.verifier checks it: scoping,
aggregation rules and schema typing. Ill-typed steps are rejected with the
verifier's message and never reach the graph. Each turn shows the plan state
(typing context and open goals), and `finish` is refused while a goal is open.
"""

from hermes.algebra import Explorer
from hermes.verifier import Verifier
from hermes.systems import register
from hermes.systems.algebra_agent import ACTIONS, AlgebraAgent

GOAL = """\
State the goal first:
  {"op": "goal", "answer": ["?c"], "kind": "entity", "type": "ex:City", "many": true}
  kind is entity, literal, number or boolean; type is optional; many is false for one answer.
"""

RULES = """\
Rules: state the goal first. A verifier checks every step before it runs and
rejects ill-typed ones with a message; read it and fix the step. Use only the
prefixes and terms shown or found with lookup and inspect. Look entities up
before using them. Finish only when the plan state says goals accomplished."""


@register("prover")
class Prover(AlgebraAgent):
    actions = GOAL + ACTIONS
    rules = RULES

    def explorer_for(self, kg):
        return Explorer(kg, sample_size=self.sample_size,
                        verifier=lambda explorer: Verifier.for_kg(kg, explorer.prefixes))

    def state(self, explorer):
        if explorer.goal is None:
            return "Plan state:\n  no goal stated yet"
        judgment = explorer.observe(explorer.plan).judgment
        return "Plan state:\n" + judgment.render(explorer.short)

    def refuse_finish(self, explorer):
        if explorer.goal is None:
            return "refused: state the goal first"
        judgment = explorer.observe(explorer.plan).judgment
        if judgment.errors:
            return "refused: " + "; ".join(judgment.errors)
        if judgment.goals:
            return "refused: open goals remain: " + "; ".join(judgment.goals)
        return None

    def report(self, explorer):
        return {"verifier_rejections": explorer.rejected,
                "goal": explorer.goal.describe() if explorer.goal else ""}
