"""The prover: the algebra agent with a kernel, in the style of Lean.

The model first states the goal (what the question asks for), then applies
algebra actions as tactics. Before any plan runs, hermes.kernel checks it:
scoping, aggregation rules and schema typing. Ill-typed steps are rejected
with the kernel's message and never reach the graph. Each turn shows the proof
state (typing context and open goals), and `finish` is refused while a goal
is open.
"""

from hermes.algebra import Explorer
from hermes.kernel import Kernel
from hermes.systems import register
from hermes.systems.algebra_agent import ACTIONS, AlgebraAgent

GOAL = """\
State the goal first, like a theorem:
  {"op": "goal", "answer": ["?d"], "kind": "entity", "type": "ex:Department", "many": true}
  kind is entity, literal, number or boolean; type is optional; many is false for one answer.
"""

RULES = """\
Rules: state the goal first. A kernel checks every step before it runs and
rejects ill-typed ones with a message; read it and fix the step. Use only the
prefixes and terms shown or found with lookup and inspect. Look entities up
before using them. Finish only when the proof state says goals accomplished."""


@register("prover")
class Prover(AlgebraAgent):
    actions = GOAL + ACTIONS
    rules = RULES

    def explorer_for(self, kg):
        return Explorer(kg, sample_size=self.sample_size,
                        kernel=lambda explorer: Kernel.for_kg(kg, explorer.prefixes))

    def state(self, explorer):
        if explorer.goal is None:
            return "Proof state:\n  no goal stated yet"
        judgment = explorer.observe(explorer.plan).judgment
        return "Proof state:\n" + judgment.render(explorer.short)

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
        return {"kernel_rejections": explorer.rejected,
                "goal": explorer.goal.describe() if explorer.goal else ""}
