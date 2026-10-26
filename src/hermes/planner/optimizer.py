"""The optimizer: apply rewrite rules to a translated query, and say what changed."""

from dataclasses import dataclass

from hermes.planner.algebra import Query, clone
from hermes.planner.estimate import Estimator
from hermes.planner.rules import DEFAULT_RULES, Context


@dataclass
class Rewrite:
    rule: str
    text: str


@dataclass
class Optimized:
    """A query as written and as rewritten, with the rewrites and their estimates."""

    written: Query
    query: Query
    rewrites: list
    estimator: Estimator

    @property
    def cost_before(self):
        return self.estimator(self.written.root).cost

    @property
    def cost_after(self):
        return self.estimator(self.query.root).cost


class Optimizer:
    def __init__(self, statistics, rules=DEFAULT_RULES):
        self.statistics = statistics
        self.rules = tuple(rules)

    def __call__(self, query):
        context = Context(self.statistics, query)
        root = clone(query.root)
        for rule in self.rules:
            root = rule.apply(root, context)
        rewritten = Query(query.form, query.variables, root, query.namespaces, query.text)
        return Optimized(query, rewritten, [Rewrite(*entry) for entry in context.log],
                         Estimator(self.statistics))
