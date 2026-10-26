"""SPARQL as relational algebra: translate, estimate, rewrite, execute.

    from hermes.planner import plan, PlanExecutor
    optimized = plan(sparql, kg)          # translated, then rewritten
    result = PlanExecutor(kg)(optimized.query)
"""

from hermes.planner.algebra import Join, Op, Query, Scan, UnsupportedQuery, translate
from hermes.planner.estimate import Estimate, Estimator
from hermes.planner.executor import (
    EngineExecutor, Executor, PlanExecutor, Result, Trace, executor_for, same_answers,
)
from hermes.planner.optimizer import Optimized, Optimizer, Rewrite
from hermes.planner.render import expression, formula, label, tree
from hermes.planner.rules import DEFAULT_RULES, OrderJoins, PruneScans, PushSelections, Rule
from hermes.pricing.statistics import statistics_for


def plan(sparql, kg, rules=DEFAULT_RULES):
    """Translate a query and rewrite it with the graph's statistics."""
    return Optimizer(statistics_for(kg), rules)(translate(sparql))


__all__ = [
    "DEFAULT_RULES", "EngineExecutor", "Estimate", "Estimator", "Executor", "Join", "Op",
    "Optimized", "Optimizer", "OrderJoins", "PlanExecutor", "PruneScans", "PushSelections",
    "Query", "Result", "Rewrite", "Rule", "Scan", "Trace", "UnsupportedQuery", "executor_for",
    "expression", "formula", "label", "plan", "same_answers", "translate", "tree",
]
