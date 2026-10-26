"""Pricing for the budgeted planner's flat plans (hermes.baselines.explorer.Plan).

Estimates rows and C_out of a plan before it runs, from the graph's statistics
(hermes.pricing.statistics); seconds come from a line fitted online.
"""

import math
import re
from dataclasses import dataclass, field

from hermes.pricing.seconds import SecondsModel
from hermes.pricing.statistics import (
    EQUALITY, RANGE, TYPE, _distinct, _path, _variable, qerror, scan, statistics_for,
)
from hermes.sparql.verifier import split_triples

@dataclass
class Estimate:
    """The priced plan: rows it returns, C_out, and seconds once fitted."""

    rows: float
    cost: float
    seconds: float | None = None
    unknown: list = field(default_factory=list)

    def to_dict(self):
        return {"rows": round(self.rows, 2), "cost": round(self.cost, 1),
                "seconds": None if self.seconds is None else round(self.seconds, 4),
                "unknown": self.unknown}


class Pricer:
    """Estimate rows, C_out and seconds of a plan; learn seconds from executions."""

    def __init__(self, statistics, expand, seconds=None):
        self.statistics = statistics
        self.expand = expand
        self.seconds = seconds or SecondsModel()
        self.errors = []  # q-errors of row estimates, for reporting

    @classmethod
    def for_kg(cls, kg, expand):
        intercept = 0.005 if kg.store.local else 0.3
        return cls(statistics_for(kg), expand, SecondsModel(intercept=intercept))

    def estimate(self, plan):
        unknown = []
        rows, cost, distinct = self._bgp(plan.triples, unknown)
        for group in plan.optionals:
            extra_rows, extra_cost, extra = self._bgp(group, unknown)
            joined, distinct = _join(rows, distinct, extra_rows, extra)
            cost += extra_cost + joined
            rows = max(rows, joined)
        for branches in plan.unions:
            union_rows, union_distinct = 0.0, {}
            for branch in branches:
                branch_rows, branch_cost, branch_distinct = self._bgp(branch, unknown)
                cost += branch_cost
                union_rows += branch_rows
                for var, value in branch_distinct.items():
                    union_distinct[var] = union_distinct.get(var, 0.0) + value
            if plan.triples or rows:
                rows, distinct = _join(rows, distinct, union_rows, union_distinct)
                cost += rows
            else:
                rows, distinct = union_rows, union_distinct
        for group in plan.minus:
            cost += self._bgp(group, unknown)[1]
        for expr in plan.filters:
            rows *= _selectivity(expr)
        if plan.group_by:
            groups = math.prod(distinct.get(v, rows) for v in plan.group_by)
            rows = min(rows, groups)
        elif plan.aggregates:
            rows = min(rows, 1.0)
        for _expr in plan.having:
            rows *= RANGE
        if plan.limit is not None:
            rows = min(rows, float(plan.limit))
        if plan.form == "ASK":
            rows = min(rows, 1.0)
        return Estimate(rows, cost, self.seconds.predict(cost), sorted(set(unknown)))

    def nominal(self, estimate):
        """A price for ranking: seconds at the prior rate, from C_out alone."""
        return self.seconds.nominal(estimate.cost)

    def observe(self, estimate, seconds, rows=None):
        """Record a real execution: fit seconds, and the q-error of the row estimate."""
        self.seconds.observe(estimate.cost, seconds)
        if rows is not None:
            self.errors.append(qerror(estimate.rows, rows))

    def _bgp(self, patterns, unknown):
        """Rows, C_out and distinct values per variable of a group of triple patterns."""
        scans = []
        for pattern in patterns:
            for triple in split_triples(pattern)[0]:
                scans.append(self._scan(*triple, unknown))
        if not scans:
            return 0.0, 0.0, {}
        scans.sort(key=lambda scan: scan[0])
        rows, distinct = scans[0]
        cost = rows
        remaining = scans[1:]
        while remaining:
            # Joined next: the smallest scan sharing a variable, else the smallest.
            connected = [s for s in remaining if set(s[1]) & set(distinct)]
            scan = (connected or remaining)[0]
            remaining.remove(scan)
            cost += scan[0]
            rows, distinct = _join(rows, distinct, *scan)
            cost += rows
        return rows, cost, distinct

    def _scan(self, subject, predicate, value, unknown):
        """Rows of one triple pattern and distinct values of its variables."""
        if predicate == "a":
            prop = TYPE
        elif _variable(predicate) or _path(predicate):
            prop = None
        else:
            prop = self.expand(predicate)
        cls = self.expand(value) if prop == TYPE and not _variable(value) else None
        return scan(self.statistics, subject, predicate, value, prop, cls, unknown)


def _join(rows, distinct, other_rows, other):
    shared = set(distinct) & set(other)
    joined = rows * other_rows
    for var in shared:
        joined /= max(distinct[var], other[var], 1.0)
    merged = {}
    for var in set(distinct) | set(other):
        merged[var] = min(joined, distinct.get(var, other.get(var)), other.get(var, distinct.get(var)))
    return joined, merged


def _selectivity(expr):
    if re.search(r"(?<![!<>])=(?!=)", expr) or re.match(r"\s*(sameTerm|regex)\b", expr, re.I):
        return EQUALITY
    return RANGE
