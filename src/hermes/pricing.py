"""Query pricing: what a plan will cost, estimated before it is sent.

A database optimizer costs a plan before it runs one; Hermes prices every
query a planner might send the same way, so a budget can be spent where it is
most useful and a query that would explode is never sent. Estimates follow
System R (Selinger et al., SIGMOD 1979):

- statistics per predicate (triples, distinct subjects, distinct objects) and
  per class (instances), read from the graph itself;
- a triple pattern with a constant keeps 1/distinct of its predicate's
  triples; patterns are independent, and a join on a shared variable keeps
  1/max(distinct) of the cross product;
- filters keep a fixed fraction (1/10 for equality, 1/3 otherwise).

Patterns are joined greedily, smallest first and connected before cartesian,
as engines order a BGP. The cost of a plan is C_out: the estimated sizes of
every scan and intermediate result, summed. Seconds come from a line fitted
online to (C_out, measured seconds) pairs, so the price adapts to the store.

A local graph is summarized once with grouped queries. An endpoint such as
DBpedia is asked about each predicate or class the first time a plan uses it,
and the answer is kept; a term the endpoint cannot count is priced with a
default and reported as unknown.
"""

import math
import re
import threading
import weakref
from dataclasses import dataclass, field

from hermes.verifier import TYPE, split_triples

EQUALITY = 0.1
RANGE = 1 / 3
DEFAULT_ROWS = 10_000.0  # a predicate or class the store could not count
_STATS = weakref.WeakKeyDictionary()
_STATS_LOCK = threading.Lock()


@dataclass
class Count:
    triples: float
    subjects: float
    objects: float


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


class Statistics:
    """Counts per predicate and per class, eager for local graphs, lazy for endpoints."""

    def __init__(self, kg, eager=None):
        self.kg = kg
        self.predicates = {}
        self.classes = {}
        self.total = None
        self.complete = False  # every predicate and class is known
        self._lock = threading.Lock()
        if kg.store.local if eager is None else eager:
            self._summarize()

    def _summarize(self):
        rows = self.kg.try_query(
            "SELECT ?p (COUNT(*) AS ?n) (COUNT(DISTINCT ?s) AS ?ds) (COUNT(DISTINCT ?o) AS ?do) "
            "WHERE { ?s ?p ?o } GROUP BY ?p"
        )["results"]["bindings"]
        for row in rows:
            self.predicates[row["p"]["value"]] = Count(
                _number(row, "n"), _number(row, "ds"), _number(row, "do"))
        rows = self.kg.try_query(
            f"SELECT ?c (COUNT(DISTINCT ?x) AS ?n) WHERE {{ ?x <{TYPE}> ?c }} GROUP BY ?c"
        )["results"]["bindings"]
        for row in rows:
            self.classes[row["c"]["value"]] = _number(row, "n")
        rows = self.kg.try_query(
            "SELECT (COUNT(*) AS ?n) (COUNT(DISTINCT ?s) AS ?ds) (COUNT(DISTINCT ?o) AS ?do) "
            "WHERE { ?s ?p ?o }"
        )["results"]["bindings"]
        self.total = Count(_number(rows[0], "n"), _number(rows[0], "ds"),
                           _number(rows[0], "do")) if rows else Count(0, 0, 0)
        self.complete = True

    def predicate(self, iri):
        """Count for a predicate IRI, or None if the store could not count it."""
        with self._lock:
            if iri in self.predicates or self.complete:
                return self.predicates.get(iri, Count(0, 0, 0) if self.complete else None)
        rows = self.kg.try_query(
            f"SELECT (COUNT(*) AS ?n) (COUNT(DISTINCT ?s) AS ?ds) (COUNT(DISTINCT ?o) AS ?do) "
            f"WHERE {{ ?s <{iri}> ?o }}"
        )["results"]["bindings"]
        count = Count(_number(rows[0], "n"), _number(rows[0], "ds"),
                      _number(rows[0], "do")) if rows and "n" in rows[0] else None
        with self._lock:
            self.predicates[iri] = count
        return count

    def instances(self, iri):
        with self._lock:
            if iri in self.classes or self.complete:
                return self.classes.get(iri, 0.0 if self.complete else None)
        rows = self.kg.try_query(
            f"SELECT (COUNT(*) AS ?n) WHERE {{ ?x <{TYPE}> <{iri}> }}"
        )["results"]["bindings"]
        count = _number(rows[0], "n") if rows and "n" in rows[0] else None
        with self._lock:
            self.classes[iri] = count
        return count

    def everything(self):
        """Count over all triples; None for an endpoint, where it is too costly to ask."""
        return self.total


def statistics_for(kg):
    """One Statistics per KnowledgeGraph, shared by every pricer that uses it."""
    with _STATS_LOCK:
        stats = _STATS.get(kg)
        if stats is None:
            stats = _STATS[kg] = Statistics(kg)
        return stats


class SecondsModel:
    """seconds = a + b * C_out, fitted by least squares to observed executions."""

    def __init__(self, intercept=0.005, slope=1e-6, minimum=3):
        self.prior = (intercept, slope)
        self.minimum = minimum
        self.points = []
        self._lock = threading.Lock()

    def observe(self, cost, seconds):
        with self._lock:
            self.points.append((float(cost), float(seconds)))

    def coefficients(self):
        with self._lock:
            points = list(self.points)
        if len(points) < self.minimum:
            return self.prior
        n = len(points)
        mean_x = sum(x for x, _ in points) / n
        mean_y = sum(y for _, y in points) / n
        spread = sum((x - mean_x) ** 2 for x, _ in points)
        if spread == 0:
            return (mean_y, 0.0)
        slope = max(0.0, sum((x - mean_x) * (y - mean_y) for x, y in points) / spread)
        return (max(0.0, mean_y - slope * mean_x), slope)

    def predict(self, cost):
        intercept, slope = self.coefficients()
        return intercept + slope * cost


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
        stats = self.statistics
        if predicate == "a":
            prop = TYPE
        elif _variable(predicate) or _path(predicate):
            prop = None
        else:
            prop = self.expand(predicate)
        if prop == TYPE and not _variable(value):
            cls = self.expand(value)
            count = stats.instances(cls)
            if count is None:
                unknown.append(value)
                count = DEFAULT_ROWS
            rows = count if _variable(subject) else min(1.0, count)
            return rows, _distinct({subject: rows})
        if prop is None:
            total = stats.everything()
            if total is None:
                unknown.append(predicate)
                total = Count(DEFAULT_ROWS * 100, DEFAULT_ROWS * 10, DEFAULT_ROWS * 10)
            count = total
        else:
            count = stats.predicate(prop)
            if count is None:
                unknown.append(predicate)
                count = Count(DEFAULT_ROWS, DEFAULT_ROWS / 10, DEFAULT_ROWS / 10)
        rows = count.triples
        if not _variable(subject):
            rows /= max(count.subjects, 1.0)
        if not _variable(value):
            rows /= max(count.objects, 1.0)
        found = {}
        if _variable(subject):
            found[subject] = min(rows, count.subjects)
        if _variable(predicate):
            found[predicate] = min(rows, max(1.0, len(stats.predicates)))
        if _variable(value):
            found[value] = min(rows, count.objects)
        return rows, _distinct(found)


def qerror(estimated, actual):
    """max(est/act, act/est), with both floored at 1: the standard estimate error."""
    estimated, actual = max(float(estimated), 1.0), max(float(actual), 1.0)
    return max(estimated / actual, actual / estimated)


def _join(rows, distinct, other_rows, other):
    shared = set(distinct) & set(other)
    joined = rows * other_rows
    for var in shared:
        joined /= max(distinct[var], other[var], 1.0)
    merged = {}
    for var in set(distinct) | set(other):
        merged[var] = min(joined, distinct.get(var, other.get(var)), other.get(var, distinct.get(var)))
    return joined, merged


def _distinct(found):
    return {var: max(value, 0.0) for var, value in found.items()}


def _selectivity(expr):
    if re.search(r"(?<![!<>])=(?!=)", expr) or re.match(r"\s*(sameTerm|regex)\b", expr, re.I):
        return EQUALITY
    return RANGE


def _variable(token):
    return bool(token) and token[0] in "?$"


def _path(token):
    return any(mark in token for mark in "/|^*+") and not token.startswith("<")


def _number(row, name):
    try:
        return float(row[name]["value"])
    except (KeyError, ValueError):
        return 0.0
