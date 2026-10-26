"""System R estimates for every node of a plan, from the graph's statistics.

Scans use per-predicate and per-class counts (hermes.pricing.statistics);
joins assume independence and keep 1/max(distinct) of the cross product per
shared variable; selections keep 1/10 for equality with a constant and 1/3
otherwise. The cost of a plan is C_out: the estimated sizes of every scan and
intermediate result, summed (Leis et al., PVLDB 2015, use the same measure).

Scans are priced as the executor runs them. A scan joined after operands
that bind some of its variables is an index access (a bind join): it fetches
about (distinct bindings) x (rows per binding), not the whole pattern, as
long as there are at most BIND_LIMIT bindings to send.
"""

import math
from dataclasses import dataclass

from rdflib import URIRef, Variable
from rdflib.paths import AlternativePath, InvPath, MulPath, NegatedPath, SequencePath

from hermes.planner.algebra import Join, Op, Scan, field
from hermes.pricing.statistics import DEFAULT_ROWS, EQUALITY, RANGE, TYPE, Count, scan


BIND_LIMIT = 500  # a bind join sends at most this many distinct bindings


@dataclass
class Estimate:
    rows: float
    cost: float
    distinct: dict


class Estimator:
    """Estimate of a node, memoized per node."""

    def __init__(self, statistics):
        self.statistics = statistics
        self.memo = {}

    def __call__(self, node):
        if id(node) not in self.memo:
            self.memo[id(node)] = self._estimate(node)
        return self.memo[id(node)]

    def _estimate(self, node):
        if isinstance(node, Scan):
            return self._scan(node)
        if isinstance(node, Join):
            left, right = self(node.left), self(node.right)
            rows, distinct = join(left, right)
            fetched = access(left, right, node.right)
            return Estimate(rows, left.cost + fetched + rows, distinct)
        kids = [self(c) for c in node.children]
        cost = sum(k.cost for k in kids)
        first = kids[0] if kids else Estimate(1.0, 0.0, {})
        kind, part = node.kind, node.part
        if kind == "Filter":
            rows = first.rows * selectivity(part.expr)
            return Estimate(rows, cost, cap(first.distinct, rows))
        if kind == "LeftJoin":
            rows, _distinct = join(kids[0], kids[1])
            rows = max(rows, kids[0].rows)
            return Estimate(rows, cost + rows, {**kids[1].distinct, **kids[0].distinct})
        if kind == "Union":
            rows = kids[0].rows + kids[1].rows
            names = set(kids[0].distinct) | set(kids[1].distinct)
            return Estimate(rows, cost + rows, {v: kids[0].distinct.get(v, 0)
                                                + kids[1].distinct.get(v, 0) for v in names})
        if kind == "Minus":
            return Estimate(first.rows, cost + first.rows, first.distinct)
        if kind == "Distinct":
            values = math.prod(max(d, 1.0) for d in first.distinct.values()) \
                if first.distinct else first.rows
            rows = min(first.rows, values)
            return Estimate(rows, cost, cap(first.distinct, rows))
        if kind == "Slice":
            length = field(part, "length")
            rows = first.rows if length is None else min(first.rows, float(length))
            return Estimate(rows, cost, cap(first.distinct, rows))
        if kind in ("Group", "AggregateJoin"):
            group = node if kind == "Group" else (node.children[0] if node.children else node)
            by = [e for e in (field(group.part, "expr") or []) if isinstance(e, Variable)]
            groups = math.prod(max(first.distinct.get(v, first.rows), 1.0) for v in by) \
                if by else 1.0
            rows = min(first.rows, groups)
            return Estimate(rows, cost, cap(first.distinct, rows))
        if kind == "values":
            return Estimate(float(len(part.res)), 0.0, {})
        if kind == "Project":
            keep = set(part.PV)
            return Estimate(first.rows, cost,
                            {v: d for v, d in first.distinct.items() if v in keep})
        return Estimate(first.rows, cost, dict(first.distinct))

    def _scan(self, node):
        s, p, o = (f"?{t}" if isinstance(t, Variable) else "<const>" for t in node.terms)
        prop = str(node.p) if isinstance(node.p, URIRef) else None
        cls = str(node.o) if prop == TYPE and isinstance(node.o, URIRef) else None
        count = None
        if not isinstance(node.p, (URIRef, Variable)):
            count = path_count(node.p, self.statistics)
        rows, distinct = scan(self.statistics, s, p, o, prop, cls, count=count)
        return Estimate(rows, rows, {Variable(k[1:]): v for k, v in distinct.items()})


def path_count(path, statistics):
    """Triples, distinct subjects and objects of a property path, from its predicates.

    A sequence p/q is the join of p and q on p's objects and q's subjects; an
    alternative p|q sums them; ^p swaps subjects and objects; p* and p+ are
    taken as twice p (one or two steps), and p? as p.
    """
    if isinstance(path, URIRef):
        count = statistics.predicate(str(path))
        return count if count is not None else Count(DEFAULT_ROWS, DEFAULT_ROWS / 10,
                                                     DEFAULT_ROWS / 10)
    if isinstance(path, InvPath):
        inner = path_count(path.arg, statistics)
        return Count(inner.triples, inner.objects, inner.subjects)
    if isinstance(path, SequencePath):
        counts = [path_count(arg, statistics) for arg in path.args]
        total = counts[0]
        for step in counts[1:]:
            triples = total.triples * step.triples / max(total.objects, step.subjects, 1.0)
            total = Count(triples, min(total.subjects, triples), min(step.objects, triples))
        return total
    if isinstance(path, AlternativePath):
        counts = [path_count(arg, statistics) for arg in path.args]
        return Count(sum(c.triples for c in counts), sum(c.subjects for c in counts),
                     sum(c.objects for c in counts))
    if isinstance(path, MulPath):
        inner = path_count(path.path, statistics)
        grow = 1.0 if path.mod == "?" else 2.0
        return Count(inner.triples * grow, inner.subjects * grow, inner.objects * grow)
    if isinstance(path, NegatedPath):
        total = statistics.everything()
        return total if total is not None else Count(DEFAULT_ROWS * 100, DEFAULT_ROWS * 10,
                                                     DEFAULT_ROWS * 10)
    return Count(DEFAULT_ROWS, DEFAULT_ROWS / 10, DEFAULT_ROWS / 10)


def scan_below(node):
    """The scan under a chain of selections, or None: what a bind join can index."""
    while isinstance(node, Op) and node.kind == "Filter":
        node = node.children[0]
    return node if isinstance(node, Scan) else None


def access(left, right, node):
    """What joining `node` (estimated as `right`) after `left` fetches from the store.

    A scan, or selections over one, that shares variables with the left side
    is fetched by binding: about (distinct bindings) x (rows per binding),
    capped at the whole pattern. Anything else is fetched whole.
    """
    shared = set(left.distinct) & set(right.distinct)
    if scan_below(node) is None or not shared:
        return right.cost
    keys = min(left.rows, math.prod(max(left.distinct[v], 1.0) for v in shared))
    if keys > BIND_LIMIT:
        return right.cost
    per = right.rows / max(max(right.distinct[v] for v in shared), 1.0)
    return min(right.cost, max(keys, 1.0) * per)


def selectivity(expr):
    """System R's guesses: equality with a constant keeps 1/10, anything else 1/3."""
    name = getattr(expr, "name", "")
    if name == "RelationalExpression" and expr.op == "=":
        return EQUALITY
    if name == "ConditionalAndExpression":
        return math.prod(selectivity(e) for e in [expr.expr, *(field(expr, "other") or [])])
    if name == "ConditionalOrExpression":
        return min(1.0, sum(selectivity(e) for e in [expr.expr, *(field(expr, "other") or [])]))
    if name in ("Builtin_sameTerm", "Builtin_REGEX"):
        return EQUALITY
    return RANGE


def join(left, right):
    """Rows and distinct values of the natural join of two estimates."""
    shared = set(left.distinct) & set(right.distinct)
    rows = left.rows * right.rows
    for var in shared:
        rows /= max(left.distinct[var], right.distinct[var], 1.0)
    distinct = {}
    for var in set(left.distinct) | set(right.distinct):
        values = [d[var] for d in (left.distinct, right.distinct) if var in d]
        distinct[var] = min(rows, *values)
    return rows, distinct


def cap(distinct, rows):
    return {v: min(d, rows) for v, d in distinct.items()}
