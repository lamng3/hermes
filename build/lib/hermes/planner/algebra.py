"""SPARQL as relational algebra: plans, and the translation from SPARQL text.

A triple pattern is a relation over its variables, a basic graph pattern is
the natural join of its patterns, and every other SPARQL operator is a
relational one (W3C SPARQL 1.1, section 18; Pérez et al., TODS 2009):

    triple pattern   scan              FILTER          σ  selection
    join, BGP        ⋈                 OPTIONAL        ⟕  left outer join
    SELECT vars      π  projection     UNION, MINUS    ∪, −
    BIND             ε  extension      DISTINCT        δ
    ORDER BY         τ                 LIMIT, OFFSET   λ  slice
    GROUP BY         γ

`translate` keeps the query as written: its patterns are joined left to
right, so that every reordering is the optimizer's and can be measured.
"""

from dataclasses import dataclass, field as dataclass_field

from rdflib import Variable
from rdflib.plugins.sparql import algebra as rdflib_algebra
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.parserutils import CompValue

from hermes.sparql.validate import _PARSE_LOCK


@dataclass(eq=False)
class Scan:
    """One triple pattern. `keep` is the set of variables fetched, once pruned."""

    s: object
    p: object
    o: object
    keep: frozenset = None

    @property
    def terms(self):
        return (self.s, self.p, self.o)

    @property
    def bound(self):
        return {t for t in self.terms if isinstance(t, Variable)}

    @property
    def out(self):
        return self.bound if self.keep is None else set(self.keep)


@dataclass(eq=False)
class Join:
    left: object
    right: object


@dataclass(eq=False)
class Op:
    """Any other operator: rdflib's algebra node `part`, over our `children`.

    `keys` says where each child sits in `part` (p, or p1 and p2), so that the
    executor can hand rdflib the node with its children's rows in their place.
    """

    kind: str
    part: CompValue
    children: list = dataclass_field(default_factory=list)
    keys: tuple = ()


@dataclass
class Query:
    form: str  # SELECT or ASK
    variables: list
    root: object
    namespaces: object
    text: str


class UnsupportedQuery(ValueError):
    """The query uses a form or operator the planner does not handle."""


def field(part, name, default=None):
    """A field of an rdflib algebra node; CompValue.get returns the name itself when missing."""
    return part[name] if name in part else default


def translate(sparql):
    """The query as relational algebra, with its patterns in the order written."""
    with _PARSE_LOCK:
        reorder = rdflib_algebra.reorderTriples
        rdflib_algebra.reorderTriples = lambda triples: list(triples)
        try:
            parsed = rdflib_algebra.translateQuery(parseQuery(sparql))
        finally:
            rdflib_algebra.reorderTriples = reorder
    top = parsed.algebra
    if top.name not in ("SelectQuery", "AskQuery"):
        raise UnsupportedQuery(f"{top.name} is not supported; plan SELECT or ASK queries.")
    form = "ASK" if top.name == "AskQuery" else "SELECT"
    return Query(form, list(field(top, "PV") or []), _node(top.p),
                 parsed.prologue.namespace_manager, sparql)


def _node(part):
    if part.name == "BGP":
        scans = [Scan(*triple) for triple in part.triples]
        if not scans:
            return Op("values", CompValue("values", res=[{}]))
        tree = scans[0]
        for scan in scans[1:]:
            tree = Join(tree, scan)
        return tree
    if part.name == "Join":
        return Join(_node(part.p1), _node(part.p2))
    if part.name in ("Graph", "ServiceGraphPattern"):
        raise UnsupportedQuery(f"{part.name} is not supported.")
    keys = tuple(k for k in ("p1", "p2", "p") if isinstance(field(part, k), CompValue))
    return Op(part.name, part, [_node(part[k]) for k in keys], keys)


def children(node):
    if isinstance(node, Scan):
        return []
    if isinstance(node, Join):
        return [node.left, node.right]
    return node.children


def clone(node):
    """A copy of the plan's structure; rdflib's nodes are shared and never modified."""
    if isinstance(node, Scan):
        return Scan(node.s, node.p, node.o, node.keep)
    if isinstance(node, Join):
        return Join(clone(node.left), clone(node.right))
    return Op(node.kind, node.part, [clone(c) for c in node.children], node.keys)


def copy_part(part):
    """A shallow copy of an rdflib algebra node, which the copy module cannot rebuild."""
    copied = CompValue(part.name)
    copied.update(part)
    return copied


def variables(node):
    """Variables a solution of this node may bind."""
    if isinstance(node, Scan):
        return node.out
    if isinstance(node, Join):
        return variables(node.left) | variables(node.right)
    kind, kids = node.kind, [variables(c) for c in node.children]
    if kind == "Project":
        return set(node.part.PV) & (kids[0] if kids else set())
    if kind == "Extend":
        return kids[0] | {node.part.var}
    if kind == "AggregateJoin":
        return kids[0] | {a.res for a in node.part.A}
    if kind == "Minus":
        return kids[0]
    if kind == "values":
        return {k for row in node.part.res for k in row}
    return set().union(*kids) if kids else set()


def certain(node):
    """Variables every solution of this node binds."""
    if isinstance(node, Scan):
        return node.out
    if isinstance(node, Join):
        return certain(node.left) | certain(node.right)
    kind, kids = node.kind, [certain(c) for c in node.children]
    if kind in ("LeftJoin", "Minus"):
        return kids[0]
    if kind == "Union":
        return kids[0] & kids[1]
    if kind == "Project":
        return set(node.part.PV) & kids[0]
    if kind in ("Group", "AggregateJoin", "values"):
        return set()
    return set().union(*kids) if kids else set()


def expression_variables(expr):
    """Variables an rdflib expression mentions."""
    found = set()

    def walk(value):
        if isinstance(value, Variable):
            found.add(value)
        elif isinstance(value, CompValue):
            for key, item in value.items():
                if key != "_vars":
                    walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(expr)
    return found


def has_pattern(expr):
    """Does the expression hold a graph pattern (EXISTS, NOT EXISTS)?"""
    if isinstance(expr, CompValue):
        if "exists" in expr.name.lower():
            return True
        return any(has_pattern(v) for k, v in expr.items() if k != "_vars")
    if isinstance(expr, (list, tuple)):
        return any(has_pattern(v) for v in expr)
    return False
