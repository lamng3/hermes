"""SPARQL as relational algebra: translate a query, estimate it, rewrite it, run it.

A triple pattern is a relation over its variables, a basic graph pattern is
the natural join of its patterns, and every other SPARQL operator is a
relational one (W3C SPARQL 1.1, section 18; Pérez et al., TODS 2009):

    triple pattern   scan              FILTER          σ  selection
    join, BGP        ⋈                 OPTIONAL        ⟕  left outer join
    SELECT vars      π  projection     UNION, MINUS    ∪, −
    BIND             ε  extension      DISTINCT        δ
    ORDER BY         τ                 LIMIT, OFFSET   λ  slice
    GROUP BY         γ

`translate` keeps the query as written: patterns are joined left to right.
`optimize` rewrites it as a database optimizer would, with the graph's
statistics (hermes.pricing): selections move down to the lowest operator that
binds their variables, each join group is reordered by dynamic programming
over its cost (C_out, the summed sizes of scans and joins), and scans fetch
only the variables used above them.

`execute` runs a plan as if the graph were a black box that answers only
single-pattern queries: a scan is one SPARQL query, and a join is a hash join,
or a bind join that sends the left side's bindings along with the scan, so the
join order decides how much is fetched. Every other operator is evaluated by
rdflib's SPARQL engine over the rows of its children, so its semantics are the
standard ones.
"""

import itertools
import math
import time
from dataclasses import dataclass, field

from rdflib import BNode, Graph, Literal, URIRef, Variable
from rdflib.plugins.sparql import algebra
from rdflib.plugins.sparql.evaluate import evalPart
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.plugins.sparql.sparql import QueryContext

from hermes.pricing.statistics import EQUALITY, RANGE, scan as scan_estimate, statistics_for
from hermes.sparql.validate import _PARSE_LOCK
from hermes.sparql.verifier import TYPE

SYMBOL = {
    "Join": "⋈", "LeftJoin": "⟕", "Filter": "σ", "Project": "π", "Distinct": "δ",
    "Reduced": "δ", "Union": "∪", "Minus": "−", "Extend": "ε", "OrderBy": "τ", "Slice": "λ",
    "Group": "γ", "AggregateJoin": "γ", "values": "values", "ToMultiSet": "",
}
DP_LIMIT = 10  # join groups with more operands are ordered greedily
BIND_LIMIT = 500  # a bind join sends at most this many distinct bindings
BIND_CHUNK = 100  # bindings per scan query of a bind join


# ---- plans ----

@dataclass(eq=False)
class Scan:
    """One triple pattern. `keep` is the set of variables fetched, when pruned."""

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
    """Any other operator: rdflib's algebra node `part`, over our `children`."""

    kind: str
    part: CompValue
    children: list = field(default_factory=list)
    keys: tuple = ()  # where the children sit in `part`


@dataclass
class Query:
    form: str  # SELECT or ASK
    variables: list
    root: object
    namespaces: object
    text: str


def translate(sparql):
    """The query as relational algebra, with its patterns in the order written."""
    with _PARSE_LOCK:
        reorder = algebra.reorderTriples
        algebra.reorderTriples = lambda triples: list(triples)  # keep the written order
        try:
            parsed = algebra.translateQuery(parseQuery(sparql))
        finally:
            algebra.reorderTriples = reorder
    top = parsed.algebra
    if top.name not in ("SelectQuery", "AskQuery"):
        raise ValueError(f"{top.name} is not supported; translate SELECT or ASK queries.")
    form = "ASK" if top.name == "AskQuery" else "SELECT"
    return Query(form, list(top.get("PV") or []), _node(top.p),
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
        raise ValueError(f"{part.name} is not supported.")
    keys = tuple(k for k in ("p1", "p2", "p") if isinstance(part.get(k), CompValue))
    return Op(part.name, part, [_node(part[k]) for k in keys], keys)


# ---- variables ----

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


def _has_pattern(expr):
    """Does the expression hold a graph pattern (EXISTS, NOT EXISTS)?"""
    if isinstance(expr, CompValue):
        if "exists" in expr.name.lower():
            return True
        return any(_has_pattern(v) for k, v in expr.items() if k != "_vars")
    if isinstance(expr, (list, tuple)):
        return any(_has_pattern(v) for v in expr)
    return False


# ---- estimates ----

@dataclass
class Estimate:
    rows: float
    cost: float
    distinct: dict


class Estimator:
    """System R estimates for every node of a plan, from the graph's statistics."""

    def __init__(self, statistics):
        self.statistics = statistics
        self.memo = {}

    def __call__(self, node):
        if id(node) not in self.memo:
            self.memo[id(node)] = self._estimate(node)
        return self.memo[id(node)]

    def _estimate(self, node):
        if isinstance(node, Scan):
            s, p, o = (_token(t) for t in node.terms)
            prop = None if isinstance(node.p, Variable) else (str(node.p) if isinstance(node.p, URIRef) else None)
            cls = str(node.o) if prop == TYPE and isinstance(node.o, URIRef) else None
            rows, distinct = scan_estimate(self.statistics, s, p, o, prop, cls)
            return Estimate(rows, rows, {Variable(k[1:]): v for k, v in distinct.items()})
        if isinstance(node, Join):
            left, right = self(node.left), self(node.right)
            rows, distinct = _join(left, right)
            return Estimate(rows, left.cost + right.cost + rows, distinct)
        kids = [self(c) for c in node.children]
        cost = sum(k.cost for k in kids)
        first = kids[0] if kids else Estimate(1.0, 0.0, {})
        kind = node.kind
        if kind == "Filter":
            rows = first.rows * selectivity(node.part.expr)
            return Estimate(rows, cost, _cap(first.distinct, rows))
        if kind == "LeftJoin":
            rows, distinct = _join(kids[0], kids[1])
            rows = max(rows, kids[0].rows)
            return Estimate(rows, cost + rows, {**kids[1].distinct, **kids[0].distinct})
        if kind == "Union":
            return Estimate(kids[0].rows + kids[1].rows, cost + kids[0].rows + kids[1].rows,
                            {v: kids[0].distinct.get(v, 0) + kids[1].distinct.get(v, 0)
                             for v in set(kids[0].distinct) | set(kids[1].distinct)})
        if kind == "Minus":
            return Estimate(first.rows, cost + first.rows, first.distinct)
        if kind == "Distinct":
            rows = min(first.rows, math.prod(max(d, 1.0) for d in first.distinct.values())
                       if first.distinct else first.rows)
            return Estimate(rows, cost, _cap(first.distinct, rows))
        if kind == "Slice":
            length = node.part.length
            rows = first.rows if length is None else min(first.rows, float(length))
            return Estimate(rows, cost, _cap(first.distinct, rows))
        if kind in ("Group", "AggregateJoin"):
            group = node if kind == "Group" else (node.children[0] if node.children else node)
            by = [e for e in (group.part.get("expr") or []) if isinstance(e, Variable)]
            groups = math.prod(max(first.distinct.get(v, first.rows), 1.0) for v in by) if by else 1.0
            rows = min(first.rows, groups)
            return Estimate(rows, cost, _cap(first.distinct, rows))
        if kind == "values":
            return Estimate(float(len(node.part.res)), 0.0, {})
        if kind == "Project":
            keep = set(node.part.PV)
            return Estimate(first.rows, cost, {v: d for v, d in first.distinct.items() if v in keep})
        return Estimate(first.rows, cost, dict(first.distinct))


def selectivity(expr):
    """System R's guesses: equality with a constant keeps 1/10, anything else 1/3."""
    name = getattr(expr, "name", "")
    if name == "RelationalExpression" and expr.op == "=":
        return EQUALITY
    if name == "ConditionalAndExpression":
        return math.prod(selectivity(e) for e in [expr.expr, *(expr.other or [])])
    if name == "ConditionalOrExpression":
        return min(1.0, sum(selectivity(e) for e in [expr.expr, *(expr.other or [])]))
    if name in ("Builtin_sameTerm", "Builtin_REGEX"):
        return EQUALITY
    return RANGE


def _join(left, right):
    shared = set(left.distinct) & set(right.distinct)
    rows = left.rows * right.rows
    for var in shared:
        rows /= max(left.distinct[var], right.distinct[var], 1.0)
    distinct = {}
    for var in set(left.distinct) | set(right.distinct):
        values = [d[var] for d in (left.distinct, right.distinct) if var in d]
        distinct[var] = min(rows, *values)
    return rows, distinct


def _cap(distinct, rows):
    return {v: min(d, rows) for v, d in distinct.items()}


def _token(term):
    return f"?{term}" if isinstance(term, Variable) else "<const>"


# ---- rewriting ----

class Optimizer:
    """Rewrite a plan: push selections down, reorder joins, prune what scans fetch."""

    def __init__(self, statistics, namespaces=None):
        self.statistics = statistics
        self.namespaces = namespaces
        self.log = []

    def __call__(self, query):
        root = self.push(clone(query.root))
        root = self.order(root)
        needed = set(query.variables) if query.form == "SELECT" else set()
        self.prune(root, needed or certain(root))
        return Query(query.form, query.variables, root, query.namespaces, query.text)

    # Selections down.
    def push(self, node):
        if isinstance(node, Scan):
            return node
        if isinstance(node, Join):
            return Join(self.push(node.left), self.push(node.right))
        node.children = [self.push(c) for c in node.children]
        if node.kind != "Filter" or _has_pattern(node.part.expr):
            return node
        wanted = expression_variables(node.part.expr)
        moved = self._sink(node.children[0], node, wanted)
        if moved is not None:
            names = sorted("?" + v for v in wanted)
            self.log.append(f"σ {expression(node.part.expr, self.namespaces)} moves down to "
                            f"where {', '.join(names)} {'is' if len(names) == 1 else 'are'} bound")
            return moved
        return node

    def _sink(self, child, select, wanted):
        """Child with the selection placed inside it, or None if it cannot go lower."""
        if isinstance(child, Join):
            for side in ("left", "right"):
                branch = getattr(child, side)
                if wanted <= certain(branch):
                    inner = self._sink(branch, select, wanted)
                    setattr(child, side, inner if inner is not None else _filter(select, branch))
                    return child
            return None
        if isinstance(child, Op):
            if child.kind == "LeftJoin" and wanted <= certain(child.children[0]):
                inner = self._sink(child.children[0], select, wanted)
                child.children[0] = inner if inner is not None else _filter(select, child.children[0])
                return child
            if child.kind == "Union":
                child.children = [self._sink(c, select, wanted) or _filter(select, c)
                                  for c in child.children]
                return child
            if (child.kind in ("Filter", "Distinct", "Reduced", "OrderBy")
                    or (child.kind == "Extend" and child.part.var not in wanted)
                    or (child.kind == "Project" and wanted <= set(child.part.PV))):
                inner = self._sink(child.children[0], select, wanted)
                child.children[0] = inner if inner is not None else _filter(select, child.children[0])
                return child
        return None

    # Joins reordered.
    def order(self, node):
        if isinstance(node, Scan):
            return node
        if isinstance(node, Op):
            node.children = [self.order(c) for c in node.children]
            if node.kind != "Filter" or not _is_group(node.children[0]):
                return node
        operands, selections = [], []
        _flatten(node, operands, selections)
        operands = [self.order(o) if not isinstance(o, Scan) else o for o in operands]
        if len(operands) < 2:
            return node
        estimate = Estimator(self.statistics)
        before = _cost_of(node, estimate)
        chosen = self._best(operands, selections, estimate)
        tree = _build(chosen, operands, selections)
        after = _cost_of(tree, estimate)
        if after < before - 1e-9:
            names = ", ".join(_describe(operands[i], self.namespaces) for i in chosen)
            self.log.append(f"⋈ order {names}: estimated C_out {before:,.0f} → {after:,.0f}")
            return tree
        return node

    def _best(self, operands, selections, estimate):
        n = len(operands)
        leaves = [estimate(o) for o in operands]
        bound = [certain(o) for o in operands]
        if n > DP_LIMIT:
            return self._greedy(operands, selections, estimate)
        best = {}
        for i in range(n):
            rows, distinct = _with_filters(leaves[i].rows, leaves[i].distinct, bound[i],
                                           selections, set())
            best[frozenset([i])] = (leaves[i].rows, rows, distinct, (i,), bound[i])
        for size in range(2, n + 1):
            for subset in itertools.combinations(range(n), size):
                key = frozenset(subset)
                options = []
                for j in subset:
                    rest = key - {j}
                    if rest not in best:
                        continue
                    cost, rows, distinct, order, vars_ = best[rest]
                    connected = bool(vars_ & set(leaves[j].distinct))
                    joined, merged = _join(Estimate(rows, 0, distinct), leaves[j])
                    applied = {f for f, (wanted, _sel) in enumerate(selections) if wanted <= vars_}
                    joined, merged = _with_filters(joined, merged, vars_ | bound[j], selections,
                                                   applied)
                    options.append((not connected, cost + leaves[j].rows + joined, joined, merged,
                                    order + (j,), vars_ | bound[j]))
                if options:
                    _cart, *value = min(options, key=lambda o: (o[0], o[1]))
                    best[key] = tuple(value)
        return list(best[frozenset(range(n))][3])

    def _greedy(self, operands, selections, estimate):
        leaves = [estimate(o) for o in operands]
        remaining = set(range(len(operands)))
        first = min(remaining, key=lambda i: leaves[i].rows)
        order, vars_ = [first], certain(operands[first])
        remaining.discard(first)
        while remaining:
            connected = [i for i in remaining if vars_ & set(leaves[i].distinct)] or list(remaining)
            nxt = min(connected, key=lambda i: leaves[i].rows)
            order.append(nxt)
            vars_ |= certain(operands[nxt])
            remaining.discard(nxt)
        return order

    # What scans fetch.
    def prune(self, node, needed):
        if isinstance(node, Scan):
            keep = node.bound & needed
            if not keep:  # a scan must still bind something, to keep its multiplicity
                keep = {sorted(node.bound, key=str)[0]} if node.bound else set()
            if keep != node.bound:
                node.keep = frozenset(keep)
                dropped = ", ".join(sorted("?" + v for v in node.bound - keep))
                self.log.append(f"π {_describe(node, self.namespaces)} fetches only "
                                f"{', '.join(sorted('?' + v for v in keep))} (not {dropped})")
            return
        if isinstance(node, Join):
            shared = variables(node.left) & variables(node.right)
            self.prune(node.left, needed | shared)
            self.prune(node.right, needed | shared)
            return
        kind, part = node.kind, node.part
        if kind == "Project":
            return self.prune(node.children[0], set(part.PV))
        if kind == "Filter":
            return self.prune(node.children[0], needed | expression_variables(part.expr))
        if kind == "Extend":
            return self.prune(node.children[0], (needed - {part.var}) | expression_variables(part.expr))
        if kind == "OrderBy":
            return self.prune(node.children[0], needed | expression_variables(part.expr))
        if kind == "Slice":
            return self.prune(node.children[0], needed)
        if kind == "LeftJoin":
            shared = variables(node.children[0]) & variables(node.children[1])
            extra = expression_variables(part.get("expr"))
            for child in node.children:
                self.prune(child, needed | shared | extra)
            return
        if kind == "Union":
            for child in node.children:
                self.prune(child, needed)
            return
        for child in node.children:  # anything else needs every variable below it
            self.prune(child, variables(child))


def clone(node):
    """A copy of the plan's structure; rdflib's nodes are shared, never modified."""
    if isinstance(node, Scan):
        return Scan(node.s, node.p, node.o, node.keep)
    if isinstance(node, Join):
        return Join(clone(node.left), clone(node.right))
    return Op(node.kind, node.part, [clone(c) for c in node.children], node.keys)


def _filter(select, child):
    part = _copy(select.part)
    return Op("Filter", part, [child], ("p",))


def _is_group(node):
    return isinstance(node, Join)


def _flatten(node, operands, selections):
    """Operands and selections of a join group: joins, with selections between them."""
    if isinstance(node, Join):
        _flatten(node.left, operands, selections)
        _flatten(node.right, operands, selections)
    elif isinstance(node, Op) and node.kind == "Filter" and not _has_pattern(node.part.expr) \
            and isinstance(node.children[0], (Join, Op)) and _within_group(node.children[0]):
        selections.append((expression_variables(node.part.expr), node))
        _flatten(node.children[0], operands, selections)
    else:
        operands.append(node)


def _within_group(node):
    return isinstance(node, Join) or (isinstance(node, Op) and node.kind == "Filter"
                                      and _within_group(node.children[0]))


def _with_filters(rows, distinct, bound, selections, applied):
    for index, (wanted, select) in enumerate(selections):
        if index not in applied and wanted <= bound:
            rows *= selectivity(select.part.expr)
    return rows, _cap(distinct, rows)


def _build(order, operands, selections):
    placed, tree, bound = set(), None, set()
    for i in order:
        tree = operands[i] if tree is None else Join(tree, operands[i])
        bound |= certain(operands[i])
        for index, (wanted, select) in enumerate(selections):
            if index not in placed and wanted <= bound:
                tree = _filter(select, tree)
                placed.add(index)
    for index, (_wanted, select) in enumerate(selections):
        if index not in placed:
            tree = _filter(select, tree)
    return tree


def _cost_of(node, estimate):
    return estimate(node).cost


# ---- execution ----

@dataclass
class Trace:
    """What a run did: rows per node, store queries, rows fetched, seconds."""

    rows: dict = field(default_factory=dict)
    queries: int = 0
    fetched: int = 0
    seconds: float = 0.0

    @property
    def intermediate(self):
        return sum(self.rows.values())


class Executor:
    """Run a plan against a store that answers single-pattern SPARQL queries."""

    def __init__(self, kg, bind_joins=True):
        self.kg = kg
        self.bind_joins = bind_joins
        self.context = QueryContext(graph=Graph())

    def __call__(self, query):
        trace = Trace()
        started = time.perf_counter()
        rows = self.run(query.root, trace)
        trace.seconds = time.perf_counter() - started
        if query.form == "ASK":
            return bool(rows), trace
        return [{v: row.get(v) for v in query.variables} for row in rows], trace

    def run(self, node, trace, bindings=None):
        if isinstance(node, Scan):
            rows = self.scan(node, trace, bindings)
        elif isinstance(node, Join):
            left = self.run(node.left, trace)
            shared = sorted(variables(node.left) & node.right.bound, key=str) \
                if isinstance(node.right, Scan) else []
            keys = {tuple(r.get(v) for v in shared) for r in left}
            if (self.bind_joins and shared and left and len(keys) <= BIND_LIMIT
                    and all(None not in k for k in keys)):
                right = self.run(node.right, trace, (shared, sorted(keys, key=str)))
            else:
                right = self.run(node.right, trace)
            rows = _hash_join(left, right)
        else:
            rows = self.operator(node, trace)
        trace.rows[id(node)] = len(rows)
        return rows

    def scan(self, node, trace, bindings=None):
        out = sorted(node.out, key=str)
        pattern = " ".join(t.n3() for t in node.terms)
        chunks = [None]
        if bindings:
            shared, keys = bindings
            chunks = [keys[i:i + BIND_CHUNK] for i in range(0, len(keys), BIND_CHUNK)]
        rows = []
        for chunk in chunks:
            values = ""
            if chunk is not None:
                head = " ".join(v.n3() for v in shared)
                body = " ".join("(" + " ".join(t.n3() for t in key) + ")" for key in chunk)
                values = f"VALUES ({head}) {{ {body} }} "
                columns = sorted(set(out) | set(shared), key=str)
            else:
                columns = out
            select = " ".join(v.n3() for v in columns) or "*"
            result = self.kg.query(f"SELECT {select} WHERE {{ {values}{pattern} . }}")
            trace.queries += 1
            for binding in result["results"]["bindings"]:
                rows.append({Variable(k): _term(v) for k, v in binding.items()})
        trace.fetched += len(rows)
        return rows

    def operator(self, node, trace):
        part = _copy(node.part)
        if node.kind == "AggregateJoin" and node.children and node.children[0].kind == "Group":
            group = node.children[0]
            inner = _copy(group.part)
            inner["p"] = _values(self.run(group.children[0], trace))
            part["p"] = inner
        elif node.kind == "values":
            return [dict(row) for row in part.res]
        else:
            for key, child in zip(node.keys, node.children):
                part[key] = _values(self.run(child, trace))
        return [{k: v for k, v in solution.items()} for solution in evalPart(self.context, part)]


def _copy(part):
    """A shallow copy of an rdflib algebra node, which the copy module cannot rebuild."""
    copied = CompValue(part.name)
    copied.update(part)
    return copied


def _values(rows):
    return CompValue("ToMultiSet", p=CompValue("values", res=rows))


def _hash_join(left, right):
    if not left or not right:
        return []
    shared = sorted(set().union(*map(set, left)) & set().union(*map(set, right)), key=str)
    complete = all(all(r.get(v) is not None for v in shared) for r in itertools.chain(left, right))
    if complete:
        index = {}
        for row in right:
            index.setdefault(tuple(row[v] for v in shared), []).append(row)
        return [{**l, **r} for l in left for r in index.get(tuple(l[v] for v in shared), [])]
    joined = []
    for l in left:  # some rows leave a shared variable unbound: compare row by row
        for r in right:
            if all(l.get(v) is None or r.get(v) is None or l[v] == r[v] for v in set(l) & set(r)):
                joined.append({**l, **r})
    return joined


def _term(value):
    kind = value["type"]
    if kind == "uri":
        return URIRef(value["value"])
    if kind == "bnode":
        return BNode(value["value"])
    return Literal(value["value"], lang=value.get("xml:lang"),
                   datatype=URIRef(value["datatype"]) if value.get("datatype") else None)


# ---- reading plans ----

def expression(expr, namespaces=None):
    """SPARQL-like text of an rdflib expression."""
    show = lambda e: expression(e, namespaces)  # noqa: E731
    if isinstance(expr, Variable):
        return f"?{expr}"
    if isinstance(expr, (URIRef, Literal, BNode)):
        return expr.n3(namespaces) if namespaces is not None else expr.n3()
    if isinstance(expr, (list, tuple)):
        return ", ".join(show(e) for e in expr)
    if not isinstance(expr, CompValue):
        return str(expr)
    name = expr.name
    if name == "RelationalExpression":
        other = expr.other
        right = f"({show(other)})" if isinstance(other, list) else show(other)
        return f"{show(expr.expr)} {expr.op} {right}"
    if name in ("ConditionalAndExpression", "ConditionalOrExpression"):
        op = " && " if name.startswith("ConditionalAnd") else " || "
        return op.join(show(e) for e in [expr.expr, *(expr.other or [])])
    if name in ("AdditiveExpression", "MultiplicativeExpression"):
        text = show(expr.expr)
        for op, other in zip(expr.op or [], expr.other or []):
            text += f" {op} {show(other)}"
        return text
    if name == "UnaryNot":
        return f"!{show(expr.expr)}"
    if name == "UnaryMinus":
        return f"-{show(expr.expr)}"
    if name == "OrderCondition":
        return f"{expr.order}({show(expr.expr)})" if expr.order else show(expr.expr)
    if name.startswith("Builtin_") or name.startswith("Aggregate_"):
        args = [expr[k] for k in ("arg", "arg1", "arg2", "arg3", "vars") if expr.get(k) is not None]
        distinct = "DISTINCT " if expr.get("distinct") else ""
        return f"{name.split('_', 1)[1]}({distinct}{', '.join(show(a) for a in args)})"
    if name == "Function":
        return f"{show(expr.iri)}({show(expr.expr or [])})"
    if name == "TrueFilter":
        return "true"
    return name


def _describe(node, namespaces=None):
    if isinstance(node, Scan):
        return "(" + " ".join(expression(t, namespaces) for t in node.terms) + ")"
    if isinstance(node, Join):
        return "⋈"
    if node.kind == "Filter" and node.children:
        return "σ" + _describe(node.children[0], namespaces)
    return SYMBOL.get(node.kind, node.kind)


def label(node, namespaces=None):
    """One line for a node: its operator and arguments."""
    if isinstance(node, Scan):
        pruned = "" if node.keep is None else \
            "  π " + " ".join("?" + v for v in sorted(node.keep, key=str))
        return "scan " + _describe(node, namespaces) + pruned
    if isinstance(node, Join):
        return "⋈"
    part, kind = node.part, node.kind
    if kind == "Filter":
        return "σ " + expression(part.expr, namespaces)
    if kind == "Project":
        return "π " + " ".join(expression(v) for v in part.PV)
    if kind == "Extend":
        return f"ε ?{part.var} ← {expression(part.expr, namespaces)}"
    if kind == "OrderBy":
        return "τ " + expression(part.expr, namespaces)
    if kind == "Slice":
        return f"λ offset {part.start or 0}" + (f" limit {part.length}" if part.length is not None else "")
    if kind == "LeftJoin":
        expr = part.get("expr")
        cond = "" if expr is None or getattr(expr, "name", "") == "TrueFilter" else " " + expression(expr, namespaces)
        return "⟕" + cond
    if kind == "Group":
        by = part.get("expr") or []
        return "γ by " + (expression(by, namespaces) if by else "()")
    if kind == "AggregateJoin":
        return "γ " + ", ".join(f"?{a.res} ← {expression(a, namespaces)}" for a in part.A)
    if kind == "values":
        return f"values ({len(part.res)} rows)"
    if kind == "ToMultiSet":
        return "multiset"
    return SYMBOL.get(kind, kind)


def children(node):
    if isinstance(node, Scan):
        return []
    if isinstance(node, Join):
        return [node.left, node.right]
    return node.children


def tree(node, namespaces=None, notes=None, prefix=""):
    """The plan as an indented tree; `notes(node)` adds text after a node's line."""
    lines = [label(node, namespaces) + (notes(node) if notes else "")]
    kids = children(node)
    for index, child in enumerate(kids):
        last = index == len(kids) - 1
        sub = tree(child, namespaces, notes).splitlines()
        lines.append(("└─ " if last else "├─ ") + sub[0])
        lines.extend(("   " if last else "│  ") + line for line in sub[1:])
    return "\n".join(lines)


def formula(node, namespaces=None):
    """The plan as one relational-algebra expression."""
    if isinstance(node, Scan):
        return _describe(node, namespaces)
    if isinstance(node, Join):
        return f"({formula(node.left, namespaces)} ⋈ {formula(node.right, namespaces)})"
    kids = [formula(c, namespaces) for c in node.children]
    if node.kind in ("LeftJoin", "Union", "Minus"):
        return f"({kids[0]} {SYMBOL[node.kind]} {kids[1]})"
    head = label(node, namespaces)
    sym, _, rest = head.partition(" ")
    sub = f"_{{{rest}}}" if rest else ""
    return f"{sym}{sub}({kids[0]})" if kids else head


# ---- all together ----

def same_answers(ours, engine, query):
    """Do our rows equal the engine's, as multisets of values over the projected variables?"""
    if query.form == "ASK":
        return ours == bool(engine.get("boolean"))
    names = [str(v) for v in query.variables]
    mine = sorted(tuple(str(row.get(Variable(n))) if row.get(Variable(n)) is not None else None
                        for n in names) for row in ours)
    theirs = sorted(tuple(b[n]["value"] if n in b else None for n in names)
                    for b in engine["results"]["bindings"])
    return mine == theirs

def plan(sparql, kg, optimize=True):
    """Translate, estimate, rewrite: (as written, optimized, rewrites, estimates)."""
    written = translate(sparql)
    statistics = statistics_for(kg)
    optimizer = Optimizer(statistics, written.namespaces)
    better = optimizer(written) if optimize else written
    return written, better, optimizer.log, Estimator(statistics)
