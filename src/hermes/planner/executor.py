"""Running plans.

`PlanExecutor` runs a plan as if the graph were a black box that answers only
single-pattern queries: a scan is one SPARQL query, and a join is a hash join,
or a bind join that sends the left side's bindings along with the scan, so
the plan's join order decides how much is fetched. Every other operator is
evaluated by rdflib's SPARQL engine over the rows of its children, so its
semantics are the standard ones.

`EngineExecutor` sends the query text to the store and lets the engine plan
it; it is the reference the plan executor's answers are checked against.
"""

import itertools
import time
from dataclasses import dataclass, field

from decimal import Decimal, InvalidOperation

from rdflib import BNode, Graph, Literal, URIRef, Variable
from rdflib.plugins.sparql.evaluate import evalPart
from rdflib.store import Store
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.plugins.sparql.sparql import QueryContext

from hermes.planner.algebra import Join, Scan, copy_part, variables
from hermes.planner.estimate import BIND_LIMIT, scan_below

BIND_CHUNK = 100  # bindings per scan query of a bind join
MAX_ROWS = 200_000  # rows one scan may fetch before the run stops


class PlanTooLarge(RuntimeError):
    """A scan would fetch more rows than the executor allows."""


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


@dataclass
class Result:
    form: str
    variables: list
    rows: list  # dicts of Variable -> rdflib term, for SELECT
    boolean: bool = None  # for ASK
    trace: Trace = None


class Executor:
    def __call__(self, query):
        raise NotImplementedError


class PlanExecutor(Executor):
    """Follow the plan: scans as single-pattern queries, joins in the plan's order."""

    def __init__(self, kg, bind_joins=True, max_rows=MAX_ROWS):
        self.kg = kg
        self.bind_joins = bind_joins
        self.max_rows = max_rows
        # EXISTS and property paths read the graph through rdflib: answer them from the store.
        self.context = QueryContext(graph=Graph(store=StoreTriples(kg)), initBindings={})

    def __call__(self, query):
        trace = Trace()
        started = time.perf_counter()
        rows = self.run(query.root, trace)
        trace.seconds = time.perf_counter() - started
        if query.form == "ASK":
            return Result("ASK", [], [], bool(rows), trace)
        return Result("SELECT", query.variables,
                      [{v: row.get(v) for v in query.variables} for row in rows], None, trace)

    def run(self, node, trace, bindings=None):
        if isinstance(node, Scan):
            rows = self.scan(node, trace, bindings)
        elif isinstance(node, Join):
            rows = self.join(node, trace)
        else:
            rows = self.operator(node, trace)
        trace.rows[id(node)] = len(rows)
        return rows

    def join(self, node, trace):
        left = self.run(node.left, trace)
        scan = scan_below(node.right)
        shared = sorted(variables(node.left) & scan.bound, key=str) if scan else []
        keys = {tuple(r.get(v) for v in shared) for r in left}
        if (self.bind_joins and shared and left and len(keys) <= BIND_LIMIT
                and all(None not in key for key in keys)):
            right = self.bound(node.right, trace, (shared, sorted(keys, key=str)))
        else:
            right = self.run(node.right, trace)
        return hash_join(left, right)

    def bound(self, node, trace, bindings):
        """A scan, or selections over one, fetched for the left side's bindings."""
        if isinstance(node, Scan):
            rows = self.scan(node, trace, bindings)
        else:
            part = copy_part(node.part)
            part["p"] = rows_node(self.bound(node.children[0], trace, bindings))
            rows = [dict(solution.items()) for solution in evalPart(self.context, part)]
        trace.rows[id(node)] = len(rows)
        return rows

    def scan(self, node, trace, bindings=None):
        pattern = " ".join(t.n3() for t in node.terms)
        out = sorted(node.out, key=str)
        chunks, shared = [None], []
        if bindings:
            shared, keys = bindings
            chunks = [keys[i:i + BIND_CHUNK] for i in range(0, len(keys), BIND_CHUNK)]
        rows = []
        for chunk in chunks:
            values, columns = "", out
            if chunk is not None:
                head = " ".join(v.n3() for v in shared)
                body = " ".join("(" + " ".join(t.n3() for t in key) + ")" for key in chunk)
                values = f"VALUES ({head}) {{ {body} }} "
                columns = sorted(set(out) | set(shared), key=str)
            select = " ".join(v.n3() for v in columns) or "*"
            result = self.kg.query(f"SELECT {select} WHERE {{ {values}{pattern} . }} "
                                   f"LIMIT {self.max_rows + 1}")
            trace.queries += 1
            for binding in result["results"]["bindings"]:
                rows.append({Variable(k): term(v) for k, v in binding.items()})
            if len(rows) > self.max_rows:
                raise PlanTooLarge(f"the scan {pattern} fetches more than {self.max_rows:,} rows")
        trace.fetched += len(rows)
        return rows

    def operator(self, node, trace):
        part = copy_part(node.part)
        if node.kind == "values":
            return [dict(row) for row in part.res]
        if node.kind == "AggregateJoin" and node.children and node.children[0].kind == "Group":
            group = node.children[0]  # rdflib evaluates γ and its aggregates together
            inner = copy_part(group.part)
            inner["p"] = rows_node(self.run(group.children[0], trace))
            part["p"] = inner
        else:
            for key, child in zip(node.keys, node.children):
                part[key] = rows_node(self.run(child, trace))
        return [dict(solution.items()) for solution in evalPart(self.context, part)]


class EngineExecutor(Executor):
    """Send the query text to the store and let its engine plan it."""

    def __init__(self, kg):
        self.kg = kg

    def __call__(self, query):
        trace = Trace(queries=1)
        started = time.perf_counter()
        result = self.kg.query(query.text)
        trace.seconds = time.perf_counter() - started
        if query.form == "ASK":
            return Result("ASK", [], [], bool(result.get("boolean")), trace)
        rows = [{Variable(k): term(v) for k, v in b.items()} for b in result["results"]["bindings"]]
        trace.fetched = len(rows)
        return Result("SELECT", query.variables,
                      [{v: row.get(v) for v in query.variables} for row in rows], None, trace)


EXECUTORS = {"plan": PlanExecutor, "engine": EngineExecutor}


def executor_for(kind, kg):
    """Factory: "plan" follows the optimized plan, "engine" lets the store plan."""
    try:
        return EXECUTORS[kind](kg)
    except KeyError:
        raise ValueError(f"Unknown executor {kind!r}; use one of {', '.join(EXECUTORS)}.") from None


def same_answers(one, other):
    """Do two results hold the same answers, as multisets of values?"""
    if one.form == "ASK" or other.form == "ASK":
        return one.boolean == other.boolean
    names = list(one.variables)

    def values(result):
        return sorted(tuple(comparable(row.get(v)) for v in names) for row in result.rows)

    return values(one) == values(other)


class StoreTriples(Store):
    """Adapter: an rdflib store whose triples come from single-pattern queries to the graph."""

    def __init__(self, kg):
        super().__init__()
        self.kg = kg

    def triples(self, pattern, context=None):
        names = ("s", "p", "o")
        terms = [t.n3() if t is not None else f"?{n}" for t, n in zip(pattern[:3], names)]
        free = [f"?{n}" for t, n in zip(pattern[:3], names) if t is None]
        result = self.kg.query(f"SELECT {' '.join(free) or '*'} WHERE {{ {' '.join(terms)} }}"
                               if free else f"ASK {{ {' '.join(terms)} }}")
        if not free:
            if result.get("boolean"):
                yield tuple(pattern[:3]), iter([None])
            return
        for binding in result["results"]["bindings"]:
            values = {k: term(v) for k, v in binding.items()}
            yield tuple(t if t is not None else values[n] for t, n in zip(pattern[:3], names)), \
                iter([None])

    def __len__(self, context=None):
        return 0


def comparable(value):
    """A value as text, numbers in one canonical form: engines print them differently."""
    if value is None:
        return ""
    if isinstance(value, Literal) and value.datatype is not None and str(value.datatype).endswith(
            ("#decimal", "#double", "#float", "#integer", "#int", "#long")):
        try:
            return format(Decimal(str(value)).normalize(), "f")
        except InvalidOperation:
            pass
    return str(value)


def rows_node(rows):
    return CompValue("ToMultiSet", p=CompValue("values", res=rows))


def hash_join(left, right):
    """The natural join of two lists of solutions, compatible on their shared variables."""
    if not left or not right:
        return []
    shared = sorted(set().union(*map(set, left)) & set().union(*map(set, right)), key=str)
    complete = all(row.get(v) is not None for row in itertools.chain(left, right) for v in shared)
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


def term(value):
    """An rdflib term from a SPARQL JSON result value."""
    kind = value["type"]
    if kind == "uri":
        return URIRef(value["value"])
    if kind == "bnode":
        return BNode(value["value"])
    datatype = value.get("datatype")
    return Literal(value["value"], lang=value.get("xml:lang"),
                   datatype=URIRef(datatype) if datatype else None)
