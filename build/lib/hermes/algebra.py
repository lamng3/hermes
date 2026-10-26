"""SPARQL algebra as symbolic actions, with execution feedback.

A query is a logical plan built one operator at a time. Each action is one
operator of the SPARQL 1.1 algebra (W3C SPARQL 1.1, section 18):

    triple     BGP / Join          optional   LeftJoin
    filter     Filter              union      Union
    minus      Minus               bind       Extend
    group      Group               aggregate  Aggregation
    having     Filter on groups    order      OrderBy
    project    Project             distinct   Distinct
    limit      Slice               offset     Slice
    ask        the ASK query form  remove     retract a triple or filter

Two more actions read the graph without changing the plan: `lookup` finds
entities by label and `inspect` lists the predicates around an entity or a
class. `undo` restores the previous plan.

The Explorer applies actions and runs the plan after each one, reporting what
happened: whether it parses, how many solutions it has, how that count moved,
and a sample of rows. `preview` answers "what if I apply this?" without
committing, the way an optimizer costs a rewrite before choosing it. A policy,
LLM or search, chooses the actions; see hermes.systems.algebra_agent.
"""

import json
import re
from dataclasses import MISSING, dataclass, field, fields, replace

from hermes.linking import linker_for
from hermes.validate import parse_error

CORE_PREFIXES = (
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
)


@dataclass(frozen=True)
class Plan:
    """An immutable logical plan for one SELECT or ASK query."""

    triples: tuple = ()
    filters: tuple = ()
    optionals: tuple = ()
    unions: tuple = ()
    minus: tuple = ()
    binds: tuple = ()
    group_by: tuple = ()
    aggregates: tuple = ()
    having: tuple = ()
    order_by: tuple = ()
    projection: tuple = ()
    distinct: bool = False
    limit: int | None = None
    offset: int | None = None
    form: str = "SELECT"

    @property
    def empty(self):
        return not (self.triples or self.optionals or self.unions or self.binds)


def render(plan, header=""):
    """Builder: plan -> SPARQL text."""
    where = [_statement(triple) for triple in plan.triples]
    where += [f"OPTIONAL {{ {_group(patterns)} }}" for patterns in plan.optionals]
    where += [
        " UNION ".join(f"{{ {_group(branch)} }}" for branch in branches)
        for branches in plan.unions
    ]
    where += [f"MINUS {{ {_group(patterns)} }}" for patterns in plan.minus]
    where += [f"BIND({expr} AS {var})" for expr, var in plan.binds]
    where += [f"FILTER({expr})" for expr in plan.filters]
    body = "\n  ".join(where)
    if plan.form == "ASK":
        return f"{header}ASK WHERE {{\n  {body}\n}}"
    aliases = {alias: expr for expr, alias in plan.aggregates}
    if plan.projection:
        # Project exactly what was asked; an aggregate alias is computed in place.
        columns = [f"({aliases[v]} AS {v})" if v in aliases else v for v in plan.projection]
        hidden = {alias: expr for alias, expr in aliases.items() if alias not in plan.projection}
    else:
        columns = list(plan.group_by) + [f"({e} AS {a})" for a, e in aliases.items()]
        hidden = {}
    select = "SELECT DISTINCT" if plan.distinct else "SELECT"
    lines = [f"{header}{select} {' '.join(columns) or '*'} WHERE {{\n  {body}\n}}"]
    if plan.group_by:
        lines.append("GROUP BY " + " ".join(plan.group_by))
    if plan.having:
        having = " && ".join(f"({_inline(expr, aliases)})" for expr in plan.having)
        lines.append(f"HAVING({having})")
    if plan.order_by:
        lines.append("ORDER BY " + " ".join(_inline(expr, hidden) for expr in plan.order_by))
    if plan.limit is not None:
        lines.append(f"LIMIT {plan.limit}")
    if plan.offset is not None:
        lines.append(f"OFFSET {plan.offset}")
    return "\n".join(lines)


# Actions (commands). Each one maps a plan to a new plan.

ACTIONS = {}


def action(op):
    def decorator(cls):
        cls.op = op
        ACTIONS[op] = cls
        return cls

    return decorator


class Action:
    op = ""
    reads_graph = False

    def apply(self, plan):
        return plan

    def to_dict(self):
        data = {"op": self.op}
        for item in fields(self):
            value = getattr(self, item.name)
            data[item.name] = list(map(_listify, value)) if isinstance(value, tuple) else value
        return data

    def __str__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)


@action("triple")
@dataclass(frozen=True)
class Triple(Action):
    pattern: str

    def apply(self, plan):
        pattern = _clean(self.pattern)
        if pattern in plan.triples:
            return plan
        return replace(plan, triples=plan.triples + (pattern,))


@action("remove")
@dataclass(frozen=True)
class Remove(Action):
    text: str

    def apply(self, plan):
        text = _clean(self.text)
        expr = _filter_expr(text)
        return replace(
            plan,
            triples=tuple(t for t in plan.triples if t != text),
            filters=tuple(f for f in plan.filters if f != expr),
        )


@action("filter")
@dataclass(frozen=True)
class Filter(Action):
    expr: str

    def apply(self, plan):
        return replace(plan, filters=plan.filters + (_filter_expr(self.expr),))


@action("optional")
@dataclass(frozen=True)
class Optional(Action):
    patterns: tuple

    def apply(self, plan):
        return replace(plan, optionals=plan.optionals + (_patterns(self.patterns),))


@action("union")
@dataclass(frozen=True)
class Union(Action):
    branches: tuple

    def apply(self, plan):
        branches = tuple(_patterns(branch) for branch in self.branches)
        return replace(plan, unions=plan.unions + (branches,))


@action("minus")
@dataclass(frozen=True)
class Minus(Action):
    patterns: tuple

    def apply(self, plan):
        return replace(plan, minus=plan.minus + (_patterns(self.patterns),))


@action("bind")
@dataclass(frozen=True)
class Bind(Action):
    expr: str
    var: str

    def apply(self, plan):
        return replace(plan, binds=plan.binds + ((self.expr.strip(), _var(self.var)),))


@action("group")
@dataclass(frozen=True)
class Group(Action):
    vars: tuple

    def apply(self, plan):
        return replace(plan, group_by=tuple(_var(v) for v in self.vars))


@action("aggregate")
@dataclass(frozen=True)
class Aggregate(Action):
    expr: str
    alias: str

    def apply(self, plan):
        item = (self.expr.strip(), _var(self.alias))
        return replace(plan, aggregates=plan.aggregates + (item,))


@action("having")
@dataclass(frozen=True)
class Having(Action):
    expr: str

    def apply(self, plan):
        return replace(plan, having=plan.having + (self.expr.strip(),))


@action("order")
@dataclass(frozen=True)
class Order(Action):
    expr: str

    def apply(self, plan):
        return replace(plan, order_by=plan.order_by + (self.expr.strip(),))


@action("project")
@dataclass(frozen=True)
class Project(Action):
    vars: tuple

    def apply(self, plan):
        return replace(plan, projection=tuple(_var(v) for v in self.vars))


@action("distinct")
@dataclass(frozen=True)
class Distinct(Action):
    def apply(self, plan):
        return replace(plan, distinct=True)


@action("limit")
@dataclass(frozen=True)
class Limit(Action):
    n: int

    def apply(self, plan):
        return replace(plan, limit=int(self.n))


@action("offset")
@dataclass(frozen=True)
class Offset(Action):
    n: int

    def apply(self, plan):
        return replace(plan, offset=int(self.n))


@action("ask")
@dataclass(frozen=True)
class Ask(Action):
    def apply(self, plan):
        return replace(plan, form="ASK")


@action("lookup")
@dataclass(frozen=True)
class Lookup(Action):
    text: str
    reads_graph = True


@action("inspect")
@dataclass(frozen=True)
class Inspect(Action):
    iri: str
    reads_graph = True


@action("goal")
@dataclass(frozen=True)
class State(Action):
    """State what the question asks for, like a theorem statement."""

    answer: tuple = ()
    kind: str = "entity"
    type: str = ""
    many: bool = True


@action("undo")
@dataclass(frozen=True)
class Undo(Action):
    pass


@action("finish")
@dataclass(frozen=True)
class Finish(Action):
    pass


def parse_action(data):
    """Action from a dict or a JSON text such as {"op": "triple", ...}."""
    if isinstance(data, str):
        data = _first_json_object(data)
    if not isinstance(data, dict) or data.get("op") not in ACTIONS:
        known = ", ".join(sorted(ACTIONS))
        raise ValueError(f"Expected a JSON object with op in: {known}.")
    cls = ACTIONS[data["op"]]
    arguments = {}
    for item in fields(cls):
        if item.name not in data:
            if item.default is MISSING:
                raise ValueError(f"Action {cls.op!r} needs {item.name!r}.")
            continue
        value = data[item.name]
        if item.type in (tuple, "tuple") and isinstance(value, str) and item.name != "branches":
            value = [value]  # "answer": "?x" means ["?x"]
        _check_type(cls.op, item, value)
        arguments[item.name] = _tuplify(value) if isinstance(value, list) else value
    return cls(**arguments)


def _check_type(op, item, value):
    """Reject malformed model output (null, wrong shapes) as an invalid action."""
    expected = {str: str, "str": str, int: int, "int": int, bool: bool, "bool": bool,
                tuple: list, "tuple": list}.get(item.type)
    if expected is None:
        return
    if expected is int and isinstance(value, str) and value.strip().isdigit():
        return
    if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
        raise ValueError(f"Action {op!r}: {item.name!r} must be a {expected.__name__}.")
    if expected is list:
        nested = item.name == "branches"  # union: a list of lists of patterns
        parts = [v for part in value for v in part] if nested and all(
            isinstance(part, list) for part in value) else value
        if (nested and parts is value) or not all(isinstance(v, str) for v in parts):
            shape = "lists of strings" if nested else "strings"
            raise ValueError(f"Action {op!r}: {item.name!r} must hold {shape}.")


# Feedback.


@dataclass
class Observation:
    action: str
    sparql: str = ""
    ok: bool = True
    error: str = ""
    count: int | None = None
    previous: int | None = None
    columns: list = field(default_factory=list)
    sample: list = field(default_factory=list)
    note: str = ""
    judgment: object = None

    def summary(self):
        if not self.ok:
            return f"rejected: {self.error}"
        parts = []
        if self.count is not None:
            if self.previous is not None and self.previous != self.count:
                parts.append(f"{self.previous} -> {self.count} solutions")
            else:
                parts.append(f"{self.count} solutions")
        if self.note:
            parts.append(self.note)
        if self.sample:
            rows = "; ".join(
                ", ".join(f"{k}={v}" for k, v in row.items()) for row in self.sample
            )
            parts.append(f"sample: {rows}")
        return " | ".join(parts) or "ok"


class Explorer:
    """Apply actions to a plan and observe their effect on the graph."""

    def __init__(self, kg, sample_size=5, verifier=None):
        self.kg = kg
        self.sample_size = sample_size
        self.goal = None
        self.rejected = 0
        self.linker = linker_for(kg)
        known = {prefix for prefix, _namespace in kg.vocabulary["prefixes"]}
        self.prefixes = list(kg.vocabulary["prefixes"]) + [
            pair for pair in CORE_PREFIXES if pair[0] not in known
        ]
        self.header = "".join(f"PREFIX {p}: <{ns}>\n" for p, ns in self.prefixes)
        self.plans = [Plan()]
        self.counts = [None]
        self.history = []
        self.verifier = verifier(self) if callable(verifier) else verifier

    @property
    def plan(self):
        return self.plans[-1]

    @property
    def sparql(self):
        return render(self.plan, self.header)

    def run(self, action):
        """Apply one action: commit plan actions, answer graph reads."""
        if isinstance(action, State):
            observation = self.state_goal(action)
        elif isinstance(action, Undo):
            observation = self.undo()
        elif action.reads_graph:
            observation = self.read(action)
        else:
            observation = self.step(action)
        self.history.append((action, observation))
        return observation

    def preview(self, action):
        """What would this action do? Nothing is committed."""
        return self.observe(action.apply(self.plan), str(action), self.counts[-1])

    def step(self, action):
        observation = self.preview(action)
        if observation.ok:
            self.plans.append(action.apply(self.plan))
            self.counts.append(observation.count)
        return observation

    def state_goal(self, action):
        from hermes.verifier import parse_goal

        try:
            self.goal = parse_goal(action.to_dict())
        except ValueError as exc:
            return Observation(str(action), ok=False, error=str(exc))
        judgment = self.verifier.check(self.plan, self.goal) if self.verifier else None
        return Observation(str(action), note=f"goal: {self.goal.describe()}", judgment=judgment)

    def undo(self):
        if len(self.plans) > 1:
            self.plans.pop()
            self.counts.pop()
        return self.observe(self.plan, "undo")

    def observe(self, plan, label="", previous=None):
        judgment = self.verifier.check(plan, self.goal) if self.verifier else None
        if plan.empty:
            return Observation(label, ok=False, error="the plan has no triple pattern yet",
                               judgment=judgment)
        sparql = render(plan, self.header)
        if judgment is not None and not judgment.ok:
            self.rejected += 1
            # The "kernel:" prefix predates the rename; models have seen it in every
            # cached prover prompt, so changing it would stop those runs replaying.
            return Observation(label, sparql, ok=False,
                               error="kernel: " + "; ".join(judgment.errors), judgment=judgment)
        error = parse_error(sparql)
        if error:
            return Observation(label, sparql, ok=False, error=error.splitlines()[0][:300])
        try:
            if plan.form == "ASK":
                answer = bool(self.kg.query(sparql).get("boolean"))
                return Observation(label, sparql, count=int(answer), note=f"ASK is {answer}",
                                   judgment=judgment)
            inner = render(plan)
            counted = self.kg.query(
                f"{self.header}SELECT (COUNT(*) AS ?hermes_n) WHERE {{ {{ {inner} }} }}"
            )
            count = int(counted["results"]["bindings"][0]["hermes_n"]["value"])
            sampled = self.kg.query(
                f"{self.header}SELECT * WHERE {{ {{ {inner} }} }} LIMIT {self.sample_size}"
            )
        except Exception as exc:
            return Observation(label, sparql, ok=False, error=_first_line(exc))
        columns = sampled["head"].get("vars", [])
        sample = [
            {f"?{name}": self.short(row[name]["value"]) for name in columns if name in row}
            for row in sampled["results"]["bindings"]
        ]
        if judgment is not None and self.goal and not self.goal.many and count > 1:
            judgment.goals.append(f"⊢ one answer, but the plan has {count} solutions")
        return Observation(label, sparql, count=count, previous=previous,
                           columns=columns, sample=sample, judgment=judgment)

    def read(self, action):
        if isinstance(action, Lookup):
            found = self.linker.search(action.text, limit=self.sample_size * 2)
            sample = [{"entity": self.short(iri), "label": label} for iri, label in found]
            note = f"{len(found)} entities" if found else "no entity with that label"
            return Observation(str(action), note=note, sample=sample)
        iri = self.expand(action.iri)
        patterns = (
            ("out", "?o", f"<{iri}> ?p ?o"),
            ("in", "?s", f"?s ?p <{iri}>"),
            ("instances", "?o", f"?x a <{iri}> ; ?p ?o"),
        )
        queries = [
            (direction, f"SELECT ?p (SAMPLE({value}) AS ?example) (COUNT(*) AS ?n) "
                        f"WHERE {{ {pattern} }} GROUP BY ?p ORDER BY DESC(?n) LIMIT 25")
            for direction, value, pattern in patterns
        ]
        sample = []
        for direction, query in queries:
            for row in self.kg.try_query(query)["results"]["bindings"]:
                sample.append({
                    "direction": direction,
                    "predicate": self.short(row["p"]["value"]),
                    "example": self.short(row.get("example", {}).get("value", "")),
                    "count": row["n"]["value"],
                })
        note = f"{len(sample)} predicates" if sample else "nothing is connected to it"
        return Observation(str(action), note=note, sample=sample)

    def short(self, value):
        for prefix, namespace in self.prefixes:
            local = value[len(namespace):]
            if value.startswith(namespace) and re.fullmatch(r"[A-Za-z_][\w.-]*", local):
                return f"{prefix}:{local}"
        if re.match(r"https?://", value):
            return f"<{value}>"
        return value

    def expand(self, name):
        name = name.strip().strip("<>")
        prefix, _, local = name.partition(":")
        for known, namespace in self.prefixes:
            if prefix == known and not local.startswith("//"):
                return namespace + local
        return name


def _inline(expr, aliases):
    """Replace aggregate aliases that are not in scope by their expressions."""
    for alias, aggregate in aliases.items():
        expr = re.sub(re.escape(alias) + r"(?![\w])", aggregate, expr)
    return expr


def _statement(pattern):
    return pattern if pattern.endswith(".") else f"{pattern} ."


def _group(patterns):
    return " ".join(_statement(pattern) for pattern in patterns)


def _clean(text):
    return " ".join(str(text).split()).rstrip(".").strip()


def _filter_expr(text):
    text = _clean(text)
    match = re.fullmatch(r"FILTER\s*\((.*)\)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text


def _patterns(patterns):
    if isinstance(patterns, str):
        patterns = (patterns,)
    return tuple(_clean(pattern) for pattern in patterns)


def _var(name):
    name = str(name).strip()
    return name if name.startswith(("?", "$")) else f"?{name}"


def _tuplify(value):
    return tuple(_tuplify(item) for item in value) if isinstance(value, list) else value


def _listify(value):
    return list(map(_listify, value)) if isinstance(value, tuple) else value


def _first_json_object(text):
    start = text.find("{")
    while start >= 0:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("No JSON object found.")


def _first_line(exc):
    return (str(exc).strip().splitlines() or [type(exc).__name__])[0][:300]
