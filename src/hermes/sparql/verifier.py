"""The verifier: goals, typing and well-formedness of a plan, checked
before anything runs.

It plays the role of a database's binder and semantic analyzer, with the
ontology as the catalog. A question's goal states what must be returned; the
verifier resolves every term of a plan against the schema, infers a typing
context, and lists the goals the plan does not yet meet. It is a pure
function, `Verifier.check(plan, goal)`: it never touches the graph, and a plan
it rejects is never run.

    Database binder           Hermes verifier
    catalog                   Schema: classes, properties, domains, ranges
    name resolution           a term outside the schema's namespaces is unknown
    type checking             domain/range violations, literals as subjects
    output columns            Goal: answer variables, kind, type, cardinality
    bound query               no errors and no open goals

Typing judgments come from the schema: `?x a C` gives `?x : C`, and a triple
`?x p ?y` gives `?x : domain(p)` and `?y : range(p)`. Classes are compared up
to rdfs:subClassOf. The open-world assumption makes class conflicts warnings,
not errors; using a literal where an entity is required is an error.
"""

import difflib
import re
from dataclasses import dataclass, field

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD = "http://www.w3.org/2001/XMLSchema#"
TYPE = RDF + "type"
LITERAL = "literal"
NUMERIC = {XSD + name for name in (
    "integer", "int", "decimal", "double", "float", "long", "short", "byte",
    "nonNegativeInteger", "positiveInteger", "negativeInteger", "nonPositiveInteger",
    "unsignedInt", "unsignedLong",
)}
KINDS = ("entity", "literal", "number", "boolean")

_TOKEN = re.compile(
    r"<[^>]*>"
    r'|"(?:[^"\\]|\\.)*"(?:@[\w-]+|\^\^\S+)?'
    r"|'(?:[^'\\]|\\.)*'(?:@[\w-]+|\^\^\S+)?"
    r"|[?$]\w+"
    r"|[;,]"
    r"|[^\s;,]+"
)
_VARIABLE = re.compile(r"[?$][A-Za-z_]\w*")


@dataclass(frozen=True)
class Goal:
    """What the question asks for: the statement to prove."""

    answer: tuple = ()
    kind: str = "entity"
    type: str = ""
    many: bool = True

    def describe(self):
        if self.kind == "boolean":
            return "⊢ a yes/no answer"
        what = ", ".join(self.answer) or "?"
        typed = f" : {self.type}" if self.type else f" : {self.kind}"
        count = "" if self.many else " (one)"
        return f"⊢ {what}{typed}{count}"


@dataclass
class Judgment:
    """The proof state after one step."""

    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    context: dict = field(default_factory=dict)
    goals: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.errors

    @property
    def accomplished(self):
        return not self.errors and not self.goals

    def render(self, short):
        lines = []
        for variable, types in sorted(self.context.items()):
            shown = ", ".join(sorted(short(t) for t in types)) or "?"
            lines.append(f"  {variable} : {shown}")
        state = "context:\n" + ("\n".join(lines) if lines else "  (empty)")
        goals = "\n".join(f"  {goal}" for goal in self.goals) or "  goals accomplished"
        state += f"\nopen goals:\n{goals}"
        if self.warnings:
            state += "\nwarnings:\n" + "\n".join(f"  {w}" for w in self.warnings)
        return state


class Schema:
    """Classes, properties, domains, ranges and subclass closure of a graph."""

    def __init__(self, classes=(), properties=(), domains=None, ranges=None, parents=None):
        self.classes = set(classes)
        self.properties = set(properties)
        self.domains = domains or {}
        self.ranges = ranges or {}
        self.ancestors = _closure(parents or {})
        self.namespaces = {_namespace(term) for term in self.classes | self.properties}

    @classmethod
    def from_kg(cls, kg):
        store = kg.schema

        def pairs(query):
            rows = store.try_query(query)["results"]["bindings"]
            return [(row["a"]["value"], row["b"]["value"]) for row in rows
                    if row["a"]["type"] == "uri" and row["b"]["type"] == "uri"]

        prefixes = (
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
        )
        classes = pairs(prefixes + "SELECT ?a ?b WHERE { VALUES ?b { owl:Class rdfs:Class } ?a a ?b }")
        properties = pairs(prefixes + "SELECT ?a ?b WHERE { VALUES ?b { owl:ObjectProperty "
                           "owl:DatatypeProperty rdf:Property } ?a a ?b }")
        domains, ranges, parents = {}, {}, {}
        for prop, domain in pairs(prefixes + "SELECT ?a ?b WHERE { ?a rdfs:domain ?b }"):
            domains.setdefault(prop, set()).add(domain)
        for prop, range_ in pairs(prefixes + "SELECT ?a ?b WHERE { ?a rdfs:range ?b }"):
            ranges.setdefault(prop, set()).add(range_)
        for child, parent in pairs(prefixes + "SELECT ?a ?b WHERE { ?a rdfs:subClassOf ?b }"):
            parents.setdefault(child, set()).add(parent)
        return cls(
            {c for c, _ in classes}, {p for p, _ in properties}, domains, ranges, parents
        )

    def subsumes(self, general, specific):
        return general == specific or general in self.ancestors.get(specific, ())

    def compatible(self, one, other):
        return self.subsumes(one, other) or self.subsumes(other, one)

    def is_literal_type(self, term):
        return term == LITERAL or term.startswith(XSD) or term == RDFS + "Literal"

    def covers(self, iri):
        return _namespace(iri) in self.namespaces


class Verifier:
    """Check a plan against the schema and the goal. Nothing is executed."""

    def __init__(self, schema, prefixes, kind_of=None):
        self.schema = schema
        self.prefixes = list(prefixes)
        # For properties without a declared range: "literal" or "entity" when the
        # data's objects are all one kind (hermes.pricing.Statistics), else None.
        self.kind_of = kind_of

    @classmethod
    def for_kg(cls, kg, prefixes, statistics=None):
        kind_of = None
        if statistics is not None:
            def kind_of(iri):
                count = statistics.predicate(iri)
                return count.kind if count else None
        return cls(Schema.from_kg(kg), prefixes, kind_of)

    def check(self, plan, goal=None):
        judgment = Judgment()
        context = judgment.context
        binding = list(plan.triples)
        for group in plan.optionals:
            binding.extend(group)
        for branches in plan.unions:
            for branch in branches:
                binding.extend(branch)
        excluded = [pattern for group in plan.minus for pattern in group]
        for pattern in binding + excluded:
            for subject, predicate, value in self.triples(pattern, judgment):
                self._type_triple(subject, predicate, value, judgment)
        # MINUS checks its patterns but binds none of their variables.
        in_scope = set()
        for pattern in binding:
            in_scope |= set(_VARIABLE.findall(pattern))
        for variable in list(context):
            if variable not in in_scope:
                del context[variable]
        for variable in in_scope:
            context.setdefault(variable, set())
        for expr, variable in plan.binds:
            self._require(expr, in_scope, "BIND", judgment)
            if variable in in_scope:
                judgment.errors.append(f"BIND target {variable} is already bound")
            in_scope.add(variable)
            context.setdefault(variable, set())
        for expr in plan.filters:
            self._require(expr, in_scope, "FILTER", judgment)
        aliases = set()
        for expr, alias in plan.aggregates:
            self._require(expr, in_scope, "aggregate", judgment)
            aliases.add(alias)
            context[alias] = {XSD + "decimal"} if _counts(expr) else set()
        for variable in plan.group_by:
            if variable not in in_scope:
                judgment.errors.append(f"unbound variable {variable} in GROUP BY")
        visible = in_scope | aliases
        for expr in plan.having:
            self._require(expr, visible, "HAVING", judgment)
        for expr in plan.order_by:
            self._require(expr, visible, "ORDER BY", judgment)
        for variable in plan.projection:
            if variable not in visible:
                judgment.errors.append(f"unbound variable {variable} in SELECT")
        if plan.aggregates or plan.group_by:
            for variable in plan.projection:
                if variable in in_scope and variable not in plan.group_by:
                    judgment.errors.append(
                        f"{variable} is neither grouped nor aggregated; "
                        "add it to GROUP BY or aggregate it"
                    )
        if goal:
            judgment.goals = self.open_goals(plan, goal, judgment, visible)
        return judgment

    def open_goals(self, plan, goal, judgment, visible):
        goals = []
        if goal.kind == "boolean":
            if plan.form != "ASK":
                goals.append("⊢ a yes/no answer: use ask")
            return goals
        if plan.form == "ASK":
            goals.append("⊢ the question asks for values, not yes/no")
        projected = list(plan.projection) or sorted(visible)
        for variable in goal.answer:
            if variable not in visible:
                goals.append(f"⊢ bind {variable}")
                continue
            if variable not in projected:
                goals.append(f"⊢ project {variable}")
            types = judgment.context.get(variable, set())
            if goal.kind == "number" and not (types & NUMERIC or _counts_alias(plan, variable)):
                goals.append(f"⊢ {variable} : number (aggregate it or use a numeric property)")
            if goal.kind == "entity" and any(self.schema.is_literal_type(t) for t in types):
                goals.append(f"⊢ {variable} : entity, but it is a literal")
            if goal.type:
                wanted = self.expand(goal.type)
                if not any(self.schema.subsumes(wanted, t) for t in types):
                    goals.append(f"⊢ {variable} : {goal.type} (state it with a type triple)")
        extra = [v for v in projected if v not in goal.answer]
        if extra:
            goals.append(f"⊢ project only the answer; drop {', '.join(extra)}")
        return goals

    def triples(self, pattern, judgment):
        """Split a pattern into (subject, predicate, object) token triples."""
        triples, complete = split_triples(pattern)
        if not complete:
            judgment.errors.append(f"incomplete triple pattern: {pattern}")
        return triples

    def expand(self, token):
        if token == "a":
            return TYPE
        if token.startswith("<") and token.endswith(">"):
            return token[1:-1]
        prefix, sep, local = token.partition(":")
        if sep:
            for known, namespace in self.prefixes:
                if known == prefix:
                    return namespace + local
        return token

    def _type_triple(self, subject, predicate, value, judgment):
        schema, context = self.schema, judgment.context
        if _is_path(predicate):
            if predicate.startswith("^") and _is_literal(value):
                judgment.errors.append(
                    f"type mismatch: {predicate} reads backwards, so the literal {value} "
                    "would be a subject")
            return
        prop = self.expand(predicate)
        if _is_variable(predicate):
            context.setdefault(predicate, set())
            prop = None
        elif prop != TYPE and schema.covers(prop) and prop not in schema.properties:
            judgment.errors.append(self._unknown(predicate, schema.properties, "property"))
            return
        if _is_literal(subject):
            judgment.errors.append(f"type mismatch: the literal {subject} cannot be a subject")
        if prop == TYPE:
            if _is_variable(value):
                context.setdefault(value, set())
                return
            cls = self.expand(value)
            if schema.covers(cls) and cls not in schema.classes:
                judgment.errors.append(self._unknown(value, schema.classes, "class"))
                return
            self._assign(subject, cls, f"{subject} a {value}", judgment)
            return
        if prop is None:
            return
        for domain in schema.domains.get(prop, ()):
            self._assign(subject, domain, f"the domain of {predicate}", judgment)
        ranges = schema.ranges.get(prop, ())
        if ranges and _is_iri(value) and all(schema.is_literal_type(r) for r in ranges):
            judgment.errors.append(
                f"type mismatch: {predicate} expects a literal, got the entity {value}")
        for range_ in ranges:
            if _is_literal(value) and not schema.is_literal_type(range_):
                judgment.errors.append(
                    f"type mismatch: {predicate} expects an entity of {self.short(range_)}, "
                    f"got the literal {value}"
                )
            elif _is_variable(value):
                self._assign(value, range_, f"the range of {predicate}", judgment)
        if not ranges and self.kind_of and prop != TYPE:
            kind = self.kind_of(prop)
            if kind == "literal" and _is_iri(value):
                judgment.errors.append(
                    f"type mismatch: {predicate} holds only literals in this graph, "
                    f"got the entity {value}")
            elif kind == "entity" and _is_literal(value):
                judgment.errors.append(
                    f"type mismatch: {predicate} holds only entities in this graph, "
                    f"got the literal {value}")
            elif kind == "literal" and _is_variable(value):
                self._assign(value, LITERAL, f"the values of {predicate}", judgment)
        if _is_variable(value) and not schema.ranges.get(prop):
            context.setdefault(value, set())

    def _assign(self, term, cls, reason, judgment):
        if not _is_variable(term):
            return
        types = judgment.context.setdefault(term, set())
        literal = self.schema.is_literal_type(cls)
        if literal and any(not self.schema.is_literal_type(t) for t in types):
            judgment.errors.append(
                f"type mismatch: {term} is an entity but {reason} makes it a literal"
            )
        elif not literal and any(self.schema.is_literal_type(t) for t in types):
            judgment.errors.append(
                f"type mismatch: {term} is a literal but {reason} makes it an entity"
            )
        elif not literal:
            for other in types:
                if not self.schema.is_literal_type(other) and not self.schema.compatible(cls, other):
                    judgment.warnings.append(
                        f"{term} is both {self.short(other)} and {self.short(cls)} ({reason})"
                    )
        types.add(cls)

    def _require(self, expr, scope, where, judgment):
        for variable in _VARIABLE.findall(_strip_strings(expr)):
            if variable not in scope:
                judgment.errors.append(f"unbound variable {variable} in {where}")

    def _unknown(self, token, known, kind):
        names = {self.short(term): term for term in known}
        close = difflib.get_close_matches(token, list(names), n=1, cutoff=0.6)
        hint = f" (did you mean {close[0]}?)" if close else ""
        return f"unknown {kind} {token}{hint}"

    def short(self, iri):
        for prefix, namespace in self.prefixes:
            if iri.startswith(namespace):
                return f"{prefix}:{iri[len(namespace):]}"
        return iri if iri == LITERAL else f"<{iri}>"


def parse_goal(data):
    answer = data.get("answer") or []
    if isinstance(answer, str):
        answer = [answer]
    kind = data.get("kind", "entity")
    if kind not in KINDS:
        raise ValueError(f"goal kind must be one of {', '.join(KINDS)}")
    answer = tuple(a if a.startswith(("?", "$")) else f"?{a}" for a in answer)
    return Goal(answer, kind, data.get("type", "") or "", bool(data.get("many", True)))


def split_triples(pattern):
    """(subject, predicate, object) token triples of a pattern, and whether it is complete."""
    triples, subject, predicate, expect = [], None, None, "s"
    for token in _TOKEN.findall(pattern):
        if token == ";":
            expect = "p"
        elif token == ",":
            expect = "o"
        elif expect == "s":
            subject, expect = token, "p"
        elif expect == "p":
            predicate, expect = token, "o"
        else:
            triples.append((subject, predicate, token))
    return triples, expect == "s" or bool(triples)


def _closure(parents):
    ancestors = {}

    def visit(node, seen):
        for parent in parents.get(node, ()):
            if parent not in seen:
                seen.add(parent)
                visit(parent, seen)
        return seen

    for node in parents:
        ancestors[node] = visit(node, set())
    return ancestors


def _namespace(iri):
    if "#" in iri:
        return iri.rsplit("#", 1)[0] + "#"
    return iri.rsplit("/", 1)[0] + "/"


def _is_variable(token):
    return bool(token) and token[0] in "?$"


def _is_literal(token):
    return bool(token) and (token[0] in "\"'" or re.fullmatch(r"[-+]?\d+(\.\d+)?", token)
                            or token in ("true", "false"))


def _is_iri(token):
    return bool(token) and not _is_variable(token) and not _is_literal(token) and token != "a"


def _is_path(token):
    return any(mark in token for mark in "/|^*+") and not token.startswith("<")


def _strip_strings(expr):
    return re.sub(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'', '""', expr)


def _counts(expr):
    return expr.strip().upper().startswith(("COUNT", "SUM", "AVG", "MIN", "MAX"))


def _counts_alias(plan, variable):
    return any(alias == variable and _counts(expr) for expr, alias in plan.aggregates)
