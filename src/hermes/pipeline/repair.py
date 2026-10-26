"""The query refiner: make a generated query parse, name known terms, and type-check.

Repairs, cheapest first, each recorded:

1. rules: declare the prefixes the query uses but forgot, and move ORDER BY,
   LIMIT and the like out of the WHERE block (hermes.sparql.validate);
2. unknown terms: a property or class the catalog does not know is replaced
   by the closest one when their names are close enough;
3. guessed entities: an entity IRI the graph does not mention is replaced by
   the entity whose label matches its name;
4. the model: a query that still does not parse, that the type check
   rejects (a class used as a property, a literal where a property holds
   entities, an entity where it holds literals), or that asks the graph for a
   fact about a named entity it does not have (one ASK query per pattern with
   a constant: does Heinrich Hoch have a pv:hasDirectReport at all?) goes
   back to the model with the problems, at most `max_repairs` times and only
   while the budget allows.
"""

import difflib
import re
from dataclasses import dataclass, field

from rdflib import Literal, URIRef

from hermes.graph.linking import linker_for, local_name
from hermes.planner import UnsupportedQuery, translate
from hermes.planner.algebra import Scan, children
from hermes.pipeline.prompt import PromptBuilder
from hermes.sparql.validate import extract_sparql, lift_modifiers, parse_error

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
SIMILAR = 0.8  # how close a known term's name must be to replace an unknown one
_PREFIXED = re.compile(r"(?<![\w<?$])([A-Za-z][\w-]*):([\w.%-]*[\w%-]|[\w%-])")


@dataclass
class Refined:
    sparql: str
    repairs: list = field(default_factory=list)
    problems: list = field(default_factory=list)  # what is still wrong, if anything


class QueryRefiner:
    def __init__(self, catalog, names, llm=None, spend=None, max_repairs=2):
        self.catalog = catalog
        self.names = names
        self.llm = llm
        self.spend = spend
        self.max_repairs = max_repairs
        self.linker = linker_for(catalog.kg)

    def __call__(self, sparql, prompt_base=None):
        refined = Refined(sparql)
        for attempt in range(self.max_repairs + 1):
            self.rules(refined)
            problems = self.problems(refined)
            if not problems or attempt == self.max_repairs or self.llm is None:
                refined.problems = problems
                return refined
            reply = self.ask(refined.sparql, problems, prompt_base)
            if reply is None:
                refined.problems = problems
                return refined
            refined.repairs.append(f"the model rewrote the query to fix: {'; '.join(problems)}")
            refined.sparql = reply
        return refined

    def ask(self, sparql, problems, prompt_base):
        builder = (prompt_base or PromptBuilder()).problems(sparql, problems)
        prompt = builder.build()
        if self.spend is not None and not self.spend.affordable(self.llm.price(prompt)):
            return None
        return extract_sparql(self.llm.complete(prompt))

    # ---- rules ----
    def rules(self, refined):
        declared = self.declare_prefixes(refined.sparql)
        if declared != refined.sparql:
            added = sorted(set(re.findall(r"PREFIX (\w[\w-]*):", declared)) -
                           set(re.findall(r"PREFIX (\w[\w-]*):", refined.sparql)))
            refined.repairs.append(f"declared the prefixes {', '.join(added)}")
            refined.sparql = declared
        lifted = lift_modifiers(refined.sparql)
        if lifted != refined.sparql:
            refined.repairs.append("moved solution modifiers out of the WHERE block")
            refined.sparql = lifted
        if parse_error(refined.sparql):
            return
        for old, new, why in self.unknown_terms(refined.sparql):
            refined.sparql = re.sub(rf"(?<![\w:<]){re.escape(old)}(?![\w:>-])", new, refined.sparql)
            refined.repairs.append(f"{old} → {new} ({why})")

    def declare_prefixes(self, sparql):
        declared = set(re.findall(r"PREFIX\s+([\w-]*):", sparql, re.IGNORECASE))
        body = re.sub(r"<[^>]*>|\"(?:[^\"\\\\]|\\\\.)*\"|'(?:[^'\\\\]|\\\\.)*'", " ", sparql)
        used = {m.group(1) for m in _PREFIXED.finditer(body)}
        known = dict(self.names.pairs)
        missing = [p for p in sorted(used - declared) if p in known]
        if not missing:
            return sparql
        return "".join(f"PREFIX {p}: <{known[p]}>\n" for p in missing) + sparql

    def unknown_terms(self, sparql):
        """(written, replacement, why) for terms the graph does not know."""
        try:
            query = translate(sparql)
        except (UnsupportedQuery, Exception):
            return []
        found, seen = [], set()
        for scan in _scans(query.root):
            for position, term in (("property", scan.p), ("class", scan.o if
                                   isinstance(scan.p, URIRef) and str(scan.p) == RDF_TYPE else None),
                                   ("entity", scan.s), ("entity", scan.o)):
                if not isinstance(term, URIRef) or str(term) in seen:
                    continue
                seen.add(str(term))
                if position == "entity" and (str(term) in self.catalog.terms or str(term) == RDF_TYPE
                                             or isinstance(scan.p, URIRef) and str(scan.p) == RDF_TYPE
                                             and term is scan.o):
                    continue
                replacement = self.replace(str(term), position)
                if replacement:
                    written = self.names.short(str(term))
                    if written.startswith("<") and written not in sparql:
                        continue
                    found.append((written if written in sparql else f"<{term}>", *replacement))
        return found

    def replace(self, iri, position):
        """A known term for an unknown one, with the reason, or None."""
        if position in ("property", "class"):
            if iri in self.catalog.terms or iri == RDF_TYPE or iri.startswith(_SCHEMA):
                return None
            kind = "property" if position == "property" else "class"
            wanted = _words(local_name(iri))
            best, score = None, 0.0
            for term in self.catalog.terms.values():
                if term.kind != kind:
                    continue
                for name in (term.label, _words(local_name(term.iri))):
                    ratio = difflib.SequenceMatcher(None, wanted, name.lower()).ratio()
                    if ratio > score:
                        best, score = term, ratio
            if best is not None and score >= SIMILAR:
                return self.names.short(best.iri), f"no {kind} {local_name(iri)}; closest name"
            return None
        if self._mentioned(iri):
            return None
        found = self.linker.search(_words(local_name(iri)), limit=1)
        if found:
            return self.names.short(found[0][0]), f"the graph has no {local_name(iri)}; " \
                                                  f"the entity named \"{found[0][1]}\""
        return None

    def _mentioned(self, iri):
        result = self.catalog.kg.try_query(f"ASK {{ {{ <{iri}> ?p ?o }} UNION {{ ?s ?p <{iri}> }} }}")
        return bool(result.get("boolean"))

    # ---- checks ----
    def problems(self, refined):
        error = parse_error(refined.sparql)
        if error:
            problems = [f"it does not parse: {error.splitlines()[0][:200]}"]
            if re.search(r"FILTER\s*\([^{}]*\(\s*SELECT\b", refined.sparql, re.IGNORECASE):
                problems.append("SPARQL has no subqueries inside FILTER; for the largest or "
                                "smallest value use ORDER BY DESC(?value) LIMIT 1, or a nested "
                                "{ SELECT (MAX(?v) AS ?max) WHERE { ... } } joined on ?max")
            if re.search(r"\bFROM\s*\{", refined.sparql, re.IGNORECASE):
                problems.append("FROM takes a graph IRI, not a pattern; write the pattern in WHERE")
            return problems
        try:
            query = translate(refined.sparql)
        except UnsupportedQuery as exc:
            return [str(exc)]
        problems = []
        bound = _bound(query.root)
        for variable in query.variables:
            if variable not in bound:
                problems.append(f"?{variable} is selected but no pattern binds it")
        for scan in _scans(query.root):
            if not isinstance(scan.p, URIRef):
                continue
            term = self.catalog.terms.get(str(scan.p))
            if term is not None and term.kind == "class":
                problems.append(f"{self.names.short(str(scan.p))} is a class, used as a property; "
                                f"write ?x a {self.names.short(str(scan.p))} to ask for its instances")
                continue
            if term is None:
                if str(scan.p) != RDF_TYPE and not str(scan.p).startswith(_SCHEMA):
                    problems.append(f"{self.names.short(str(scan.p))} is not a property of this graph")
                continue
            if term.holds == "entity" and isinstance(scan.o, Literal):
                problems.append(f"{self.names.short(term.iri)} links to entities, but the query "
                                f"gives it the literal {scan.o.n3()}; use an entity or another property")
            if term.holds == "literal" and isinstance(scan.o, URIRef):
                problems.append(f"{self.names.short(term.iri)} holds literal values, but the query "
                                f"gives it the entity {self.names.short(str(scan.o))}")
        if problems:
            return problems
        return [self.impossible(scan) for scan in _scans(query.root) if self._pinned(scan)
                and not self._holds(scan)]

    def _pinned(self, scan):
        """A pattern about a named entity, with a known predicate: cheap to check."""
        return isinstance(scan.p, URIRef) and (isinstance(scan.s, URIRef)
                                               or isinstance(scan.o, (URIRef, Literal)))

    def _holds(self, scan):
        terms = " ".join(t.n3() if not hasattr(t, "toPython") or isinstance(t, (URIRef, Literal))
                         else f"?{t}" for t in scan.terms)
        return bool(self.catalog.kg.try_query(f"ASK {{ {terms} }}").get("boolean"))

    def impossible(self, scan):
        text = " ".join(self.names.short(str(t)) if isinstance(t, URIRef) else
                        t.n3() if isinstance(t, Literal) else f"?{t}" for t in scan.terms)
        return f"the graph has no match for the pattern {text}; use another property, " \
               f"direction or entity"


_SCHEMA = ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "http://www.w3.org/2000/01/rdf-schema#",
           "http://www.w3.org/2002/07/owl#")


def _bound(node):
    """Variables the query's WHERE clause can bind, including aggregates' and BIND's."""
    from hermes.planner.algebra import Op, variables

    found = set(variables(node))
    if isinstance(node, Op):
        for child in node.children:
            found |= _bound(child)
    return found


def _scans(node):
    if isinstance(node, Scan):
        yield node
    for child in children(node):
        yield from _scans(child)


def _words(name):
    return " ".join(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).replace("_", " ").split()).lower()
