"""The ontology catalog: every class and property a question may need, described.

It plays the part of NL2KQL's semantic data catalog, with the ontology as
the schema and the knowledge graph as the data. From the ontology: classes
and properties with their labels, comments, declared domains and ranges.
From the data (hermes.pricing.statistics, and for local graphs a few grouped
queries): how often each term is used, which properties the instances of a
class use, which classes a property's objects belong to, whether a property
holds literals or entities, and the values of properties with few distinct
literal values (NL2KQL's enum values), such as countries or currencies.
"""

import threading
import weakref
from dataclasses import dataclass, field

from hermes.graph.linking import local_name
from hermes.pricing.statistics import statistics_for
from hermes.retrieval import Document

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
CORE = (RDF, RDFS, OWL, "http://www.w3.org/2001/XMLSchema#")
LABEL = RDFS + "label"
MAX_VALUES = 100  # a literal property with at most this many distinct values lists them
MAX_MEMBERS = 100  # a class with at most this many instances lists them, with their labels
NUMBERS = ("integer", "decimal", "double", "float", "int", "long", "short", "byte", "date")
_CATALOGS = weakref.WeakKeyDictionary()
_LOCK = threading.Lock()

_TERMS = """
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?term ?kind ?label ?comment ?domain ?range WHERE {
  VALUES (?type ?kind) { (owl:Class "class") (rdfs:Class "class")
    (owl:ObjectProperty "property") (owl:DatatypeProperty "property") (rdf:Property "property") }
  ?term a ?type
  OPTIONAL { ?term rdfs:label ?label FILTER(lang(?label) = "" || langMatches(lang(?label), "en")) }
  OPTIONAL { ?term rdfs:comment ?comment
             FILTER(lang(?comment) = "" || langMatches(lang(?comment), "en")) }
  OPTIONAL { ?term rdfs:domain ?domain FILTER(isIRI(?domain)) }
  OPTIONAL { ?term rdfs:range ?range FILTER(isIRI(?range)) }
  FILTER(isIRI(?term))
}
"""
_USED_BY = """SELECT ?c ?p (COUNT(*) AS ?n) WHERE { ?s a ?c ; ?p ?o } GROUP BY ?c ?p"""
_POINTS_TO = """SELECT ?p ?c (COUNT(*) AS ?n) WHERE { ?s ?p ?o . ?o a ?c } GROUP BY ?p ?c"""


@dataclass
class Term:
    iri: str
    kind: str  # class or property
    labels: set = field(default_factory=set)
    comment: str = ""
    count: float = 0.0  # instances of a class, triples of a property
    domain: set = field(default_factory=set)  # declared
    range: set = field(default_factory=set)  # declared
    used_by: dict = field(default_factory=dict)  # property: classes whose instances use it
    points_to: dict = field(default_factory=dict)  # property: classes of its objects
    values: list = field(default_factory=list)  # property: (value, count) when few
    members: list = field(default_factory=list)  # class: (iri, label) when few
    holds: str = ""  # property: literal or entity, from the data

    @property
    def label(self):
        return sorted(self.labels, key=len)[0] if self.labels else _words(local_name(self.iri))

    @property
    def subjects(self):
        return set(self.domain) | set(self.used_by)

    @property
    def objects(self):
        return set(self.range) | set(self.points_to)


class OntologyCatalog:
    """Classes and properties of a knowledge graph, with documents to retrieve them by."""

    def __init__(self, kg, values=True):
        self.kg = kg
        self.statistics = statistics_for(kg)
        self.terms = {}
        self._read_ontology()
        self._read_data(values and kg.store.local)

    def _read_ontology(self):
        for row in self.kg.schema.try_query(_TERMS)["results"]["bindings"]:
            iri = row["term"]["value"]
            if iri.startswith(CORE):
                continue
            term = self.terms.setdefault(iri, Term(iri, row["kind"]["value"]))
            if "label" in row:
                term.labels.add(row["label"]["value"])
            if "comment" in row and not term.comment:
                term.comment = row["comment"]["value"]
            if "domain" in row:
                term.domain.add(row["domain"]["value"])
            if "range" in row:
                term.range.add(row["range"]["value"])

    def _read_data(self, local):
        stats = self.statistics
        for iri, count in stats.predicates.items():
            if count is None or (iri.startswith(CORE) and iri != LABEL):
                continue
            term = self.terms.setdefault(iri, Term(iri, "property"))
            term.count = count.triples
            term.holds = count.kind or ""
        for iri, count in stats.classes.items():
            if count is None or iri.startswith(CORE):
                continue
            self.terms.setdefault(iri, Term(iri, "class")).count = count
        if not local:
            return
        for query, slot in ((_USED_BY, "used_by"), (_POINTS_TO, "points_to")):
            for row in self.kg.try_query(query)["results"]["bindings"]:
                prop, cls = self.terms.get(row["p"]["value"]), row["c"]["value"]
                if prop is not None and prop.kind == "property" and not cls.startswith(CORE):
                    getattr(prop, slot)[cls] = float(row["n"]["value"])
        for prop in self.properties():
            count = stats.predicates.get(prop.iri)
            # Enumerable: few distinct literal values, each used more than once on average.
            if (count is None or prop.holds != "literal" or count.objects > MAX_VALUES
                    or count.triples < 2 * count.objects):
                continue
            rows = self.kg.try_query(
                f"SELECT ?o (COUNT(*) AS ?n) WHERE {{ ?s <{prop.iri}> ?o }} "
                f"GROUP BY ?o ORDER BY DESC(?n) LIMIT {MAX_VALUES}")["results"]["bindings"]
            if any(r["o"].get("datatype", "").endswith(NUMBERS) for r in rows):
                continue
            prop.values = [(r["o"]["value"], int(float(r["n"]["value"]))) for r in rows]
        for cls in self.classes():
            if not 0 < cls.count <= MAX_MEMBERS:
                continue
            rows = self.kg.try_query(
                f"SELECT ?x (SAMPLE(?l) AS ?label) WHERE {{ ?x a <{cls.iri}> ; <{LABEL}> ?l }} "
                f"GROUP BY ?x")["results"]["bindings"]
            cls.members = [(r["x"]["value"], r["label"]["value"]) for r in rows if "label" in r]

    def classes(self):
        return [t for t in self.terms.values() if t.kind == "class"]

    def properties(self):
        return [t for t in self.terms.values() if t.kind == "property"]

    def name(self, iri):
        """A term's label, for describing other terms."""
        term = self.terms.get(iri)
        return term.label if term else _words(local_name(iri))

    def documents(self):
        """One document per term, and one per listed value of a property."""
        docs = []
        for term in self.terms.values():
            text = " ".join([*term.labels, _words(local_name(term.iri)), term.comment])
            if term.kind == "property":
                text += " " + " ".join(self.name(c) for c in term.subjects | term.objects)
            docs.append(Document(text, term, {"kind": term.kind}))
            for value, count in term.values:
                docs.append(Document(f"{value} {term.label}", (term, value),
                                     {"kind": "value", "count": count}))
            for iri, label in term.members:
                docs.append(Document(label, (term, iri, label), {"kind": "member"}))
        return docs


def catalog_for(kg):
    """One catalog per knowledge graph: reading a large ontology takes seconds."""
    with _LOCK:
        catalog = _CATALOGS.get(kg)
        if catalog is None:
            catalog = _CATALOGS[kg] = OntologyCatalog(kg)
        return catalog


def _words(name):
    import re

    return " ".join(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).replace("_", " ").split())
