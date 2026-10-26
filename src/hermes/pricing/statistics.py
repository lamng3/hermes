"""Statistics of a knowledge graph, and System R estimates of triple patterns.

A local graph is summarized once with grouped queries: per predicate, its
triples, distinct subjects, distinct objects and literal objects; per class,
its instances. An endpoint such as DBpedia is asked about each predicate or
class the first time an estimate needs it, and the answer is kept; a term the
endpoint cannot count is priced with a default and reported as unknown.

`scan` estimates one triple pattern as System R would (Selinger et al.,
SIGMOD 1979): a constant keeps 1/distinct of its predicate's triples.
"""

import math
import threading
import weakref
from dataclasses import dataclass

from hermes.sparql.verifier import TYPE

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
    literals: float = 0.0  # triples whose object is a literal

    @property
    def kind(self):
        """literal or entity when every object is one, else None."""
        if not self.triples:
            return None
        if self.literals == self.triples:
            return "literal"
        return "entity" if self.literals == 0 else None


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
            f"SELECT ?p {_COUNTS} WHERE {{ ?s ?p ?o }} GROUP BY ?p"
        )["results"]["bindings"]
        for row in rows:
            self.predicates[row["p"]["value"]] = _count(row)
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
            f"SELECT {_COUNTS} WHERE {{ ?s <{iri}> ?o }}"
        )["results"]["bindings"]
        count = _count(rows[0]) if rows and "n" in rows[0] else None
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


def scan(stats, subject, predicate, value, prop, cls=None, unknown=None):
    """System R estimate of one triple pattern: its rows, and distinct values per variable.

    `subject`, `predicate` and `value` are the pattern's terms as written, where
    variables start with ? or $. `prop` is the predicate's IRI, or None for a
    variable or a path; `cls` is the class IRI of `?x rdf:type C`. Terms the
    store could not count are appended to `unknown` and priced with a default.
    """
    unknown = [] if unknown is None else unknown
    if cls is not None:
        count = stats.instances(cls)
        if count is None:
            unknown.append(value)
            count = DEFAULT_ROWS
        rows = count if _variable(subject) else min(1.0, count)
        return rows, _distinct({subject: rows} if _variable(subject) else {})
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


def _distinct(found):
    return {var: max(value, 0.0) for var, value in found.items()}


def _variable(token):
    return bool(token) and token[0] in "?$"


def _path(token):
    return any(mark in token for mark in "/|^*+") and not token.startswith("<")


_COUNTS = ("(COUNT(*) AS ?n) (COUNT(DISTINCT ?s) AS ?ds) (COUNT(DISTINCT ?o) AS ?do) "
           "(SUM(IF(isLiteral(?o), 1, 0)) AS ?lit)")


def _count(row):
    return Count(_number(row, "n"), _number(row, "ds"), _number(row, "do"), _number(row, "lit"))


def _number(row, name):
    try:
        return float(row[name]["value"])
    except (KeyError, ValueError):
        return 0.0
