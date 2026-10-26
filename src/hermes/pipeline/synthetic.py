"""Synthetic few-shot examples, generated from the graph itself.

NL2KQL builds its few-shot database by having a model write queries over
sampled tables, explain them as questions, and keep the pairs that survive a
round trip. Here the queries come from templates over the ontology catalog,
filled with sampled entities and listed values, and kept when they return
between 1 and `MAX_ROWS` rows:

    attribute     SELECT ?v WHERE { <x> p ?v }                 p holds literals
    relation      SELECT ?y WHERE { <x> p ?y }                 p links entities
    inverse       SELECT ?x WHERE { ?x p <y> }
    value         SELECT ?x WHERE { ?x a C ; p "v" }           v a listed value
    count         SELECT (COUNT(DISTINCT ?x) AS ?n) WHERE { ?x a C ; p <y> }
    superlative   SELECT ?x ?v WHERE { ?x a C ; p ?v } ORDER BY DESC(?v) LIMIT 1
    two hops      SELECT ?z WHERE { <x> p ?y . ?y q ?z }

The model then writes the question each query answers, from the labels of
its terms (never IRIs). A pair is kept only if the pipeline, without
examples, answers that question with the same answers (F1 at least
`threshold`), so the question says what the query asks. No benchmark question
is involved, so no test answer can leak into the examples.
"""

import random
from dataclasses import dataclass

from hermes.evaluation import flatten, set_scores
from hermes.pipeline.fewshot import Example

MAX_ROWS = 50
NUMERIC = ("integer", "decimal", "double", "float", "int", "long")
VERBALIZE = """\
Here is a SPARQL query over a knowledge graph, and the names of the things it mentions.
Write the one question, in plain English, that a user would ask to get exactly these
answers. Use the names, never IRIs or variable names. Reply with the question only.

Names:
{names}

Query:
{sparql}"""


@dataclass
class Candidate:
    template: str
    sparql: str
    names: dict  # IRI -> label


class SyntheticExamples:
    def __init__(self, kg, catalog, names, llm, seed=0, threshold=0.7):
        self.kg = kg
        self.catalog = catalog
        self.names = names
        self.llm = llm
        self.random = random.Random(seed)
        self.threshold = threshold

    def build(self, count, answer, attempts=None, log=None):
        """Up to `count` examples; `answer(question)` is the pipeline's SPARQL for a question."""
        kept, tried = [], 0
        attempts = attempts or count * 4
        for candidate in self.candidates(attempts):
            if len(kept) >= count:
                break
            tried += 1
            question = self.verbalize(candidate)
            if not question:
                continue
            score = self.agree(candidate.sparql, answer(question))
            if log:
                log(f"{candidate.template:11} {score:.2f} {question}")
            if score >= self.threshold:
                kept.append(Example(question, candidate.sparql, "synthetic",
                                    {"template": candidate.template, "roundtrip_f1": round(score, 3)}))
        return kept, tried

    def verbalize(self, candidate):
        names = "\n".join(f"- {self.names.short(iri)}: {label}"
                          for iri, label in candidate.names.items())
        reply = self.llm.complete(VERBALIZE.format(names=names, sparql=candidate.sparql),
                                  "You write the questions users ask.").strip()
        question = reply.splitlines()[0].strip().strip('"') if reply else ""
        return question if question.endswith("?") and "http" not in question else ""

    def agree(self, gold, predicted):
        if not predicted:
            return 0.0
        gold_values = flatten(self.kg.try_query(gold))
        return set_scores(flatten(self.kg.try_query(predicted)), gold_values)["set_F"]

    # ---- templates ----
    def candidates(self, attempts):
        templates = [self.attribute, self.relation, self.inverse, self.value, self.count,
                     self.superlative, self.two_hops]
        for _ in range(attempts * 3):
            made = self.random.choice(templates)()
            if made is None:
                continue
            rows = len(self.kg.try_query(made.sparql)["results"]["bindings"])
            if 1 <= rows <= MAX_ROWS:
                yield made

    def _instance(self, cls):
        rows = self.kg.try_query(
            f"SELECT ?x ?l WHERE {{ ?x a <{cls.iri}> ; <http://www.w3.org/2000/01/rdf-schema#label> ?l }} "
            "LIMIT 200")["results"]["bindings"]
        if not rows:
            return None
        row = self.random.choice(rows)
        return row["x"]["value"], row["l"]["value"]

    def _property(self, holds=None, numeric=False):
        options = [p for p in self.catalog.properties() if p.subjects and p.count
                   and (holds is None or p.holds == holds) and p.label
                   and not p.iri.startswith(("http://www.w3.org/", "http://xmlns.com/foaf/0.1/depiction"))]
        if numeric:
            options = [p for p in options if any(str(o).endswith(NUMERIC) for o in p.objects)]
        return self.random.choice(options) if options else None

    def _class_of(self, prop, side="subjects"):
        classes = [self.catalog.terms[c] for c in getattr(prop, side) if c in self.catalog.terms
                   and self.catalog.terms[c].kind == "class" and self.catalog.terms[c].count]
        return self.random.choice(classes) if classes else None

    def _header(self, sparql):
        return self.names.header() + sparql

    def attribute(self):
        prop = self._property("literal")
        cls = prop and self._class_of(prop)
        found = cls and self._instance(cls)
        if not found:
            return None
        x, label = found
        return Candidate("attribute", self._header(
            f"SELECT ?value WHERE {{ {self.names.short(x)} {self.names.short(prop.iri)} ?value }}"),
            {x: label, prop.iri: prop.label})

    def relation(self):
        prop = self._property("entity")
        cls = prop and self._class_of(prop)
        found = cls and self._instance(cls)
        if not found:
            return None
        x, label = found
        return Candidate("relation", self._header(
            f"SELECT ?y WHERE {{ {self.names.short(x)} {self.names.short(prop.iri)} ?y }}"),
            {x: label, prop.iri: prop.label})

    def inverse(self):
        prop = self._property("entity")
        cls = prop and self._class_of(prop, "objects")
        found = cls and self._instance(cls)
        if not found:
            return None
        y, label = found
        return Candidate("inverse", self._header(
            f"SELECT ?x WHERE {{ ?x {self.names.short(prop.iri)} {self.names.short(y)} }}"),
            {y: label, prop.iri: prop.label})

    def value(self):
        options = [p for p in self.catalog.properties() if p.values and p.subjects]
        if not options:
            return None
        prop = self.random.choice(options)
        cls = self._class_of(prop)
        value = self.random.choice(prop.values)[0]
        if cls is None:
            return None
        literal = '"' + value.replace('"', '\\"') + '"'
        return Candidate("value", self._header(
            f"SELECT ?x WHERE {{ ?x a {self.names.short(cls.iri)} ; "
            f"{self.names.short(prop.iri)} {literal} }}"),
            {cls.iri: cls.label, prop.iri: prop.label})

    def count(self):
        prop = self._property("entity")
        cls = prop and self._class_of(prop)
        target = prop and self._class_of(prop, "objects")
        found = target and self._instance(target)
        if not (cls and found):
            return None
        y, label = found
        return Candidate("count", self._header(
            f"SELECT (COUNT(DISTINCT ?x) AS ?count) WHERE {{ ?x a {self.names.short(cls.iri)} ; "
            f"{self.names.short(prop.iri)} {self.names.short(y)} }}"),
            {cls.iri: cls.label, prop.iri: prop.label, y: label})

    def superlative(self):
        prop = self._property("literal", numeric=True)
        cls = prop and self._class_of(prop)
        if cls is None:
            return None
        order = self.random.choice(["DESC", "ASC"])
        return Candidate("superlative", self._header(
            f"SELECT ?x ?value WHERE {{ ?x a {self.names.short(cls.iri)} ; "
            f"{self.names.short(prop.iri)} ?value }} ORDER BY {order}(?value) LIMIT 1"),
            {cls.iri: cls.label, prop.iri: prop.label + (" (highest)" if order == "DESC" else
                                                           " (lowest)")})

    def two_hops(self):
        first = self._property("entity")
        cls = first and self._class_of(first)
        middle = first and self._class_of(first, "objects")
        if not (cls and middle):
            return None
        seconds = [p for p in self.catalog.properties() if middle.iri in p.subjects and p.count
                   and p.holds == "literal" and not p.iri.startswith("http://www.w3.org/")]
        found = self._instance(cls)
        if not (seconds and found):
            return None
        second = self.random.choice(seconds)
        x, label = found
        return Candidate("two hops", self._header(
            f"SELECT ?z WHERE {{ {self.names.short(x)} {self.names.short(first.iri)} ?y . "
            f"?y {self.names.short(second.iri)} ?z }}"),
            {x: label, first.iri: first.label, second.iri: second.label})
