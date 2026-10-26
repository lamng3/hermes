"""The ontology refiner: the slice of the ontology a question needs.

The classes, properties and listed values that rank highest for the
question, the classes of the entities it names, then the
neighbourhood that makes them usable: the properties the chosen classes'
instances use, and the classes at both ends of the chosen properties.
"""

from dataclasses import dataclass, field

from hermes.graph.linking import linker_for
from hermes.retrieval import Lexical, words


@dataclass
class Entity:
    iri: str
    label: str
    classes: list = field(default_factory=list)


@dataclass
class OntologySlice:
    classes: list  # Terms
    properties: list  # Terms
    values: list  # (Term, value)
    entities: list  # Entity

    def render(self, catalog, names):
        """The slice as prompt text."""
        lines = ["Classes:"]
        for term in self.classes:
            about = f": {term.comment}" if term.comment else ""
            lines.append(f"- {names.short(term.iri)} \"{term.label}\" "
                         f"({int(term.count):,} instances){about}")
        lines.append("Properties (subject → object):")
        for term in self.properties:
            subjects = ", ".join(sorted(catalog.name(c) for c in term.subjects)[:3]) or "?"
            objects = ", ".join(sorted(catalog.name(c) for c in term.objects)[:3]) or \
                ("a value" if term.holds == "literal" else "?")
            lines.append(f"- {names.short(term.iri)} \"{term.label}\": {subjects} → {objects}")
        if self.values:
            lines.append("Values that occur in the graph:")
            for term, value in self.values:
                lines.append(f"- {names.short(term.iri)} \"{value}\"")
        if self.entities:
            lines.append("Entities named in the question:")
            for entity in self.entities:
                kinds = ", ".join(catalog.name(c) for c in entity.classes[:2])
                lines.append(f"- {names.short(entity.iri)} \"{entity.label}\""
                             + (f" ({kinds})" if kinds else ""))
        return "\n".join(lines)


class OntologyRefiner:
    def __init__(self, catalog, retriever=None, classes=6, properties=16, values=6, entities=8):
        self.catalog = catalog
        self.retriever = (retriever or Lexical()).fit(catalog.documents())
        self.limits = {"classes": classes, "properties": properties, "values": values,
                       "entities": entities}
        self.linker = linker_for(catalog.kg)

    def __call__(self, question):
        hits = self.retriever.rank(question, k=200)
        classes = [h.document.item for h in hits if h.document.meta["kind"] == "class"]
        properties = [h.document.item for h in hits if h.document.meta["kind"] == "property"]
        values = [h.document.item for h in hits if h.document.meta["kind"] == "value"]
        asked = set(words(question))
        # A member counts when the question names most of it, not just its class ("city").
        members = [h.document.item for h in hits if h.document.meta["kind"] == "member"
                   and _covered(h.document.item[2], asked)]
        entities = self.entities(question, members[: self.limits["values"]])
        limit = self.limits
        chosen_classes = _unique([*classes[: limit["classes"]],
                                  *(self.catalog.terms[c] for e in entities for c in e.classes
                                    if c in self.catalog.terms)])
        chosen_values = values[: limit["values"]]
        chosen_properties = _unique([*properties[: limit["properties"] // 2],
                                     *(term for term, _ in chosen_values)])
        # The neighbourhood: what the chosen classes' instances use, by how often.
        used = sorted((p for p in self.catalog.properties()
                       if p.subjects & {c.iri for c in chosen_classes}),
                      key=lambda p: -p.count)
        chosen_properties = _unique([*chosen_properties, *used])[: limit["properties"]]
        ends = [self.catalog.terms[c] for p in chosen_properties for c in p.subjects | p.objects
                if c in self.catalog.terms and self.catalog.terms[c].kind == "class"]
        chosen_classes = _unique([*chosen_classes, *ends])[: limit["classes"] + 4]
        return OntologySlice(chosen_classes, chosen_properties, chosen_values, entities)

    def entities(self, question, members=()):
        """Entities the question names, with their classes.

        Exact label matches first, then members of small classes the question
        matches (a product category, a department), then partial matches of
        its capitalized mentions ("Ms. Brant" finds "Karen Brant").
        """
        iris = list(self.linker.link(question))
        iris += [iri for _cls, iri, _label in members]
        for mention in mentions(question):
            iris += [iri for iri, _label in self.linker.search(mention, limit=2)]
        iris = [i for i in dict.fromkeys(iris) if i not in self.catalog.terms]
        iris = iris[: self.limits["entities"]]
        if not iris:
            return []
        values = " ".join(f"<{iri}>" for iri in iris)
        rows = self.catalog.kg.try_query(
            "SELECT ?e ?c ?l WHERE { VALUES ?e { " + values + " } "
            "OPTIONAL { ?e a ?c } OPTIONAL { ?e <http://www.w3.org/2000/01/rdf-schema#label> ?l } }"
        )["results"]["bindings"]
        found = {iri: Entity(iri, "") for iri in iris}
        for row in rows:
            entity = found[row["e"]["value"]]
            if "c" in row and row["c"]["value"] not in entity.classes:
                entity.classes.append(row["c"]["value"])
            if "l" in row and not entity.label:
                entity.label = row["l"]["value"]
        for entity in found.values():
            entity.label = entity.label or entity.iri.rsplit("/", 1)[-1]
        return list(found.values())


def mentions(question):
    """Runs of capitalized words after the first word: likely names of things."""
    import re

    words = re.findall(r"[\w'.-]+", question)
    found, run = [], []
    for index, word in enumerate(words):
        if index and word[:1].isupper() and word.rstrip(".").lower() not in ("i",):
            run.append(word)
        elif run:
            found.append(" ".join(run))
            run = []
    if run:
        found.append(" ".join(run))
    return found


def _covered(label, asked):
    named = set(words(label))
    return bool(named) and len(named & asked) / len(named) >= 0.5


def _unique(items):
    seen, out = set(), []
    for item in items:
        if id(item) not in seen:
            seen.add(id(item))
            out.append(item)
    return out
