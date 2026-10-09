"""Load a Turtle ontology into a compact vocabulary for prompting."""

import re
from pathlib import Path

from rdflib import Graph
from rdflib.namespace import OWL, RDF, RDFS

_DECLARED_PREFIX = re.compile(
    r"@?prefix\s+([A-Za-z][\w.-]*)\s*:\s*<([^>]+)>",
    re.IGNORECASE,
)


def load_vocabulary(ontology_path, limit=60):
    path = Path(ontology_path)
    graph = Graph()
    graph.parse(path, format="turtle")
    classes = _labelled(
        graph,
        """
        SELECT ?term ?label WHERE {
          { ?term a owl:Class } UNION { ?term a rdfs:Class }
          OPTIONAL { ?term rdfs:label ?label }
          FILTER(isIRI(?term))
        }
        """,
        limit,
    )
    properties = _labelled(
        graph,
        """
        SELECT ?term ?label WHERE {
          { ?term a owl:ObjectProperty } UNION
          { ?term a owl:DatatypeProperty } UNION
          { ?term a rdf:Property }
          OPTIONAL { ?term rdfs:label ?label }
          FILTER(isIRI(?term))
        }
        """,
        limit,
    )
    declared = _declared_prefixes(path)
    iris = [term for term, _label in classes + properties]
    prefixes, by_namespace = _prefixes_for_terms(iris, declared)
    return {
        "path": str(path),
        "classes": [_term_line(term, label, by_namespace) for term, label in classes],
        "properties": [
            _term_line(term, label, by_namespace) for term, label in properties
        ],
        "prefixes": prefixes,
    }


def _declared_prefixes(ontology_path):
    text = Path(ontology_path).read_text(encoding="utf-8", errors="replace")
    found = []
    seen = set()
    for prefix, namespace in _DECLARED_PREFIX.findall(text):
        key = prefix.lower()
        if key in seen or _is_core_namespace(namespace):
            continue
        seen.add(key)
        found.append((prefix, namespace))
        if len(found) >= 12:
            break
    return found


def _prefixes_for_terms(iris, declared):
    by_namespace = {namespace: prefix for prefix, namespace in declared}
    used = {prefix.lower() for prefix, _namespace in declared}
    prefixes = list(declared)
    for iri in iris:
        namespace = _namespace(iri)
        if namespace in by_namespace or _is_core_namespace(namespace):
            continue
        prefix = _prefix_name(namespace, used)
        by_namespace[namespace] = prefix
        used.add(prefix.lower())
        prefixes.append((prefix, namespace))
        if len(prefixes) >= 12:
            break
    return prefixes, by_namespace


def _is_core_namespace(namespace):
    return namespace.startswith(
        (
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "http://www.w3.org/2000/01/rdf-schema#",
            "http://www.w3.org/2002/07/owl#",
            "http://www.w3.org/2001/XMLSchema#",
            "http://www.w3.org/XML/1998/namespace",
        )
    )


def _namespace(iri):
    text = str(iri)
    if "#" in text:
        return text.rsplit("#", 1)[0] + "#"
    return text.rsplit("/", 1)[0] + "/"


def _prefix_name(namespace, used):
    segment = namespace.rstrip("#/").rsplit("/", 1)[-1]
    slug = re.sub(r"[^A-Za-z0-9-]", "", segment).lower()[:16] or "ns"
    if not slug[0].isalpha():
        slug = f"ns{slug}"
    candidate = slug
    number = 2
    while candidate.lower() in used:
        candidate = f"{slug}{number}"
        number += 1
    return candidate


def _term_line(term, label, by_namespace):
    namespace = _namespace(term)
    local = _local_name(term)
    prefix = by_namespace.get(namespace)
    name = f"{prefix}:{local}" if prefix else f"<{term}>"
    if label:
        return f'{name} "{label}"'
    return name


def _labelled(graph, query, limit):
    rows = []
    seen = set()
    for term, label in graph.query(query, initNs={"owl": OWL, "rdfs": RDFS, "rdf": RDF}):
        iri = str(term)
        if iri in seen:
            continue
        seen.add(iri)
        rows.append((iri, str(label) if label else ""))
        if len(rows) >= limit:
            break
    return rows


def _local_name(term):
    value = str(term)
    if "#" in value:
        return value.rsplit("#", 1)[-1]
    return value.rsplit("/", 1)[-1]
