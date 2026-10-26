"""Read a compact vocabulary (classes, properties, prefixes) for prompting.

The vocabulary is queried through a graph store, so it works the same for a
local Turtle file and for a remote endpoint.
"""

import re
from pathlib import Path

_DECLARED_PREFIX = re.compile(
    r"@?prefix\s+([A-Za-z][\w.-]*)\s*:\s*<([^>]+)>",
    re.IGNORECASE,
)
_CLASSES = """
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?term ?label WHERE {
  { ?term a owl:Class } UNION { ?term a rdfs:Class }
  OPTIONAL { ?term rdfs:label ?label }
  FILTER(isIRI(?term))
}
"""
_PROPERTIES = """
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?term ?label WHERE {
  { ?term a owl:ObjectProperty } UNION
  { ?term a owl:DatatypeProperty } UNION
  { ?term a rdf:Property }
  OPTIONAL { ?term rdfs:label ?label }
  FILTER(isIRI(?term))
}
"""


def load_vocabulary(ontology_path, limit=60):
    """Vocabulary of one RDF file."""
    from hermes.store import local_store

    return vocabulary(local_store(ontology_path), declared_prefixes(ontology_path), limit)


def vocabulary(store, declared=(), limit=60):
    classes = _labelled(store, _CLASSES, limit) if limit else []
    properties = _labelled(store, _PROPERTIES, limit) if limit else []
    declared = [
        (prefix, namespace)
        for prefix, namespace in declared
        if not _is_core_namespace(namespace)
    ][:12]
    iris = [term for term, _label in classes + properties]
    prefixes, by_namespace = _prefixes_for_terms(iris, declared)
    return {
        "classes": [_term_line(term, label, by_namespace) for term, label in classes],
        "properties": [
            _term_line(term, label, by_namespace) for term, label in properties
        ],
        "prefixes": prefixes,
    }


def declared_prefixes(*paths):
    """Prefixes as written in the files, in order, without duplicates."""
    return prefixes_in(*(Path(p).read_text(encoding="utf-8", errors="replace") for p in paths))


def prefixes_in(*texts):
    """Prefixes as written in RDF texts, in order, without duplicates."""
    found = []
    seen = set()
    for text in texts:
        for prefix, namespace in _DECLARED_PREFIX.findall(text):
            key = prefix.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append((prefix, namespace))
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


def _labelled(store, query, limit):
    rows = []
    seen = set()
    result = store.try_query(query)
    for binding in result.get("results", {}).get("bindings", []):
        iri = binding["term"]["value"]
        if iri in seen:
            continue
        seen.add(iri)
        rows.append((iri, binding.get("label", {}).get("value", "")))
        if len(rows) >= limit:
            break
    return rows


def _local_name(term):
    value = str(term)
    if "#" in value:
        return value.rsplit("#", 1)[-1]
    return value.rsplit("/", 1)[-1]
