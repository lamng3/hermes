"""Prefixes of a knowledge graph: short names for IRIs, and the PREFIX lines for queries."""

import re

CORE = (
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
)
# Local names plain enough for every engine. SPARQL allows '.' and %-escapes in them, but
# Oxigraph 0.5 rejects some (`p:a.b%40c.org`), so IRIs with them are written whole.
_LOCAL = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]*")


class Prefixes:
    def __init__(self, pairs):
        seen, self.pairs = set(), []
        for prefix, namespace in list(pairs) + list(CORE):
            if prefix not in seen and namespace not in {ns for _, ns in self.pairs}:
                seen.add(prefix)
                self.pairs.append((prefix, namespace))
        # Longest namespace first, so that the most specific prefix wins.
        self.by_length = sorted(self.pairs, key=lambda pair: -len(pair[1]))

    @classmethod
    def of(cls, kg):
        declared = list(kg.prefixes)
        if kg.vocabulary_limit:
            declared += list(kg.vocabulary["prefixes"])
        return cls(declared)

    def short(self, iri):
        """`prefix:local` when the local name allows it, else `<iri>`."""
        for prefix, namespace in self.by_length:
            if iri.startswith(namespace) and _LOCAL.fullmatch(iri[len(namespace):]):
                return f"{prefix}:{iri[len(namespace):]}"
        return f"<{iri}>"

    def expand(self, name):
        name = name.strip()
        if name.startswith("<") and name.endswith(">"):
            return name[1:-1]
        prefix, sep, local = name.partition(":")
        for known, namespace in self.pairs:
            if sep and known == prefix and not local.startswith("//"):
                return namespace + local
        return name

    def header(self):
        return "".join(f"PREFIX {p}: <{ns}>\n" for p, ns in self.pairs)
