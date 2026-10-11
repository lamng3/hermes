"""A knowledge graph as systems see it: a store to query and a schema to read."""

from functools import cached_property
from pathlib import Path

from hermes.ontology import declared_prefixes, vocabulary
from hermes.store import CachedStore, EndpointStore, local_store


class KnowledgeGraph:
    """A graph store plus the vocabulary used to prompt for it.

    `store` runs queries over the instance data. `schema` is where the
    vocabulary is read from; it defaults to the store, and can be a smaller
    store holding only the ontology, as for DBpedia. `prefixes` are the
    preferred prefixes, in order.
    """

    def __init__(self, store, schema=None, prefixes=(), name="", vocabulary_limit=60):
        self.store = store if isinstance(store, CachedStore) else CachedStore(store)
        self.schema = schema or self.store
        self._prefixes = prefixes
        self.name = name
        self.vocabulary_limit = vocabulary_limit

    @classmethod
    def from_files(cls, *paths, schema=(), name="", engine="oxigraph"):
        """Instance data and schema from local RDF files."""
        schema_store = local_store(*schema, engine=engine) if schema else None
        files = tuple(schema) + tuple(paths)
        return cls(
            local_store(*paths, engine=engine),
            schema=schema_store,
            prefixes=lambda: declared_prefixes(*files),
            name=name or Path(paths[0]).stem,
        )

    @classmethod
    def from_endpoint(cls, url, schema=(), prefixes=(), name="", timeout=60.0):
        schema_store = local_store(*schema) if schema else None
        return cls(EndpointStore(url, timeout=timeout), schema_store, prefixes, name or url)

    @cached_property
    def prefixes(self):
        declared = self._prefixes() if callable(self._prefixes) else self._prefixes
        return list(declared)

    @cached_property
    def vocabulary(self):
        return vocabulary(self.schema, self.prefixes, self.vocabulary_limit)

    def query(self, sparql):
        return self.store.query(sparql)

    def try_query(self, sparql):
        return self.store.try_query(sparql)

    @property
    def executions(self):
        return self.store.executions

    def __repr__(self):
        return f"KnowledgeGraph({self.name!r}, {self.store.store!r})"


def as_knowledge_graph(value):
    """Accept a KnowledgeGraph, or a path to an RDF file."""
    if isinstance(value, KnowledgeGraph):
        return value
    if isinstance(value, (str, Path)):
        return KnowledgeGraph.from_files(value)
    raise TypeError("Expected a KnowledgeGraph or a path to an RDF file.")
