"""Hermes in a web page: graph stores backed by the page's SPARQL engine.

Under Pyodide, Hermes runs unchanged in the browser. The page loads RDF into
Oxigraph's WebAssembly build and hands its stores to `knowledge_graph`, so
every query a system sends runs on the same engine Hermes uses for local
files (through pyoxigraph).
"""

import json

from hermes.knowledge import KnowledgeGraph
from hermes.ontology import prefixes_in
from hermes.store import EMPTY, GraphStore, QueryError

JSON_RESULTS = "application/sparql-results+json"


class BrowserStore(GraphStore):
    """A store backed by a JavaScript oxigraph.Store."""

    local = True

    def __init__(self, store):
        self.store = store

    def query(self, sparql):
        from pyodide.ffi import to_js
        from js import Object

        options = to_js({"results_format": JSON_RESULTS}, dict_converter=Object.fromEntries)
        try:
            result = self.store.query(sparql, options)
        except Exception as exc:  # the engine's parse or evaluation error
            raise QueryError(str(exc).strip()) from exc
        if not isinstance(result, str):  # CONSTRUCT and DESCRIBE give quads
            return EMPTY
        return json.loads(result)

    def __repr__(self):
        return "BrowserStore()"


def knowledge_graph(data, schema=None, texts=(), name=""):
    """A KnowledgeGraph over the page's stores; `texts` are the RDF files, for their prefixes."""
    return KnowledgeGraph(
        BrowserStore(data),
        schema=BrowserStore(schema) if schema is not None else None,
        prefixes=prefixes_in(*texts),
        name=name,
    )
