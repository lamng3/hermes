"""Graph stores: where SPARQL runs.

Systems and the evaluator talk to a store, never to a file format or a server
directly. Every store answers `query(sparql)` with SPARQL 1.1 JSON results, so
local files and remote endpoints are interchangeable.

- `OxigraphStore` loads RDF files into Oxigraph, a fast in-memory engine.
  It is the default for local files.
- `LocalStore` loads RDF files into rdflib, in pure Python.
- `EndpointStore` posts to a SPARQL endpoint over HTTP.
- `CachedStore` wraps any store, memoizes results and counts executions.
"""

import json
import time
from pathlib import Path

import httpx

EMPTY = {"head": {"vars": []}, "results": {"bindings": []}}


class QueryError(RuntimeError):
    """A query failed to parse or to execute."""


class GraphStore:
    """Interface: run one query, return SPARQL 1.1 JSON results."""

    local = False

    def query(self, sparql):
        raise NotImplementedError

    def try_query(self, sparql):
        """Like query, but return empty results instead of raising."""
        try:
            return self.query(sparql)
        except QueryError:
            return EMPTY


class LocalStore(GraphStore):
    """RDF files loaded into an in-memory rdflib graph."""

    local = True

    def __init__(self, *paths):
        self.paths = [Path(path) for path in paths]
        self._graph = None

    @property
    def graph(self):
        if self._graph is None:
            from rdflib import Graph

            graph = Graph()
            for path in self.paths:
                graph.parse(path, format=_rdf_format(path))
            self._graph = graph
        return self._graph

    def query(self, sparql):
        try:
            result = self.graph.query(sparql)
            if result.type not in ("SELECT", "ASK"):
                return EMPTY
            return json.loads(result.serialize(format="json"))
        except Exception as exc:
            raise QueryError(str(exc).strip()) from exc

    def __repr__(self):
        return f"LocalStore({', '.join(str(path) for path in self.paths)})"


class OxigraphStore(GraphStore):
    """RDF files loaded into an in-memory Oxigraph store."""

    local = True

    def __init__(self, *paths):
        self.paths = [Path(path) for path in paths]
        self._store = None

    @property
    def store(self):
        if self._store is None:
            import pyoxigraph

            store = pyoxigraph.Store()
            for path in self.paths:
                store.bulk_load(path=str(path), format=_oxigraph_format(path))
            self._store = store
        return self._store

    def query(self, sparql):
        import pyoxigraph

        try:
            result = self.store.query(sparql)
            if isinstance(result, pyoxigraph.QueryTriples):
                return EMPTY
            return json.loads(result.serialize(format=pyoxigraph.QueryResultsFormat.JSON))
        except (SyntaxError, OSError, ValueError, RuntimeError) as exc:
            raise QueryError(str(exc).strip()) from exc

    def __repr__(self):
        return f"OxigraphStore({', '.join(str(path) for path in self.paths)})"


def local_store(*paths, engine="oxigraph"):
    """Factory for file-backed stores."""
    if engine == "rdflib":
        return LocalStore(*paths)
    if engine == "oxigraph":
        return OxigraphStore(*paths)
    raise ValueError(f"Unknown engine {engine!r}. Use oxigraph or rdflib.")


class EndpointStore(GraphStore):
    """A remote SPARQL 1.1 endpoint."""

    def __init__(self, url, timeout=60.0, retries=2, max_bytes=50_000_000):
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.max_bytes = max_bytes

    def query(self, sparql):
        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": "hermes-sparql (https://github.com/lamng3/hermes)",
        }
        for attempt in range(self.retries + 1):
            try:
                with httpx.stream(
                    "POST", self.url, data={"query": sparql}, headers=headers,
                    timeout=self.timeout,
                ) as response:
                    if response.status_code >= 500 and attempt < self.retries:
                        time.sleep(2**attempt)
                        continue
                    body = self._read(response)
                if response.status_code >= 400:
                    text = body[:300].decode(errors="replace")
                    raise QueryError(f"HTTP {response.status_code}: {text}")
                return json.loads(body)
            except (httpx.TransportError, json.JSONDecodeError) as exc:
                if attempt == self.retries:
                    raise QueryError(f"{self.url}: {exc}") from exc
                time.sleep(2**attempt)
        raise QueryError(f"{self.url}: no response")

    def _read(self, response):
        """Read the body, refusing results larger than max_bytes."""
        chunks, size = [], 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > self.max_bytes:
                raise QueryError(
                    f"result larger than {self.max_bytes // 1_000_000} MB; "
                    "add constraints or a LIMIT"
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def __repr__(self):
        return f"EndpointStore({self.url})"


class CachedStore(GraphStore):
    """Decorator: memoize results and count real executions."""

    def __init__(self, store):
        self.store = store
        self.local = store.local
        self.executions = 0
        self.seconds = 0.0
        self._cache = {}

    def query(self, sparql):
        if sparql not in self._cache:
            self.executions += 1
            started = time.perf_counter()
            try:
                self._cache[sparql] = self.store.query(sparql)
            except QueryError as exc:
                self._cache[sparql] = exc
            self.seconds += time.perf_counter() - started
        cached = self._cache[sparql]
        if isinstance(cached, QueryError):
            raise cached
        return cached

    def __repr__(self):
        return f"CachedStore({self.store!r})"


def _oxigraph_format(path):
    import pyoxigraph

    names = {
        "nt": pyoxigraph.RdfFormat.N_TRIPLES,
        "nquads": pyoxigraph.RdfFormat.N_QUADS,
        "xml": pyoxigraph.RdfFormat.RDF_XML,
        "trig": pyoxigraph.RdfFormat.TRIG,
        "turtle": pyoxigraph.RdfFormat.TURTLE,
    }
    name = _rdf_format(path)
    if name not in names:
        raise ValueError(f"Oxigraph cannot load {path}; use engine: rdflib.")
    return names[name]


def _rdf_format(path):
    suffixes = "".join(Path(path).suffixes).lower()
    for suffix, name in (
        (".nt", "nt"),
        (".nq", "nquads"),
        (".rdf", "xml"),
        (".owl", "xml"),
        (".jsonld", "json-ld"),
        (".trig", "trig"),
    ):
        if suffix in suffixes:
            return name
    return "turtle"
