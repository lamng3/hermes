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
import threading
import time
from pathlib import Path

import httpx

from hermes import runtime

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
        self._lock = threading.Lock()

    @property
    def graph(self):
        with self._lock:
            return self._load()

    def _load(self):
        if self._graph is None:
            from rdflib import Graph

            graph = Graph()
            for path in self.paths:
                graph.parse(path, format=_rdf_format(path))
            self._graph = graph
        return self._graph

    def query(self, sparql):
        from hermes.validate import _PARSE_LOCK  # rdflib's parser is not thread-safe

        try:
            with _PARSE_LOCK:
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
        self._lock = threading.Lock()

    @property
    def store(self):
        with self._lock:
            return self._load()

    def _load(self):
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


def local_store(*paths, engine="oxigraph", timeout=None):
    """Factory for file-backed stores; with a timeout, queries run in a worker."""
    if engine not in ("oxigraph", "rdflib"):
        raise ValueError(f"Unknown engine {engine!r}. Use oxigraph or rdflib.")
    if timeout:
        return TimeoutStore(paths, engine, timeout)
    return LocalStore(*paths) if engine == "rdflib" else OxigraphStore(*paths)


def _worker(paths, engine, connection):
    store = local_store(*paths, engine=engine)
    while True:
        sparql = connection.recv()
        if sparql is None:
            return
        try:
            connection.send((True, store.query(sparql)))
        except QueryError as exc:
            connection.send((False, str(exc)))


class TimeoutStore(GraphStore):
    """Local files queried in a worker process, killed past a timeout.

    Neither Oxigraph nor rdflib can cancel a running query, so a query that
    explodes (a cartesian product, say) would otherwise hold the run and its
    memory forever. The worker is restarted after a timeout; reloading a
    graph of tens of thousands of triples takes well under a second.
    """

    local = True

    def __init__(self, paths, engine="oxigraph", timeout=30.0):
        self.paths = [str(path) for path in paths]
        self.engine = engine
        self.timeout = timeout
        self._process = None
        self._connection = None
        self._lock = threading.Lock()  # one worker, one query at a time

    def _start(self):
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(
            target=_worker, args=(self.paths, self.engine, child), daemon=True
        )
        self._process.start()

    def _stop(self):
        if self._process is not None:
            self._process.kill()
            self._process.join()
        self._process = None

    def query(self, sparql):
        with self._lock:
            if self._process is None or not self._process.is_alive():
                self._start()
            self._connection.send(sparql)
            if not self._connection.poll(self.timeout):
                self._stop()
                raise QueryError(f"query timed out after {self.timeout:g} s")
            ok, value = self._connection.recv()
        if not ok:
            raise QueryError(value)
        return value

    def close(self):
        if self._process is not None and self._process.is_alive():
            self._connection.send(None)
            self._process.join(timeout=5)
        self._stop()

    def __del__(self):
        try:
            self._stop()
        except Exception:
            pass

    def __repr__(self):
        return f"TimeoutStore({', '.join(self.paths)}, timeout={self.timeout:g}s)"


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
                    timeout=runtime.remaining(self.timeout),
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
    """Decorator: memoize results and count real executions.

    The cache is bounded: it keeps the `max_entries` most recently used
    results and never keeps a result with more than `max_rows` rows, so a
    long run's memory stays flat.
    """

    def __init__(self, store, max_entries=2000, max_rows=10_000):
        from collections import OrderedDict

        self.store = store
        self.local = store.local
        self.executions = 0
        self.seconds = 0.0
        self.max_entries = max_entries
        self.max_rows = max_rows
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def query(self, sparql):
        runtime.check()
        with self._lock:
            cached = self._cache.get(sparql)
            if cached is not None:
                self._cache.move_to_end(sparql)
        if cached is None:
            started = time.perf_counter()
            try:
                cached = self.store.query(sparql)
            except QueryError as exc:
                cached = exc
            spent = time.perf_counter() - started
            runtime.add(graph_executions=1, graph_seconds=spent)
            with self._lock:
                self.executions += 1
                self.seconds += spent
                if _rows(cached) <= self.max_rows:
                    self._cache[sparql] = cached
                    if len(self._cache) > self.max_entries:
                        self._cache.popitem(last=False)
        if isinstance(cached, QueryError):
            raise cached
        return cached

    def __repr__(self):
        return f"CachedStore({self.store!r})"


def _rows(result):
    if isinstance(result, QueryError):
        return 0
    return len(result.get("results", {}).get("bindings", []))


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
