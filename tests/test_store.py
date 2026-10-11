import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from conftest import FIXTURES
from hermes import KnowledgeGraph
from hermes.store import CachedStore, EndpointStore, LocalStore, OxigraphStore, QueryError

QUERY = """PREFIX ex: <http://example.org/olympics#>
SELECT ?city WHERE { ?city ex:country ex:USA } ORDER BY ?city"""


def values(result):
    return [row["city"]["value"] for row in result["results"]["bindings"]]


def test_engines_agree():
    path = FIXTURES / "olympics.ttl"
    assert values(OxigraphStore(path).query(QUERY)) == values(LocalStore(path).query(QUERY))
    assert len(values(OxigraphStore(path).query(QUERY))) == 7


def test_ask_and_errors():
    store = OxigraphStore(FIXTURES / "olympics.ttl")
    ask = store.query("ASK { ?s ?p ?o }")
    assert ask["boolean"] is True
    with pytest.raises(QueryError):
        store.query("SELECT WHERE {")
    assert store.try_query("SELECT WHERE {")["results"]["bindings"] == []


def test_cache_counts_real_executions():
    store = CachedStore(OxigraphStore(FIXTURES / "olympics.ttl"))
    store.query(QUERY)
    store.query(QUERY)
    assert store.executions == 1


def test_endpoint_store_speaks_the_sparql_protocol():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"])).decode()
            seen["body"], seen["accept"] = body, self.headers["Accept"]
            data = json.dumps({"head": {}, "boolean": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/sparql-results+json")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    kg = KnowledgeGraph.from_endpoint(f"http://127.0.0.1:{server.server_port}/sparql")
    assert kg.query("ASK { ?s ?p ?o }")["boolean"] is True
    assert seen["body"].startswith("query=ASK")
    assert seen["accept"] == "application/sparql-results+json"
    server.server_close()


def test_cached_llm_records_usage_and_replays_for_free(tmp_path):
    from hermes.llm import LLM, CachedLLM

    class Counting(LLM):
        provider, model = "test", "counting"

        def complete(self, prompt, system=None):
            self.record(10, 3, 0.5)
            return "ok"

    cached = CachedLLM(Counting(), tmp_path)
    cached.complete("q")
    cached.complete("q")
    assert cached.usage == {"calls": 1, "input_tokens": 10, "output_tokens": 3, "seconds": 0.5}
    assert (cached.calls, cached.hits) == (2, 1)


def test_endpoint_store_refuses_huge_results():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"head": {"vars": ["x"]}, "results": {"bindings": [' + b" " * 5000 + b"]}}")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    store = EndpointStore(f"http://127.0.0.1:{server.server_port}/sparql", max_bytes=1000)
    with pytest.raises(QueryError, match="larger than"):
        store.query("SELECT * WHERE { ?s ?p ?o }")
    server.server_close()


def test_timeout_store_kills_slow_queries_and_recovers():
    import time

    from hermes.store import TimeoutStore

    store = TimeoutStore([FIXTURES / "olympics.ttl"], timeout=2)
    assert values(store.query(QUERY))[0].endswith("Atlanta")
    slow = """SELECT (COUNT(*) AS ?n) WHERE { ?a ?b ?c . ?d ?e ?f . ?g ?h ?i . ?j ?k ?l }"""
    started = time.time()
    with pytest.raises(QueryError, match="timed out"):
        store.query(slow)
    assert time.time() - started < 10
    assert len(values(store.query(QUERY))) == 7
    store.close()


def test_cache_is_bounded():
    store = CachedStore(OxigraphStore(FIXTURES / "olympics.ttl"), max_entries=2, max_rows=5)
    for query in [QUERY, QUERY + " LIMIT 1", QUERY + " LIMIT 2", QUERY + " LIMIT 3"]:
        store.query(query)
    assert len(store._cache) == 2
    assert QUERY not in store._cache  # 7 rows: over max_rows, never kept


def test_provider_presets(monkeypatch):
    from hermes import llm

    monkeypatch.setenv("OPENAI_API_KEY", "k1")
    monkeypatch.delenv("HERMES_LLM_BASE_URL", raising=False)
    client = llm.create(provider="openai", model="gpt-4o-mini")
    assert (client.base_url, client.api_key) == ("https://api.openai.com/v1", "k1")
    hf = llm.create(provider="huggingface", model="Qwen/Qwen2.5-7B-Instruct")
    assert hf.base_url == "https://router.huggingface.co/v1"
    server = llm.create(provider="vllm", model="m", base_url="https://x.modal.run/v1")
    assert server.base_url == "https://x.modal.run/v1"
    with pytest.raises(ValueError, match="base_url"):
        llm.create(provider="vllm", model="m")
    with pytest.raises(ValueError, match="Unknown"):
        llm.create(provider="nope")


def test_thinking_blocks_are_stripped():
    from hermes.llm import strip_thinking

    assert strip_thinking("<think>\nhmm\n</think>\n\nSELECT ?x") == "SELECT ?x"
    assert strip_thinking("SELECT ?x") == "SELECT ?x"
