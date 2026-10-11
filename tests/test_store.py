import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from conftest import FIXTURES
from hermes import KnowledgeGraph
from hermes.store import CachedStore, LocalStore, OxigraphStore, QueryError

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
