"""Serve a Hermes system behind the Text2SPARQL challenge API.

GET /?question=...&dataset=<dataset id> returns
{"dataset": ..., "question": ..., "query": ...}; an unknown dataset is 404.
This is the contract the official text2sparql-client calls, so any registered
system can be submitted to the challenge or scored with its tools.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from hermes import benchmarks
from hermes.benchmarks.sources import SOURCES


def dataset_graphs():
    """Text2SPARQL dataset id -> graph name, from the known sources."""
    return {s["dataset_id"]: s["graph"] for s in SOURCES.values() if "dataset_id" in s}


def make_handler(system, graphs):
    loaded = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path != "/":
                return self._send(404, {"detail": "Not found"})
            params = {k: v[0] for k, v in parse_qs(url.query).items()}
            question, dataset = params.get("question", ""), params.get("dataset", "")
            if dataset not in graphs:
                return self._send(404, {"detail": f"Unknown dataset {dataset}"})
            if not question:
                return self._send(422, {"detail": "question is required"})
            if dataset not in loaded:
                loaded[dataset] = benchmarks.graph(graphs[dataset])
            try:
                query = system(question, loaded[dataset], None)["sparql"]
            except Exception as exc:
                return self._send(500, {"detail": f"{type(exc).__name__}: {exc}"})
            self._send(200, {"dataset": dataset, "question": question, "query": query})

        def _send(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(system, host="127.0.0.1", port=8000, graphs=None):
    server = ThreadingHTTPServer((host, port), make_handler(system, graphs or dataset_graphs()))
    print(f"Text2SPARQL API on http://{host}:{port}/")
    server.serve_forever()
