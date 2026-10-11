import json
import shutil
import threading
import urllib.error
import urllib.parse
import urllib.request

import pytest
import yaml

from conftest import FIXTURES, ScriptedLLM
from hermes import benchmarks
from hermes.benchmarks import adapters
from hermes.experiments import Run, contamination, leaderboard
from hermes.context import Example

T2S_YAML = """
dataset:
  id: https://example.org/t2s/olympics/
  prefix: oly
questions:
- id: 1
  question:
    en: Which US cities hosted the Winter Olympics?
    de: Welche US-Staedte waren Gastgeber der Winterspiele?
  features: [SELECT]
  query:
    sparql: |
      PREFIX ex: <http://example.org/olympics#>
      SELECT DISTINCT ?city WHERE { ?city ex:hosted ?g ; ex:country ex:USA . ?g ex:season "Winter" }
- id: 2
  question:
    en: Did Oslo host the Olympics?
  features: [ASK]
  query:
    sparql: |
      PREFIX ex: <http://example.org/olympics#>
      ASK { ex:Oslo ex:hosted ?g }
"""
OLY = "http://example.org/olympics#"
GOLD = {
    "oly:1-en": {OLY + "LakePlacid": 1, OLY + "SquawValley": 1, OLY + "SaltLakeCity": 1},
    "oly:1-de": {OLY + "LakePlacid": 1, OLY + "SquawValley": 1, OLY + "SaltLakeCity": 1},
    "oly:2-en": {"true": 1},
}


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A data/ folder with the Olympics graph and one normalized benchmark."""
    monkeypatch.setenv("HERMES_DATA", str(tmp_path))
    shutil.copy(FIXTURES / "olympics.ttl", tmp_path / "olympics.ttl")
    (tmp_path / "graphs.yaml").write_text(yaml.safe_dump({"olympics": {"files": ["olympics.ttl"]}}))
    items, gold = adapters.text2sparql(T2S_YAML, json.dumps(GOLD))
    manifest = {"name": "oly", "graph": "olympics", "languages": ["en", "de"],
                "dataset_id": "https://example.org/t2s/olympics/"}
    benchmarks.save("oly", manifest, items, gold)
    return tmp_path


def test_text2sparql_adapter_uses_official_ids():
    items, gold = adapters.text2sparql(T2S_YAML, json.dumps(GOLD), languages=["en"])
    assert [item.id for item in items] == ["oly:1-en", "oly:2-en"]
    assert set(gold) == {"oly:1-en", "oly:2-en"}
    assert items[1].features == ["ASK"]


def test_qald_adapter_takes_gold_from_the_answers():
    qald = {"questions": [{
        "id": 7,
        "question": [{"language": "en", "string": "Who?"}, {"language": "en", "string": "dup"},
                     {"language": "de", "string": "Wer?"}],
        "query": {"sparql": "SELECT ?x WHERE { ?x ?p ?o }"},
        "answers": [{"head": {"vars": ["x"]},
                     "results": {"bindings": [{"x": {"type": "uri", "value": "urn:a"}}]}}],
    }]}
    items, gold = adapters.qald(json.dumps(qald), "q")
    assert [(i.id, i.question) for i in items] == [("q:7-en", "Who?"), ("q:7-de", "Wer?")]
    assert gold["q:7-en"] == {"urn:a": 1}


def test_save_and_load_roundtrip(data):
    bench = benchmarks.load("oly")
    assert bench.languages == ["en", "de"]
    assert len(bench.select(languages=["en"])) == 2
    assert benchmarks.available() == ["oly"]
    kg = benchmarks.graph(bench.graph)
    assert kg.query("ASK { ?s ?p ?o }")["boolean"] is True


def gold_system(question, kg, context):
    """Answers every question with the right query, for pipeline checks."""
    if "Oslo" in question:
        return {"sparql": "PREFIX ex: <http://example.org/olympics#>\nASK { ex:Oslo ex:hosted ?g }"}
    return {"sparql": "PREFIX ex: <http://example.org/olympics#>\nSELECT DISTINCT ?city WHERE "
                      "{ ?city ex:hosted ?g ; ex:country ex:USA . ?g ex:season \"Winter\" }"}


def test_run_writes_predictions_metrics_and_provenance(data, tmp_path):
    from hermes import systems

    if "gold-test" not in systems.available():
        systems.register("gold-test")(gold_system)
    config = {"name": "gold", "benchmark": "oly", "system": "gold-test", "params": {}, "llm": {}}
    results = tmp_path / "results"
    run = Run(config, results=results, run_id="r1", llm=ScriptedLLM([]))
    metrics = run.execute()
    assert metrics["official"]["average"]["set_F"] == 1.0
    assert metrics["reproducible_items"] == 3
    assert metrics["official"]["average-de"]["set_F"] == 1.0
    folder = results / "oly" / "gold" / "r1"
    rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    assert [row["id"] for row in rows] == ["oly:1-en", "oly:1-de", "oly:2-en"]
    assert json.loads((folder / "run.json").read_text())["system"] == "gold-test"
    assert "| gold |" in leaderboard(results)


def test_run_resumes_without_redoing_items(data, tmp_path):
    from hermes import systems

    calls = []

    def counting(question, kg, context):
        calls.append(question)
        return gold_system(question, kg, context)

    if "counting-test" not in systems.available():
        systems.register("counting-test")(counting)
    config = {"name": "c", "benchmark": "oly", "system": "counting-test", "params": {}, "llm": {},
              "limit": 1}
    Run(config, results=tmp_path / "r", run_id="x", llm=ScriptedLLM([])).execute()
    config["limit"] = None
    Run(config, results=tmp_path / "r", run_id="x", llm=ScriptedLLM([])).execute()
    assert len(calls) == 3


def test_errors_are_recorded_not_raised(data, tmp_path):
    from hermes import systems

    def broken(question, kg, context):
        raise RuntimeError("model down")

    if "broken-test" not in systems.available():
        systems.register("broken-test")(broken)
    config = {"name": "b", "benchmark": "oly", "system": "broken-test", "params": {}, "llm": {}}
    metrics = Run(config, results=tmp_path / "r", llm=ScriptedLLM([])).execute()
    assert metrics["errors"] == 3
    assert metrics["official"]["average"]["set_F"] == 0.0


def test_contamination_counts_examples_that_are_test_queries(data):
    items = benchmarks.load("oly").items
    examples = [Example(nl="x", sparql=items[0].sparql.replace("\n", "  \n")),
                Example(nl="y", sparql="SELECT * WHERE { ?s ?p ?o }")]
    assert contamination(examples, items) == 1


def test_text2sparql_api(data):
    from http.server import ThreadingHTTPServer

    from hermes.serve import make_handler

    dataset = "https://example.org/t2s/olympics/"
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(gold_system, {dataset: "olympics"}))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/"
    query = urllib.parse.urlencode({"question": "Did Oslo host?", "dataset": dataset})
    body = json.loads(urllib.request.urlopen(base + "?" + query).read())
    assert body["dataset"] == dataset and body["query"].endswith("ASK { ex:Oslo ex:hosted ?g }")
    unknown = urllib.parse.urlencode({"question": "x", "dataset": "nope"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(base + "?" + unknown)
    assert error.value.code == 404
    server.shutdown()


def test_runs_record_cost_per_item_and_per_run(data, tmp_path):
    from hermes import systems

    if "gold-cost-test" not in systems.available():
        systems.register("gold-cost-test")(gold_system)
    config = {"name": "cost", "benchmark": "oly", "system": "gold-cost-test", "params": {},
              "llm": {}}
    Run(config, results=tmp_path / "r", run_id="c", llm=ScriptedLLM([])).execute()
    folder = tmp_path / "r" / "oly" / "cost" / "c"
    rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    assert set(rows[0]["cost"]) == {"llm_calls", "input_tokens", "output_tokens", "llm_seconds",
                                    "graph_executions", "graph_seconds"}
    run = json.loads((folder / "run.json").read_text())
    assert run["graph_executions"] > 0 and "input_tokens" in run and "graph_seconds" in run


def test_items_past_their_time_limit_are_stopped(data, tmp_path):
    import time

    from hermes import systems

    def sleepy(question, kg, context):
        time.sleep(5)
        return gold_system(question, kg, context)

    if "sleepy-test" not in systems.available():
        systems.register("sleepy-test")(sleepy)
    config = {"name": "slow", "benchmark": "oly", "system": "sleepy-test", "params": {},
              "llm": {}, "limit": 1, "limits": {"item_seconds": 0.5}}
    started = time.time()
    metrics = Run(config, results=tmp_path / "r", llm=ScriptedLLM([])).execute()
    assert time.time() - started < 4
    assert metrics["errors"] == 1


def test_sweeps_expand_benchmarks_systems_and_models(tmp_path, monkeypatch):
    from hermes.experiments import expand_sweep, latest_unfinished

    monkeypatch.setenv("SERVER", "https://gpu.example/v1")
    sweep = tmp_path / "s.yaml"
    sweep.write_text(yaml.safe_dump({
        "benchmarks": ["a", "b"], "cache": "c", "limits": {"item_seconds": 9},
        "systems": [{"system": "prover"}, {"name": "short", "system": "prover",
                                           "params": {"max_steps": 3}}],
        "models": [{"name": "small", "provider": "ollama", "model": "m"},
                   {"name": "gpu", "provider": "vllm", "model": "q", "base_url": "${SERVER}"}],
    }))
    configs = expand_sweep(sweep)
    assert len(configs) == 8
    gpu = next(c for c in configs if c["name"] == "short@gpu" and c["benchmark"] == "b")
    assert gpu["llm"] == {"provider": "vllm", "model": "q", "base_url": "https://gpu.example/v1",
                          "cache": "c"}
    assert gpu["params"] == {"max_steps": 3} and gpu["limits"] == {"item_seconds": 9}
    run = tmp_path / "r" / "b" / "short@gpu" / "20260101-000000"
    run.mkdir(parents=True)
    assert latest_unfinished(tmp_path / "r", gpu) == "20260101-000000"
    (run / "metrics.json").write_text("{}")
    assert latest_unfinished(tmp_path / "r", gpu) is None
