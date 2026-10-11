"""Experiments: a config file in, predictions and metrics out.

A config (configs/<benchmark>/<system>.yaml) names a benchmark, a system and
its parameters, and the LLM. `Run` executes it:

    for each benchmark item:
        predict   the system writes SPARQL for the question
        execute   the query runs on the benchmark's graph
        score     the answers are compared with gold (hermes.evaluation)

and writes results/<benchmark>/<config name>/<run id>/ with the config,
predictions.jsonl (one row per item, with timing and errors), metrics.json
(official and strict averages) and run.json (git commit, model, counts). An
interrupted run resumes from its predictions.jsonl. Observers receive
start/item/end events, for progress output or extra logging.
"""

import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from hermes import benchmarks, llm as llms, systems
from hermes.context import Context, Example
from hermes.evaluation import aggregate, score
from hermes.store import QueryError


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config.setdefault("name", Path(path).stem)
    config.setdefault("params", {})
    config.setdefault("llm", {})
    return config


class Observer:
    def on_start(self, run, items):
        pass

    def on_item(self, run, record):
        pass

    def on_end(self, run, metrics):
        pass


class Progress(Observer):
    """One line per item on stderr."""

    def on_start(self, run, items):
        self.total = len(items)
        self.done = 0
        print(f"{run.config['name']}: {self.total} items -> {run.directory}", file=sys.stderr)

    def on_item(self, run, record):
        self.done += 1
        status = record["error"] or f"F1 {record['scores']['set_F']:.2f}"
        print(f"[{self.done}/{self.total}] {record['id']} {record['seconds']:.1f}s {status}",
              file=sys.stderr)

    def on_end(self, run, metrics):
        average = metrics["official"]["average"]
        print(f"set_F_ndcg {average.get('set_F_ndcg', 0):.3f}  F1 {average.get('set_F', 0):.3f}",
              file=sys.stderr)


class Run:
    def __init__(self, config, results="results", run_id=None, observers=(), llm=None, kg=None):
        self.config = config
        self.benchmark = benchmarks.load(config["benchmark"])
        self.kg = kg or benchmarks.graph(self.benchmark.graph)
        self.llm = llm or llms.create(**config["llm"])
        params = dict(config["params"])
        if _takes_llm(config["system"]):
            params.setdefault("llm", self.llm)
        self.system = systems.create(config["system"], **params)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.directory = Path(results) / self.benchmark.name / config["name"] / self.run_id
        self.observers = list(observers)
        self.examples = _examples(config.get("examples"))
        self.contamination = contamination(self.examples, self.items())
        if self.contamination:
            print(f"warning: {self.contamination} few-shot example queries also answer "
                  f"test items of {self.benchmark.name}.", file=sys.stderr)

    def items(self):
        return self.benchmark.select(
            languages=self.config.get("languages"),
            limit=self.config.get("limit"),
            ids=self.config.get("ids"),
        )

    def execute(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "config.yaml").write_text(
            yaml.safe_dump(self.config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        items = self.items()
        records = self._completed()
        for observer in self.observers:
            observer.on_start(self, items)
        started = time.time()
        with (self.directory / "predictions.jsonl").open("a", encoding="utf-8") as handle:
            for item in items:
                if item.id in records:
                    continue
                record = self.evaluate(item)
                records[item.id] = record
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                for observer in self.observers:
                    observer.on_item(self, record)
        chosen = {item.id: records[item.id] for item in items}
        metrics = self.metrics(chosen)
        _write_json(self.directory / "metrics.json", metrics)
        _write_json(self.directory / "run.json", self.provenance(chosen, time.time() - started))
        for observer in self.observers:
            observer.on_end(self, metrics)
        return metrics

    def evaluate(self, item):
        """Template: predict, execute, score."""
        context = Context(examples=self.examples)
        if self.config.get("leak_gold"):
            context.answers = [v for v, g in self.benchmark.gold.get(item.id, {}).items() if g]
        started = time.time()
        output, error = {}, ""
        try:
            output = self.system(item.question, self.kg, context)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        seconds = time.time() - started
        sparql = output.get("sparql", "") if isinstance(output, dict) else ""
        result = {}
        if sparql:
            try:
                result = self.kg.query(sparql)
            except QueryError as exc:
                error = error or f"QueryError: {exc}"
        gold = self.benchmark.gold.get(item.id, {})
        order = item.id in self.benchmark.order_required
        extra = {k: v for k, v in output.items() if k not in ("sparql", "model", "provider")} \
            if isinstance(output, dict) else {}
        return {
            "id": item.id,
            "question": item.question,
            "sparql": sparql,
            "error": error,
            "gold_reproduces": self.gold_reproduces(item, gold, order),
            "seconds": round(seconds, 3),
            "scores": score(result, gold, order),
            "strict": score(result, gold, order, strict=True),
            "answers": len(result.get("results", {}).get("bindings", [])) if result else 0,
            "extra": extra,
        }

    def gold_reproduces(self, item, gold, order):
        """Does the gold query, run on this graph, return the gold answers?"""
        if not item.sparql:
            return False
        try:
            result = self.kg.query(item.sparql)
        except QueryError:
            return False
        return reproduces(score(result, gold, order, strict=True))

    def metrics(self, records):
        languages = self.config.get("languages") or self.benchmark.languages
        reproducible = {k: r for k, r in records.items() if r.get("gold_reproduces")}
        return {
            "official": aggregate({k: r["scores"] for k, r in records.items()}, languages),
            "strict": aggregate(
                {k: r["strict"] for k, r in records.items()}, languages, strict=True
            ),
            "reproducible": aggregate(
                {k: r["scores"] for k, r in reproducible.items()}, languages
            ),
            "items": len(records),
            "reproducible_items": len(reproducible),
            "errors": sum(1 for r in records.values() if r["error"]),
            "seconds": _percentiles([r["seconds"] for r in records.values()]),
        }

    def provenance(self, records, wall):
        llm = self.llm
        return {
            "config": self.config["name"],
            "benchmark": self.benchmark.name,
            "system": self.config["system"],
            "model": getattr(llm, "model", ""),
            "provider": getattr(llm, "provider", ""),
            "llm_calls": getattr(llm, "calls", None),
            "llm_cache_hits": getattr(llm, "hits", None),
            "graph_executions": self.kg.executions,
            "contaminated_examples": self.contamination,
            "items": len(records),
            "wall_seconds": round(wall, 1),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "finished": datetime.now().isoformat(timespec="seconds"),
        }

    def _completed(self):
        path = self.directory / "predictions.jsonl"
        if not path.exists():
            return {}
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        return {row["id"]: row for row in rows}


def _takes_llm(name):
    import inspect

    factory = systems._lookup(name)
    if not isinstance(factory, type):
        return False
    return "llm" in inspect.signature(factory).parameters


def _examples(spec):
    """Few-shot examples from another benchmark: {benchmark: ck25, limit: 10}."""
    if not spec:
        return []
    pool = benchmarks.load(spec["benchmark"]).select(languages=spec.get("languages", ["en"]))
    return [Example(nl=i.question, sparql=i.sparql) for i in pool[: spec.get("limit", 10)]]


def contamination(examples, items):
    """How many example queries equal a test item's gold query (whitespace-blind)."""
    gold = {" ".join(item.sparql.split()) for item in items if item.sparql}
    return sum(1 for example in examples if " ".join(example.sparql.split()) in gold)


def _percentiles(values):
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction):
        return round(ordered[min(len(ordered) - 1, int(fraction * len(ordered)))], 3)

    return {"p50": at(0.5), "p95": at(0.95), "mean": round(sum(values) / len(values), 3)}


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def verify(benchmark_name):
    """Run every gold query on the local graph and compare with official gold.

    Disagreements come from differences between the graph store used here and
    the one the gold sets were computed on (data version, number formatting).
    """
    bench = benchmarks.load(benchmark_name)
    kg = benchmarks.graph(bench.graph)
    report = {}
    for qid in sorted({item.qid for item in bench.items}, key=_natural):
        item = next(i for i in bench.items if i.qid == qid)
        try:
            result = kg.query(item.sparql)
        except QueryError as exc:
            report[item.id] = {"set_F": 0.0, "error": str(exc)[:200]}
            continue
        report[item.id] = score(result, bench.gold.get(item.id, {}), strict=True)
    agree = sum(1 for value in report.values() if reproduces(value))
    return {"benchmark": bench.name, "questions": len(report), "agree": agree, "items": report}


def reproduces(scores):
    return scores.get("set_F", 0.0) > 1 - 1e-9


def _natural(text):
    return (len(text), text)


def _rank(average):
    return average.get("set_F_ndcg", average.get("set_F", 0.0))


def leaderboard(results="results", benchmark=None):
    """Markdown table of the latest run per config, with official peer rows."""
    root = Path(results)
    lines = []
    for bench_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "peers"):
        if benchmark and bench_dir.name != benchmark:
            continue
        rows = []
        for config_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
            runs = sorted(p for p in config_dir.iterdir() if (p / "metrics.json").exists())
            if not runs:
                continue
            metrics = json.loads((runs[-1] / "metrics.json").read_text())
            run = json.loads((runs[-1] / "run.json").read_text())
            rows.append((config_dir.name, run.get("model", ""), metrics["items"],
                         metrics["official"]["average"], metrics["strict"]["average"]))
        peers_path = root / "peers" / f"{bench_dir.name}.json"
        peers = json.loads(peers_path.read_text())["teams"] if peers_path.exists() else {}
        lines.append(f"## {bench_dir.name}\n")
        lines.append("| system | model | items | set_F_ndcg | F1 | P | R | strict F1 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        table = [(name, model, n, off, strict) for name, model, n, off, strict in rows]
        table += [(f"{team} (official)", "", "", avg, None) for team, avg in peers.items()]
        table.sort(key=lambda row: -_rank(row[3]))
        for name, model, n, off, strict in table:
            strict_f = f"{strict['set_F']:.3f}" if strict else ""
            lines.append(
                f"| {name} | {model} | {n} | {_rank(off):.3f} | {off['set_F']:.3f} "
                f"| {off['set_P']:.3f} | {off['set_recall']:.3f} | {strict_f} |"
            )
        lines.append("")
    return "\n".join(lines)
