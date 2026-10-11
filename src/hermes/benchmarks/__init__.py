"""Benchmarks in one normalized format, and the graphs they run on.

Every benchmark, whatever its source format, is stored as

    data/benchmarks/<name>/benchmark.yaml   manifest: graph, languages, source, license
    data/benchmarks/<name>/questions.jsonl  one row per (question, language)
    data/benchmarks/<name>/gold.json        gold answer sets, Text2SPARQL layout

A question row has `id` ("ck26:7-en"), `qid`, `lang`, `question`, `sparql`,
`features` and `order_matters`. gold.json maps each id to {value: 1} and
lists ids whose order matters under "order_required", the layout the official
Text2SPARQL scorer reads. Adapters in this package convert source formats
(Text2SPARQL YAML, QALD JSON) into this one. Graphs are declared in
data/graphs.yaml.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from hermes.knowledge import KnowledgeGraph


@dataclass
class Item:
    id: str
    qid: str
    lang: str
    question: str
    sparql: str = ""
    features: list = field(default_factory=list)
    order_matters: bool = False


@dataclass
class Benchmark:
    name: str
    manifest: dict
    items: list
    gold: dict
    root: Path

    @property
    def languages(self):
        return self.manifest.get("languages", ["en"])

    @property
    def graph(self):
        return self.manifest["graph"]

    @property
    def dataset_id(self):
        return self.manifest.get("dataset_id", "")

    @property
    def order_required(self):
        return set(self.gold.get("order_required", []))

    def select(self, languages=None, limit=None, ids=None):
        items = [
            item for item in self.items
            if (not languages or item.lang in languages) and (not ids or item.id in ids)
        ]
        return items[:limit] if limit else items


def data_dir():
    """HERMES_DATA, else the nearest data/ folder above the working directory."""
    configured = os.environ.get("HERMES_DATA")
    if configured:
        return Path(configured)
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / "data" / "graphs.yaml").exists():
            return folder / "data"
    return here / "data"


def available():
    root = data_dir() / "benchmarks"
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "benchmark.yaml").exists())


def load(name):
    root = data_dir() / "benchmarks" / name
    if not (root / "benchmark.yaml").exists():
        known = ", ".join(available()) or "(none)"
        raise KeyError(f"Unknown benchmark {name!r}. Available: {known}.")
    manifest = yaml.safe_load((root / "benchmark.yaml").read_text(encoding="utf-8"))
    items = [
        Item(**json.loads(line))
        for line in (root / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold_path = root / "gold.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8")) if gold_path.exists() else {}
    return Benchmark(name, manifest, items, gold, root)


def save(name, manifest, items, gold):
    root = data_dir() / "benchmarks" / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmark.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    with (root / "questions.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    gold_text = json.dumps(gold, indent=1, ensure_ascii=False)
    (root / "gold.json").write_text(gold_text, encoding="utf-8")
    return root


def graphs():
    path = data_dir() / "graphs.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def graph(name, query_timeout=None):
    """Factory: a KnowledgeGraph from its entry in data/graphs.yaml.

    query_timeout (seconds) bounds every instance query; it overrides the
    graph's own `timeout` entry.
    """
    spec = graphs().get(name)
    if spec is None:
        raise KeyError(f"Unknown graph {name!r}. Declare it in {data_dir() / 'graphs.yaml'}.")
    root = data_dir()
    schema = [root / path for path in spec.get("schema", [])]
    limit = spec.get("vocabulary_limit", 60)
    if "files" in spec:
        files = [root / path for path in spec["files"]]
        missing = [str(path) for path in files if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Graph {name!r} is not downloaded: {', '.join(missing)}. "
                f"Run `hermes data fetch` for a benchmark that uses it."
            )
        kg = KnowledgeGraph.from_files(
            *files, schema=schema, name=name, engine=spec.get("engine", "oxigraph"),
            timeout=query_timeout or spec.get("timeout"),
        )
    else:
        kg = KnowledgeGraph.from_endpoint(
            spec["endpoint"],
            schema=schema,
            prefixes=list((spec.get("prefixes") or {}).items()),
            name=name,
            timeout=query_timeout or spec.get("timeout", 60.0),
        )
    kg.vocabulary_limit = limit
    return kg
