"""Fit the budgeted planner's selector from the candidates its runs logged.

Every executed candidate of a `budgeted` run is run again on the benchmark's
graph and scored against gold; a candidate is correct when its set F1 is at
least --threshold. The selector's weights are fitted to those labels and
written as JSON, which a config passes back as `params: {weights: <path>}`.

Fit on one benchmark and evaluate on another: CK25 and CK26 share most
questions, so fitting on CK25 and testing on CK26 leaks (see data/README.md).

    PYTHONPATH=src python experiments/fit_selector.py \\
        results/db25/budgeted/<run> --out configs/weights/db25.json
"""

import argparse
import json
from pathlib import Path

import yaml

from hermes import benchmarks
from hermes.evaluation import score
from hermes.selection import Scorer
from hermes.store import QueryError


def examples(run, threshold):
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    benchmark = benchmarks.load(config["benchmark"])
    kg = benchmarks.graph(benchmark.graph, query_timeout=30)
    found = []
    for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        gold = benchmark.gold.get(record["id"], {})
        order = record["id"] in benchmark.order_required
        for candidate in (record.get("extra") or {}).get("candidates", []):
            if candidate.get("rows") is None:
                continue
            try:
                result = kg.query(candidate["sparql"])
            except QueryError:
                result = {}
            f1 = score(result, gold, order, strict=True)["set_F"]
            found.append((candidate["features"], int(f1 >= threshold)))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run directories of budgeted runs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    labelled = [pair for run in args.runs for pair in examples(run, args.threshold)]
    if not labelled:
        raise SystemExit("No executed candidates in these runs.")
    scorer = Scorer().fit(labelled)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scorer.to_dict(), indent=2) + "\n", encoding="utf-8")
    positives = sum(label for _features, label in labelled)
    print(f"{len(labelled)} candidates, {positives} correct; weights -> {args.out}")


if __name__ == "__main__":
    main()
