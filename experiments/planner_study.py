"""How much does the optimizer save, and are its plans right? Run it on gold queries.

For every distinct gold query of a benchmark, the query is translated as
written and optimized, both plans run with the plan executor (scans as
single-pattern queries, bind joins), and their answers are checked against
the engine running the query text. When they differ, two checks say why:
under ORDER BY with LIMIT, rows that tie on the sort key may be cut either
way, so both are rerun without LIMIT/OFFSET ("same up to ties"); and on a
local graph, rdflib's own engine referees ("engine disagrees" when rdflib
agrees with the plan: Oxigraph 0.5 reads a / b * c as a / (b * c)). Recorded per query: answers equal, store
queries, rows fetched, intermediate rows, seconds, estimated C_out before and
after, and the q-error of the root's row estimate.

    PYTHONPATH=src python experiments/planner_study.py ck26 [db26 ...]

Writes results/planner/<benchmark>/<run id>/{queries.jsonl,summary.json}.
"""

import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from hermes import benchmarks, planner
from hermes.pricing.statistics import qerror


def study(name, results="results"):
    benchmark = benchmarks.load(name)
    kg = benchmarks.graph(benchmark.graph, query_timeout=60)
    referee = benchmarks.graph(benchmark.graph, engine="rdflib") if kg.store.local else None
    seen, records = set(), []
    for item in benchmark.select():
        if not item.sparql or item.sparql in seen:
            continue
        seen.add(item.sparql)
        records.append(measure(item, kg, referee))
        r = records[-1]
        print(f"{item.id:14} {r['status']:12} fetched {r.get('fetched_written', '-'):>8} → "
              f"{r.get('fetched_optimized', '-'):>8}", file=sys.stderr)
    run = Path(results) / "planner" / name / datetime.now().strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    with (run / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(records)
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return run


def measure(item, kg, referee=None):
    record = {"id": item.id, "sparql": item.sparql}
    try:
        optimized = planner.plan(item.sparql, kg)
    except planner.UnsupportedQuery as exc:
        return {**record, "status": "unsupported", "error": str(exc)}
    except Exception as exc:  # a parse error in a gold query
        return {**record, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    record["rewrites"] = [f"{r.rule}: {r.text}" for r in optimized.rewrites]
    record["cost_written"] = round(optimized.cost_before, 1)
    record["cost_optimized"] = round(optimized.cost_after, 1)
    try:
        started = time.perf_counter()
        engine = planner.EngineExecutor(kg)(optimized.written)
        record["seconds_engine"] = round(time.perf_counter() - started, 4)
    except Exception as exc:
        return {**record, "status": "engine error", "error": f"{type(exc).__name__}: {exc}"}
    run = planner.PlanExecutor(kg)
    for label, query in (("written", optimized.written), ("optimized", optimized.query)):
        try:
            result = run(query)
        except Exception as exc:
            record[f"error_{label}"] = f"{type(exc).__name__}: {exc}"
            continue
        trace = result.trace
        record[f"same_{label}"] = planner.same_answers(result, engine)
        record[f"queries_{label}"] = trace.queries
        record[f"fetched_{label}"] = trace.fetched
        record[f"intermediate_{label}"] = trace.intermediate
        record[f"seconds_{label}"] = round(trace.seconds, 4)
        if label == "optimized" and id(query.root) in trace.rows:
            estimated = optimized.estimator(query.root).rows
            record["rows_estimated"] = round(estimated, 2)
            record["rows_actual"] = trace.rows[id(query.root)]
            record["qerror"] = round(qerror(estimated, trace.rows[id(query.root)]), 2)
    both = "same_written" in record and "same_optimized" in record
    if not both:
        record["status"] = "plan error"
    elif record["same_written"] and record["same_optimized"]:
        record["status"] = "ok"
    else:
        record["status"] = explain(optimized, kg, referee)
    return record


def explain(optimized, kg, referee):
    """Why the plan's answers differ from the engine's, when that can be told."""
    unsliced = _without_slice(optimized.written.text)
    if unsliced != optimized.written.text:
        full = planner.plan(unsliced, kg)
        engine = planner.EngineExecutor(kg)(full.written)
        run = planner.PlanExecutor(kg)
        if all(planner.same_answers(run(q), engine) for q in (full.written, full.query)):
            return "same up to ties"
    if referee is not None:
        judge = planner.EngineExecutor(referee)(optimized.written)
        run = planner.PlanExecutor(kg)
        if all(planner.same_answers(run(q), judge) for q in (optimized.written, optimized.query)):
            return "engine disagrees"
    return "differs"


def _without_slice(sparql):
    return re.sub(r"\b(LIMIT|OFFSET)\s+\d+", "", sparql, flags=re.IGNORECASE)


def summarize(records):
    # Queries whose plans ran: answers equal, or different for a reason that is not the plan's.
    ok = [r for r in records if r["status"] in ("ok", "same up to ties", "engine disagrees")]

    def total(key):
        return sum(r[key] for r in ok)

    def ratio(key):
        values = [r[f"{key}_optimized"] / r[f"{key}_written"] for r in ok if r[f"{key}_written"]]
        return round(statistics.median(values), 3) if values else None

    qerrors = sorted(r["qerror"] for r in ok if "qerror" in r)
    return {
        "queries": len(records),
        "status": {s: sum(1 for r in records if r["status"] == s)
                   for s in sorted({r["status"] for r in records})},
        "rewritten": sum(1 for r in ok if r["rewrites"]),
        "fetched": {"written": total("fetched_written"), "optimized": total("fetched_optimized"),
                    "median ratio": ratio("fetched")},
        "intermediate": {"written": total("intermediate_written"),
                         "optimized": total("intermediate_optimized"),
                         "median ratio": ratio("intermediate")},
        "store queries": {"written": total("queries_written"),
                          "optimized": total("queries_optimized")},
        "seconds": {"written": round(total("seconds_written"), 3),
                    "optimized": round(total("seconds_optimized"), 3),
                    "engine": round(sum(r.get("seconds_engine", 0) for r in ok), 3)},
        "worse": sum(1 for r in ok if r["fetched_optimized"] > r["fetched_written"]),
        "qerror": {"median": statistics.median(qerrors) if qerrors else None,
                   "p90": qerrors[int(0.9 * (len(qerrors) - 1))] if qerrors else None},
    }


if __name__ == "__main__":
    for benchmark in sys.argv[1:] or ["ck26"]:
        study(benchmark)
