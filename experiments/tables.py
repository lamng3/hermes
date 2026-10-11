"""Build the paper's result tables from the run outputs in results/.

    PYTHONPATH=src python experiments/tables.py [--results results] [--out paper]

Each table reads the latest finished run of each method and model, so
rerunning an experiment and this script updates the paper. The best value in
each model block is bold, and the Hermes rows are shaded. It writes
tab-main.tex, tab-verifier.tex, tab-features.tex, and tab-cost.tex, which
the paper includes with \\input.
"""

import argparse
import json
import statistics
from pathlib import Path

METHODS = [
    ("training-free", "OneShot"),
    ("uniqgen", "CGS"),
    ("algebra-agent", r"\hermes{}$^-$"),
    ("prover", r"\hermes"),
]
SHADED = {"algebra-agent", "prover"}
SIZES = [("qwen3-4b", "4B"), ("qwen3-8b", "8B"), ("qwen3-14b", "14B")]
TAGS = ["SELECT", "FILTER", "ORDER", "SUBSELECT", "GROUP", "COUNT", "LIMIT", "ASK"]


def latest(results, benchmark, config):
    finished = sorted(Path(results, benchmark, config).glob("*/metrics.json"))
    return finished[-1].parent if finished else None


def load(folder):
    predictions = [json.loads(line) for line in open(folder / "predictions.jsonl", encoding="utf-8")]
    return {
        "metrics": json.loads((folder / "metrics.json").read_text()),
        "run": json.loads((folder / "run.json").read_text()),
        "predictions": predictions,
    }


def summarize(data):
    official = data["metrics"]["official"]["average"]
    predictions = data["predictions"]
    seconds = sorted(p["seconds"] for p in predictions)
    total = {key: sum(p["cost"].get(key, 0) for p in predictions)
             for key in ("llm_calls", "input_tokens", "output_tokens", "graph_executions")}
    return {
        "P": official["set_P"], "R": official["set_recall"], "F1": official["set_F"],
        "Fndcg": official["set_F_ndcg"],
        "Strict": data["metrics"]["strict"]["average"]["set_F"],
        "Repro": data["metrics"]["reproducible"]["average"]["set_F"],
        "exact": sum(1 for p in predictions if (p.get("scores") or {}).get("set_F") == 1),
        "errors": sum(1 for p in predictions if p.get("error")),
        "rejections": sum(_extra(p).get("kernel_rejections", 0) for p in predictions),
        "steps": statistics.mean(_extra(p).get("steps", 0) for p in predictions),
        "wall": data["run"]["wall_seconds"] / 60,
        "p50": statistics.median(seconds),
        "p95": seconds[max(0, round(0.95 * len(seconds)) - 1)],
        **total,
    }


def _extra(prediction):
    import ast

    extra = prediction.get("extra") or {}
    if isinstance(extra, str):
        try:
            extra = ast.literal_eval(extra)
        except (ValueError, SyntaxError):
            extra = {key: int(value) for key, value in _ints(extra)}
    return extra if isinstance(extra, dict) else {}


def _ints(text):
    import re

    return re.findall(r"'(kernel_rejections|steps)': (\d+)", text)


def fmt(value, digits=3):
    return f"{value:.{digits}f}"


def row(cells, config, bold):
    shade = r"\cellcolor{hermesbg}" if config in SHADED else ""
    out = []
    for index, cell in enumerate(cells):
        text = rf"\textbf{{{cell}}}" if index in bold else cell
        out.append(shade + text if index else text)
    return " & ".join(out) + r" \\"


def best(values, high=True):
    """Indexes of the best value (ties included)."""
    target = max(values) if high else min(values)
    return {i for i, value in enumerate(values) if value == target}


def main_table(stats):
    columns = ["P", "R", "F1", "Fndcg", "Strict", "Repro"]
    lines = [
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"Model & Method & P & R & $F_1$ & $F_{\mathrm{ndcg}}$ & Strict & Repro. \\",
    ]
    for model, size in SIZES:
        lines.append(r"\midrule")
        block = [(config, label, stats[config, model]) for config, label in METHODS
                 if (config, model) in stats]
        bold = {}
        for offset, column in enumerate(columns):
            values = [round(s[column], 3) for _, _, s in block]
            for i in best(values):
                bold.setdefault(i, set()).add(offset + 2)
        for i, (config, label, s) in enumerate(block):
            first = rf"\multirow{{{len(block)}}}{{*}}{{Qwen3-{size}}}" if i == 0 else ""
            cells = [first, label] + [fmt(s[c]) for c in columns]
            lines.append(row(cells, config, bold.get(i, set())))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def verifier_table(stats):
    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{$F_1$} & \multicolumn{2}{c}{No answer} & \multicolumn{2}{c}{Steps} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        r"Model & \hermes{}$^-$ & \hermes & \hermes{}$^-$ & \hermes & \hermes{}$^-$ & \hermes \\",
        r"\midrule",
    ]
    for model, size in SIZES:
        minus, full = stats.get(("algebra-agent", model)), stats.get(("prover", model))
        if not (minus and full):
            continue
        f1 = [fmt(minus["F1"]), fmt(full["F1"])]
        errors = [str(minus["errors"]), str(full["errors"])]
        steps = [fmt(minus["steps"], 1), fmt(full["steps"], 1)]
        bold = lambda pair, high: [  # noqa: E731
            rf"\textbf{{{v}}}" if i in best([float(x) for x in pair], high) else v
            for i, v in enumerate(pair)]
        cells = [f"Qwen3-{size}", *bold(f1, True), *bold(errors, False), *steps]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def features_table(results, benchmark, model, questions):
    tags = {}
    for line in open(questions, encoding="utf-8"):
        item = json.loads(line)
        tags[item["id"]] = set(item["features"])
    scores = {}
    for config, _label in METHODS:
        folder = latest(results, benchmark, f"{config}@{model}")
        if folder:
            scores[config] = {p["id"]: (p.get("scores") or {}).get("set_F", 0)
                              for p in load(folder)["predictions"]}
    configs = [c for c, _ in METHODS if c in scores]
    header = " & ".join(label for c, label in METHODS if c in scores)
    lines = [
        r"\begin{tabular}{@{}lr" + "r" * len(configs) + "@{}}",
        r"\toprule",
        rf"Tag & $n$ & {header} \\",
        r"\midrule",
    ]
    for tag in TAGS:
        items = [i for i, t in tags.items() if tag in t]
        values = [round(sum(scores[c].get(i, 0) for i in items) / len(items), 3) for c in configs]
        top = best(values) if max(values) > 0 else set()
        cells = [rf"\kw{{{tag}}}", str(len(items))]
        for i, (config, value) in enumerate(zip(configs, values)):
            shade = r"\cellcolor{hermesbg}" if config in SHADED else ""
            text = r"\zero" if value == 0 else fmt(value)
            cells.append(shade + (rf"\textbf{{{text}}}" if i in top else text))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def cost_table(stats):
    lines = [
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r"Model & Method & Calls & In (k) & Out (k) & Exec. & Wall & p50 & p95 \\",
    ]
    for model, size in SIZES:
        lines.append(r"\midrule")
        block = [(config, label, stats[config, model]) for config, label in METHODS
                 if (config, model) in stats]
        for i, (config, label, s) in enumerate(block):
            first = rf"\multirow{{{len(block)}}}{{*}}{{{size}}}" if i == 0 else ""
            cells = [first, label, f"{s['llm_calls']:,}", f"{s['input_tokens'] / 1000:,.0f}",
                     f"{s['output_tokens'] / 1000:,.1f}", f"{s['graph_executions']:,}",
                     fmt(s["wall"], 1), fmt(s["p50"], 1), fmt(s["p95"], 1)]
            lines.append(row(cells, config, set()))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--benchmark", default="ck26")
    parser.add_argument("--out", default="paper")
    parser.add_argument("--features-model", default="qwen3-14b")
    args = parser.parse_args()

    stats = {}
    for config, _label in METHODS:
        for model, _size in SIZES:
            folder = latest(args.results, args.benchmark, f"{config}@{model}")
            if folder:
                stats[config, model] = summarize(load(folder))
                print(f"{config}@{model}: {folder.name}  F1 {stats[config, model]['F1']:.3f}")
    questions = Path("data/benchmarks", args.benchmark, "questions.jsonl")
    tables = {
        "tab-main.tex": main_table(stats),
        "tab-verifier.tex": verifier_table(stats),
        "tab-features.tex": features_table(args.results, args.benchmark, args.features_model, questions),
        "tab-cost.tex": cost_table(stats),
    }
    for name, body in tables.items():
        note = "% Generated by experiments/tables.py from results/; do not edit.\n"
        Path(args.out, name).write_text(note + body + "\n", encoding="utf-8")
        print(f"wrote {Path(args.out, name)}")


if __name__ == "__main__":
    main()
