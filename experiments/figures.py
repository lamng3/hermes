"""Build the paper's two figures from the run outputs in results/.

    PYTHONPATH=src python experiments/figures.py [--results results] [--out paper/fig]

Each figure reads the latest finished run of each configuration, so rerunning
an experiment and this script updates the paper. Colors follow the method:
Hermes blue, constraint-guided search (CGS) orange, one-shot aqua; ablated
variants keep their family color with a hatch. The palette was checked with a
colorblind-safety validator (all pairs pass; aqua needs visible labels, so
every bar is labeled).
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a199"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OUTCOME = ["#e4e3df", "#bdbbb4", "#8a8780", "#3d3b37"]  # error < zero < partial < exact

# config directory -> (label, color, hatched)
METHODS = {
    "prover": ("Hermes", BLUE, False),
    "algebra-agent": ("Hermes$^-$", BLUE, True),
    "uniqgen": ("CGS", ORANGE, False),
    "uniqgen-gold-oracle": ("CGS, gold oracle", ORANGE, True),
    "training-free": ("OneShot", AQUA, False),
}
COLUMN = 3.33  # inches, one PVLDB column


def style():
    # Linux Biolinum is the paper's sans font; tectonic caches it.
    cache = Path.home() / "Library/Caches/TectonicProject.Tectonic/bundles/data"
    fonts = glob.glob(str(cache / "*/LinBiolinum_R.otf")) + glob.glob(str(cache / "*/LinBiolinum_RB.otf"))
    family = "DejaVu Sans"
    for font in fonts:
        font_manager.fontManager.addfont(font)
    if fonts:
        family = font_manager.FontProperties(fname=fonts[0]).get_name()
    plt.rcParams.update({
        "font.family": family, "font.size": 7, "axes.titlesize": 7,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "hatch.linewidth": 0.6, "hatch.color": "white", "pdf.fonttype": 3,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def latest_runs(results, benchmark):
    runs = {}
    for config in METHODS:
        finished = sorted(Path(results, benchmark, config).glob("*/metrics.json"))
        if finished:
            folder = finished[-1].parent
            runs[config] = {
                "metrics": json.loads((folder / "metrics.json").read_text()),
                "run": json.loads((folder / "run.json").read_text()),
                "rows": [
                    json.loads(line)
                    for line in (folder / "predictions.jsonl").read_text().splitlines()
                ],
            }
    return runs


def clean(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def bar(ax, y, width, config, height):
    label, color, hatched = METHODS[config]
    ax.barh(y, width, height=height, color=color, edgecolor="white", linewidth=1.0,
            hatch="////" if hatched else None)


def outcomes(runs, out):
    """Share of items by outcome, per method."""
    order = [c for c in ("training-free", "uniqgen", "algebra-agent", "prover") if c in runs]
    fig, ax = plt.subplots(figsize=(COLUMN, 0.32 * len(order) + 0.55))
    names = ["failed", "ran, $F_1=0$", "partial", "exact"]
    for i, config in enumerate(order):
        rows = runs[config]["rows"]
        counts = [
            sum(1 for r in rows if r["error"]),
            sum(1 for r in rows if not r["error"] and r["scores"]["set_F"] == 0),
            sum(1 for r in rows if not r["error"] and 0 < r["scores"]["set_F"] < 1 - 1e-9),
            sum(1 for r in rows if not r["error"] and r["scores"]["set_F"] >= 1 - 1e-9),
        ]
        left = 0
        for k, count in enumerate(counts):
            ax.barh(i, count, left=left, height=0.62, color=OUTCOME[k], edgecolor="white",
                    linewidth=1.0, label=names[k] if i == 0 else None)
            if count >= 3:
                ax.text(left + count / 2, i, str(count), ha="center", va="center",
                        color="white" if k >= 2 else INK, fontsize=6)
            left += count
    ax.set_yticks(range(len(order)), [METHODS[c][0] for c in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("CK26 items (of 100)")
    clean(ax)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.45, 1.0), frameon=False,
              handlelength=1.0, columnspacing=1.0)
    fig.savefig(out / "outcomes.pdf")
    plt.close(fig)


def cost(runs, out):
    """(a) F1 against model calls per item; (b) seconds per item."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(2 * COLUMN + 0.33, 1.8), gridspec_kw={"width_ratios": [1, 1.15]}
    )
    for config, data in runs.items():
        label, color, hatched = METHODS[config]
        items = data["metrics"]["items"]
        calls = data["run"]["llm_calls"] / items
        f1 = data["metrics"]["official"]["average"]["set_F"]
        if config == "uniqgen-gold-oracle":
            calls = runs["uniqgen"]["run"]["llm_calls"] / items if "uniqgen" in runs else calls
        left.scatter(calls, f1, s=42, color="white" if hatched else color, edgecolor=color,
                     linewidth=1.6, zorder=3)
        offset = {"uniqgen-gold-oracle": (6, 2), "uniqgen": (6, -9)}.get(config, (6, -3))
        left.annotate(f"{label}  {f1:.3f}", (calls, f1), textcoords="offset points",
                      xytext=offset, fontsize=6.5, color=INK)
    left.set_xscale("log")
    left.set_xlim(0.8, 40)
    left.set_ylim(0, 0.12)
    left.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    left.minorticks_off()
    left.set_xlabel("model calls per item (log scale)")
    left.set_ylabel("set $F_1$ (CK26)")
    left.set_title("(a) accuracy against model calls", loc="left")
    clean(left, "both")

    floor = 0.05  # cached replays take ~0 s; plot them at the floor
    order = [c for c in ("training-free", "uniqgen", "uniqgen-gold-oracle", "algebra-agent", "prover")
             if c in runs]
    for row, config in enumerate(order):
        label, color, hatched = METHODS[config]
        seconds = sorted(max(r["seconds"], floor) for r in runs[config]["rows"])
        jitter = [((k * 37) % 11 - 5) / 22 for k in range(len(seconds))]
        right.scatter(seconds, [row + j for j in jitter], s=7,
                      color="white" if hatched else color, edgecolor=color, linewidth=0.7,
                      alpha=0.85, zorder=3)
        raw = sorted(r["seconds"] for r in runs[config]["rows"])
        median = raw[min(len(raw) - 1, int(0.5 * len(raw)))]  # the runner's p50
        shown = max(median, floor)
        right.plot([shown, shown], [row - 0.32, row + 0.32], color=INK, linewidth=1.4, zorder=4)
        text = "median <0.1 s" if median < 0.1 else f"median {median:.1f} s"
        right.annotate(text, (shown, row - 0.32), textcoords="offset points", xytext=(3, 0),
                       ha="left", va="bottom", fontsize=5.5, color=INK)
        slowest = max(runs[config]["rows"], key=lambda r: r["seconds"])
        if slowest["seconds"] > 300:
            right.annotate(f"{slowest['seconds']:,.0f} s, one item", (slowest["seconds"], row),
                           textcoords="offset points", xytext=(-4, 6), ha="right",
                           fontsize=5.5, color=MUTED)
    right.set_yticks(range(len(order)), [METHODS[c][0] for c in order])
    right.invert_yaxis()
    right.set_xscale("log")
    right.set_xlim(floor * 0.8, 4000)
    right.set_xticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"])
    right.minorticks_off()
    right.set_xlabel("seconds per item (log scale; cached replays at 0.05)")
    right.set_title("(b) time per item", loc="left")
    clean(right)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out / "cost.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="paper/fig")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    runs = latest_runs(args.results, "ck26")
    outcomes(runs, out)
    cost(runs, out)
    print(f"wrote {sorted(p.name for p in out.glob('*.pdf'))} from runs: {sorted(runs)}")


if __name__ == "__main__":
    main()
