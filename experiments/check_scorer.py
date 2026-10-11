"""Check that hermes.evaluation reproduces the official Text2SPARQL'26 scores.

Downloads every team's predicted result set and official scores from the
challenge repository (pinned commit), rescores the predictions with Hermes,
and compares every per-question and average number.

    PYTHONPATH=src python experiments/check_scorer.py
"""

import sys

import httpx

from hermes.evaluation import aggregate, ndcg, set_scores

BASE = (
    "https://raw.githubusercontent.com/AKSW/text2sparql.aksw.org/"
    "935cb5a210717a7596a6028740136ac842db7ede/docs/results"
)
TEAMS = [
    "ADFR", "INFAI-ETI-AND-FRIENDS-A", "INFAI-ETI-AND-FRIENDS-B", "INFAI-ETI-AND-FRIENDS-C",
    "IRIS", "LIBER-AI-CLAUDE", "LIBER-AI-QWEN", "SPARQL-LLM",
]
BENCHMARKS = {"ck26": ["en", "de"], "db26": ["en", "es"]}


def get(path):
    response = httpx.get(f"{BASE}/{path}", timeout=60.0)
    response.raise_for_status()
    return response.json()


def main():
    compared = differences = 0
    for bench, languages in BENCHMARKS.items():
        gold = get(f"{bench}/{bench}_true_result_set.json")
        order = set(gold.pop("order_required", []))
        for team in TEAMS:
            predicted = get(f"{bench}/{team}/{bench}_pred_result_set.json")
            official = get(f"{bench}/{team}/{bench}_results.json")
            mine = {}
            for key, values in predicted.items():
                mine[key] = set_scores(values, gold[key])
                if key in order:
                    mine[key]["ndcg"] = ndcg(values, gold[key])
            averages = aggregate(mine, languages)
            for scope, scores in list(mine.items()) + list(averages.items()):
                for metric, value in scores.items():
                    if metric in official.get(scope, {}):
                        compared += 1
                        if abs(official[scope][metric] - value) > 1e-9:
                            differences += 1
                            print(f"{bench} {team} {scope} {metric}: "
                                  f"official {official[scope][metric]} hermes {value}")
    print(f"{compared} official numbers compared, {differences} differ.")
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
