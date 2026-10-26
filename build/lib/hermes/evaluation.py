"""Answer-set metrics, computed the way the Text2SPARQL challenge computes them.

The official scorer (text2sparql-client, built on trec_eval) flattens every
binding of every variable into one set of value strings and treats each value
as a relevant document. It scores set precision, set recall and set F1 per
question, nDCG for questions whose result order matters, and ranks systems by
`set_F_ndcg`: nDCG where order matters, set F1 elsewhere.

The default mode reproduces the published numbers exactly, quirks included:

- ASK: a gold `true` scores 1.0 whatever the system answers, and a gold
  `false` scores 0.0 for every system (trec_eval relevance semantics).
- nDCG ignores result order, since every value has the same score.
- `set_F_ndcg` counts the scope's mean nDCG as one extra item.

`strict=True` fixes all three: booleans score by exact match, nDCG uses the
returned order, and `set_F_ndcg` is a plain mean.
"""

import math
import statistics

OFFICIAL = ("set_P", "set_recall", "set_F")


def flatten(result):
    """SPARQL JSON results -> {value: relevance}, as the official scorer does."""
    if not result:
        return {}
    if "boolean" in result:
        return {"true": 1 if result["boolean"] else 0}
    values = {}
    variables = result.get("head", {}).get("vars", [])
    for variable in variables:
        for binding in result.get("results", {}).get("bindings", []):
            if variable in binding:
                values[binding[variable]["value"]] = 1
    return values


def ordered_values(result):
    """Values in result order, for the order-aware strict nDCG."""
    seen = []
    if not result or "boolean" in result:
        return seen
    variables = result.get("head", {}).get("vars", [])
    for binding in result.get("results", {}).get("bindings", []):
        for variable in variables:
            value = binding.get(variable, {}).get("value")
            if value is not None and value not in seen:
                seen.append(value)
    return seen


def set_scores(predicted, gold):
    """trec_eval set_P, set_recall and set_F (beta = 1)."""
    relevant = {value for value, grade in gold.items() if grade > 0}
    hits = len(relevant & set(predicted))
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    total = precision + recall
    f1 = 2 * precision * recall / total if total else 0.0
    return {"set_P": precision, "set_recall": recall, "set_F": f1}


def ndcg(predicted, gold, ranking=None):
    """trec_eval ndcg over the whole run.

    trec_eval ranks a run by score, then by document id in decreasing order.
    Every predicted value has score 1 in the official pipeline, so the official
    ranking ignores result order; pass `ranking` to score the real order.
    """
    if ranking is None:
        ranking = sorted(predicted, reverse=True)
    gains = [gold.get(value, 0) for value in ranking]
    ideal = sorted((grade for grade in gold.values() if grade > 0), reverse=True)
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    best = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal))
    return dcg / best if best else 0.0


def score(predicted_result, gold, order_matters=False, strict=False):
    """Score one question. `gold` is the flattened gold result set."""
    predicted = flatten(predicted_result)
    if strict and _is_boolean(gold):
        same = _is_boolean(predicted) and predicted == gold
        value = 1.0 if same else 0.0
        scores = {"set_P": value, "set_recall": value, "set_F": value}
    else:
        scores = set_scores(predicted, gold)
    if order_matters:
        ranking = ordered_values(predicted_result) if strict else None
        scores["ndcg"] = ndcg(predicted, gold, ranking)
    return scores


def aggregate(per_item, languages=(), strict=False):
    """Averages in the official layout: overall and per language.

    `per_item` maps item ids such as "ck26:7-en" to score dicts.
    """
    report = {"average": _average(per_item, strict)}
    if len(languages) > 1:
        for language in languages:
            subset = {
                key: value for key, value in per_item.items() if key.endswith(f"-{language}")
            }
            report[f"average-{language}"] = _average(subset, strict)
    return report


def _average(per_item, strict=False):
    if not per_item:
        return {}
    scores = list(per_item.values())
    averages = {
        metric: statistics.fmean(item[metric] for item in scores) for metric in OFFICIAL
    }
    ordered = [item["ndcg"] for item in scores if "ndcg" in item]
    ranked = [item.get("ndcg", item["set_F"]) for item in scores]
    if ordered:
        averages["ndcg"] = statistics.fmean(ordered)
        if not strict:
            ranked.append(averages["ndcg"])
    averages["set_F_ndcg"] = statistics.fmean(ranked)
    return averages


def _is_boolean(values):
    return set(values) == {"true"}
