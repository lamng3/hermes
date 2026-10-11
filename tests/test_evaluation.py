import pytest

from hermes.evaluation import aggregate, flatten, ndcg, score, set_scores


def select(*values, var="result"):
    return {"head": {"vars": [var]},
            "results": {"bindings": [{var: {"type": "literal", "value": v}} for v in values]}}


def test_flatten_merges_all_variables_like_the_official_scorer():
    result = {"head": {"vars": ["a", "b"]}, "results": {"bindings": [
        {"a": {"type": "uri", "value": "x"}, "b": {"type": "literal", "value": "1"}},
        {"a": {"type": "uri", "value": "y"}},
    ]}}
    assert flatten(result) == {"x": 1, "y": 1, "1": 1}
    assert flatten({"head": {}, "boolean": False}) == {"true": 0}


def test_set_scores():
    assert set_scores({"a": 1, "b": 1}, {"a": 1}) == {"set_P": 0.5, "set_recall": 1.0,
                                                     "set_F": pytest.approx(2 / 3)}
    assert set_scores({}, {"a": 1})["set_F"] == 0.0


def test_official_ask_quirks_and_strict_fix():
    gold_true, gold_false = {"true": 1}, {"true": 0}
    said_false = {"head": {}, "boolean": False}
    assert score(said_false, gold_true)["set_F"] == 1.0
    assert score(said_false, gold_false)["set_F"] == 0.0
    assert score(said_false, gold_true, strict=True)["set_F"] == 0.0
    assert score(said_false, gold_false, strict=True)["set_F"] == 1.0


def test_official_ndcg_ignores_order_and_strict_uses_it():
    gold = {"a": 1, "b": 1}
    reverse = select("z", "b", "a")
    assert score(reverse, gold, order_matters=True)["ndcg"] == pytest.approx(
        ndcg({"a": 1, "b": 1, "z": 1}, gold)
    )
    assert score(select("a", "b", "z"), gold, True, strict=True)["ndcg"] == 1.0
    assert score(select("z", "a", "b"), gold, True, strict=True)["ndcg"] < 1.0


def test_set_f_ndcg_counts_the_mean_ndcg_once_more_like_the_official_scorer():
    items = {"q:1-en": {"set_P": 1, "set_recall": 1, "set_F": 1.0},
             "q:2-en": {"set_P": 0, "set_recall": 0, "set_F": 0.0, "ndcg": 0.5}}
    assert aggregate(items)["average"]["set_F_ndcg"] == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert aggregate(items, strict=True)["average"]["set_F_ndcg"] == pytest.approx(0.75)


def test_per_language_averages():
    items = {"q:1-en": {"set_P": 1, "set_recall": 1, "set_F": 1.0},
             "q:1-de": {"set_P": 0, "set_recall": 0, "set_F": 0.0}}
    report = aggregate(items, ["en", "de"])
    assert report["average-en"]["set_F"] == 1.0
    assert report["average-de"]["set_F"] == 0.0
    assert report["average"]["set_F"] == 0.5
