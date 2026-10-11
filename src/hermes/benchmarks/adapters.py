"""Adapters from source formats to Hermes' normalized benchmark format.

Each adapter returns (items, gold). Gold is the Text2SPARQL layout:
{item id: {value: relevance}, "order_required": [item ids]}.
"""

import json

import yaml

from hermes.benchmarks import Item
from hermes.evaluation import flatten

ORDER_FEATURE = "RESULT_ORDER_MATTERS"


def text2sparql(questions_yaml, gold_json=None, languages=None):
    """Text2SPARQL challenge YAML (2025, 2026) plus its true result set.

    Item ids follow the official scorer: "<prefix>:<id>-<language>". Gold comes
    from the official true result set when given.
    """
    data = yaml.safe_load(questions_yaml)
    prefix = data["dataset"]["prefix"]
    items = []
    for question in data["questions"]:
        features = list(question.get("features") or [])
        for lang, text in (question.get("question") or {}).items():
            if languages and lang not in languages:
                continue
            items.append(Item(
                id=f"{prefix}:{question['id']}-{lang}",
                qid=str(question["id"]),
                lang=lang,
                question=str(text).strip(),
                sparql=str(question.get("query", {}).get("sparql", "")).strip(),
                features=features,
                order_matters=ORDER_FEATURE in features,
            ))
    gold = {}
    if gold_json:
        official = json.loads(gold_json)
        ids = {item.id for item in items}
        gold = {key: value for key, value in official.items() if key in ids}
        ordered = [key for key in official.get("order_required", []) if key in ids]
        if ordered:
            gold["order_required"] = ordered
    return items, gold


def qald(qald_json, prefix, languages=None):
    """QALD JSON (QALD-9-plus, QALD-10): questions with answers attached."""
    data = json.loads(qald_json)
    items, gold = [], {}
    for question in data["questions"]:
        seen = set()
        answers = (question.get("answers") or [{}])[0]
        for entry in question.get("question", []):
            lang = entry.get("language")
            if not lang or lang in seen or (languages and lang not in languages):
                continue
            seen.add(lang)
            item = Item(
                id=f"{prefix}:{question['id']}-{lang}",
                qid=str(question["id"]),
                lang=lang,
                question=str(entry.get("string", "")).strip(),
                sparql=str(question.get("query", {}).get("sparql", "")).strip(),
                features=["AGGREGATION"] if question.get("aggregation") else [],
            )
            items.append(item)
            gold[item.id] = flatten(answers)
    return items, gold
