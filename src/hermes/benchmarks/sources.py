"""Known public benchmarks: where to download them and how to normalize them.

URLs are pinned to commits, so `hermes data fetch` reproduces the same files.
The manifest written for each benchmark records the URLs, their SHA-256 and
the license.
"""

import hashlib
from datetime import date

import httpx

from hermes.benchmarks import adapters, data_dir, save

T2S = "https://raw.githubusercontent.com/AKSW/text2sparql.aksw.org/935cb5a210717a7596a6028740136ac842db7ede"
T2S_EXAMPLES = "https://raw.githubusercontent.com/AKSW/text2sparql-client-examples/c6666be31796d6b8555031198a6d54df166d4406"
QALD10 = "https://raw.githubusercontent.com/KGQA/QALD-10/89caa86371c7a65c6c87a64e0eeaa0e247ff5fe5"
QALD9 = "https://raw.githubusercontent.com/KGQA/QALD_9_plus/b9fb0380902e6935efc9505170bc3652e8c242c1"
T2S_LICENSE = "CC BY-SA 4.0 (Text2SPARQL challenge, AKSW)"

SOURCES = {
    "ck26": {
        "title": "Text2SPARQL'26 corporate (CK26)",
        "adapter": "text2sparql",
        "graph": "corporate",
        "dataset_id": "https://text2sparql.aksw.org/2026/corporate/",
        "languages": ["en", "de"],
        "questions": f"{T2S}/docs/results/ck26/questions_ck26.yml",
        "gold": f"{T2S}/docs/results/ck26/ck26_true_result_set.json",
        "license": T2S_LICENSE,
    },
    "db26": {
        "title": "Text2SPARQL'26 DBpedia (DB26)",
        "adapter": "text2sparql",
        "graph": "dbpedia-t2s",
        "dataset_id": "https://text2sparql.aksw.org/2026/dbpedia/",
        "languages": ["en", "es"],
        "questions": f"{T2S}/docs/results/db26/questions_db26.yml",
        "gold": f"{T2S}/docs/results/db26/db26_true_result_set.json",
        "license": T2S_LICENSE,
    },
    "ck25": {
        "title": "Text2SPARQL'25 corporate (CK25)",
        "adapter": "text2sparql",
        "graph": "corporate",
        "dataset_id": "https://text2sparql.aksw.org/2025/corporate/",
        "languages": ["en"],
        "questions": f"{T2S_EXAMPLES}/examples_ck25/questions_ck25.yml",
        "gold": f"{T2S_EXAMPLES}/examples_ck25/ck25_true_result_set.json",
        "license": T2S_LICENSE,
    },
    "db25": {
        "title": "Text2SPARQL'25 DBpedia (DB25)",
        "adapter": "text2sparql",
        "graph": "dbpedia-t2s",
        "dataset_id": "https://text2sparql.aksw.org/2025/dbpedia/",
        "languages": ["en", "es"],
        "questions": f"{T2S_EXAMPLES}/examples_db25/questions_db25.yml",
        "gold": f"{T2S_EXAMPLES}/examples_db25/db25_true_result_set.json",
        "license": T2S_LICENSE,
    },
    "qald10": {
        "title": "QALD-10 (Wikidata)",
        "adapter": "qald",
        "graph": "wikidata",
        "languages": ["en", "de", "ru", "zh"],
        "questions": f"{QALD10}/data/qald_10/qald_10.json",
        "license": "MIT (KGQA/QALD-10)",
    },
    "qald9plus": {
        "title": "QALD-9-plus test (DBpedia)",
        "adapter": "qald",
        "graph": "dbpedia",
        "languages": ["en", "de", "ru"],
        "questions": f"{QALD9}/data/qald_9_plus_test_dbpedia.json",
        "license": "CC BY 4.0 (KGQA/QALD_9_plus)",
    },
}

GRAPH_FILES = {
    "corporate": {
        "graphs/corporate/prod-vocab.ttl": f"{T2S}/docs/participation/data/prod-vocab.ttl",
        "graphs/corporate/prod-inst.ttl": f"{T2S}/docs/participation/data/prod-inst.ttl",
    },
}


def fetch(name, refresh_graph=False):
    """Download one benchmark (and its local graph files), then normalize it."""
    source = SOURCES[name]
    files = {"questions": _download(source["questions"])}
    if "gold" in source:
        files["gold"] = _download(source["gold"])
    if source["adapter"] == "text2sparql":
        items, gold = adapters.text2sparql(
            files["questions"], files.get("gold"), source["languages"]
        )
    else:
        items, gold = adapters.qald(files["questions"], name, source["languages"])
    manifest = {
        "name": name,
        "title": source["title"],
        "graph": source["graph"],
        "languages": source["languages"],
        "license": source["license"],
        "source": {
            key: {"url": source[key], "sha256": hashlib.sha256(text.encode()).hexdigest()}
            for key, text in files.items()
        },
        "retrieved": date.today().isoformat(),
        "items": len(items),
    }
    if "dataset_id" in source:
        manifest["dataset_id"] = source["dataset_id"]
    root = save(name, manifest, items, gold)
    fetch_graph(source["graph"], refresh=refresh_graph)
    return root


def fetch_graph(name, refresh=False):
    for relative, url in GRAPH_FILES.get(name, {}).items():
        path = data_dir() / relative
        if path.exists() and not refresh:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_download(url), encoding="utf-8")


def _download(url):
    response = httpx.get(url, timeout=120.0, follow_redirects=True)
    response.raise_for_status()
    return response.text
