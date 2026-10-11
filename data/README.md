# Data

Benchmarks in one normalized format, and the graphs they run on.

```
data/graphs.yaml                     graph declarations: local files or an endpoint
data/graphs/                         downloaded graph files (not committed)
data/benchmarks/<name>/benchmark.yaml   manifest: graph, languages, license, source URLs and SHA-256
data/benchmarks/<name>/questions.jsonl  one row per (question, language)
data/benchmarks/<name>/gold.json        gold answer sets
```

`hermes data fetch <name>` downloads a benchmark from a pinned commit,
normalizes it, and downloads its graph if it is local. `hermes data verify
<name>` runs every gold query on the graph and reports where the answers
differ from the gold sets.

## Format

A question row:

```json
{"id": "ck26:7-en", "qid": "7", "lang": "en", "question": "...", "sparql": "...",
 "features": ["SELECT", "ORDER"], "order_matters": false}
```

`id` follows the official Text2SPARQL scorer, `<prefix>:<question id>-<language>`.
`gold.json` maps each id to `{value: 1}`, with every value of every variable
flattened into one set, and lists ids whose order matters under
`order_required`. This is the official layout, so official result files and
Hermes results can be compared directly. QALD answers are converted to the
same layout.

To add a benchmark, write an adapter in `src/hermes/benchmarks/adapters.py`
that returns `(items, gold)`, and an entry in `sources.py`.

## Sources and licenses

| name | source | license |
|---|---|---|
| ck26, db26 | [AKSW/text2sparql.aksw.org](https://github.com/AKSW/text2sparql.aksw.org) `docs/results` | CC BY-SA 4.0 |
| ck25, db25 | [AKSW/text2sparql-client-examples](https://github.com/AKSW/text2sparql-client-examples) | CC BY-SA 4.0 (questions from the challenge) |
| corporate graph | [AKSW/text2sparql.aksw.org](https://github.com/AKSW/text2sparql.aksw.org) `docs/participation/data` | CC BY 4.0 (vocabulary), CC BY-SA 4.0 (repository) |
| qald10 | [KGQA/QALD-10](https://github.com/KGQA/QALD-10) | MIT |
| qald9plus | [KGQA/QALD_9_plus](https://github.com/KGQA/QALD_9_plus) | CC BY 4.0 |

The normalized files are derived from these sources and keep their licenses.
Exact URLs, commits and checksums are in each `benchmark.yaml`.

## Caveats worth knowing before reading any number

1. **CK26 is CK25 paraphrased.** 49 of the 50 CK26 gold queries appear
   verbatim in CK25, and both run on the same graph. CK25 is not a
   development set for CK26, and few-shot examples from one leak the other.
   Runs count such overlap and report it as `contaminated_examples`. DB25
   and DB26 share no gold query.
2. **The official ASK scoring is degenerate.** The scorer treats an ASK
   answer as one document named `true`, with relevance 1 or 0. A gold `true`
   scores 1.0 whatever the system answers; a gold `false` scores 0.0 for
   every system. Hermes reports these official numbers and, as `strict`,
   scores booleans by exact match.
3. **Official nDCG ignores result order.** Every returned value has the same
   score, so trec_eval orders them by value, not by the system's order.
   `strict` uses the returned order.
4. **The official `set_F_ndcg` counts the mean nDCG as one extra item**
   in each average. Hermes reproduces it, so its numbers match the
   leaderboard; `strict` uses a plain mean.
5. **The CK gold answers depend on the engine they were computed on**
   (Tentris). On Oxigraph, 43 of 50 CK26 gold queries return the gold answers.
   The other seven break ties under LIMIT differently, format decimals
   differently, or use `xsd:int()`, which is not a SPARQL 1.1 function. Runs
   flag these items with `gold_reproduces` and report the `reproducible`
   subset next to the full set. On rdflib, 42 of 50 agree. On the organizers'
   DBpedia endpoint, all 50 DB26 gold queries agree.
6. **QALD answers drift.** QALD-9-plus targets DBpedia 2016-10 and QALD-10
   was answered on Wikidata in 2022; the live endpoints have changed since.
