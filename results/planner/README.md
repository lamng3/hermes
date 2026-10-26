# Planner study: the gold queries of CK26 and DB26

`PYTHONPATH=src python experiments/planner_study.py ck26 db26`. Every distinct gold query is translated into relational algebra as written and as optimized; both plans run with the plan executor (each scan a single-pattern query, joins as hash or bind joins), and their answers are checked against the store's engine running the query text.

## Config

| | |
|---|---|
| CK26 | 50 distinct gold queries; corporate graph (26,903 triples) in Oxigraph 0.5.11, local |
| DB26 | 50 distinct gold queries; Text2SPARQL DBpedia snapshot, remote endpoint (Tentris) |
| Rules | push selections, order joins (dynamic programming over C_out, bind joins priced as index access), prune scans |
| Executor | bind joins up to 500 bindings, 100 per query; at most 200,000 rows per scan |
| Referee | rdflib's engine, when Oxigraph and the plan disagree (CK26) |

## Runs

| Benchmark | Folder | Commit | Note |
|---|---|---|---|
| CK26 | `ck26/20261026-004334` | dadb476 | reported |
| CK26 | `ck26/20261026-004052`, `-004131`, `-004241` | e3e0a87 to dadb476 | superseded: before EXISTS read the graph, numbers compared canonically, the ties and referee checks, and path estimates |
| DB26 | `db26/20261026-005015` | dadb476 | reported |

## Correctness

| Status | CK26 | DB26 |
|---|--:|--:|
| same answers, as written and optimized | 46 | 41 |
| same up to rows tied under ORDER BY with LIMIT | 1 | 0 |
| engine wrong, plan right (Oxigraph reads `a / b * c` as `a / (b * c)`; rdflib agrees with the plan) | 1 | 0 |
| the engine cannot run the gold query (`xsd:int(...)` unsupported) | 2 | 0 |
| a scan over 200,000 rows (`?x a dbo:Person`, `?film dbo:starring ?actor`) | 0 | 4 |
| the endpoint answered HTTP 500 | 0 | 5 |
| **plans that ran and gave other answers** | **0** | **0** |

## Work, on the queries that ran

| | CK26 as written | CK26 optimized | DB26 as written | DB26 optimized |
|---|--:|--:|--:|--:|
| rows fetched | 56,460 | 48,429 | 622,861 | 588,535 |
| intermediate rows | 1,302,989 | 106,719 | 1,097,107 | 1,060,291 |
| store queries | 188 | 184 | 110 | 110 |
| plan seconds | 4.67 | 1.35 | 68.4 | 29.7 |
| engine seconds (reference) | 0.29 | | 15.1 | |
| queries rewritten | | 19 | | 11 |
| queries that fetched more | | 2 | | 2 |

Most gold queries have one or two patterns and nothing to reorder (the median ratio of rows fetched is 1.0); the savings come from the multi-pattern ones, for example CK26 question 28 (intermediate rows 419,035 → 3,353) and DB26 question 41 (rows fetched 32,896 → 27).

## Estimates

| | CK26 | DB26 |
|---|--:|--:|
| q-error of the result's row estimate, median | 1.10 | 1.41 |
| q-error, 90th percentile | 5.17 | 10.79 |
