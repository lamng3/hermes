# CK26 and DB26 · Budgeted planner, Llama 3.1 8B

Runs from 2026-10-25, committed under `results/ck26/budgeted/` and `results/db26/budgeted/`. Baselines are the Llama 3.1 runs of 2026-10-11.

## Config

| | |
|---|---|
| Configs | `configs/ck26/budgeted.yaml`, `configs/db26/budgeted.yaml` |
| Benchmarks | CK26 (100 items, en and de; corporate graph on Oxigraph), DB26 (100 items, en and es; DBpedia endpoint) |
| Model | llama3.1 (8B), Ollama, local, temperature 0 |
| Budget | 3 model calls, 8 candidate executions per item |
| Selector | default weights (`hermes/selection.py`) |
| Limits | 30 s per query, 300 s per item |
| Development exposure | binding repairs designed on CK26 questions 1–10 (20 items); held-out = questions 11–50 (80 items) |

## Runs

| Run | Folder | Commit |
|---|---|---|
| budgeted, CK26 | `20261025-222853` | 18d203d |
| budgeted, DB26 | `20261025-225005` | 18d203d |
| training-free, CK26 | `20261011-003832` | (2026-10-11) |
| uniqgen, CK26 | `20261011-004445` | (2026-10-11) |
| algebra-agent, CK26 | `20261011-020623` | (2026-10-11) |
| training-free, DB26 | `20261011-015735` | (2026-10-11) |

Later commits (1862fef: catalog cache, English labels, label tie-break) change binding on DBpedia; the runs above predate them.

## Accuracy

| Run | P | R | F1 | F1-nDCG | Strict | Repro. | Exact | F1 dev (Q1–10) | F1 held-out (Q11–50) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| training-free, CK26 | 0.036 | 0.015 | 0.017 | 0.018 | 0.017 | 0.020 | 1 | 0.050 | 0.009 |
| uniqgen, CK26 | 0.061 | 0.142 | 0.064 | 0.064 | 0.064 | 0.066 | 3 | 0.072 | 0.062 |
| algebra-agent, CK26 | 0.101 | 0.300 | 0.094 | 0.094 | 0.094 | 0.092 | 4 | 0.146 | 0.081 |
| **budgeted, CK26** | **0.127** | 0.118 | **0.111** | **0.110** | **0.111** | **0.114** | **8** | 0.200 | **0.089** |
| training-free, DB26 | 0.038 | 0.011 | 0.017 | 0.017 | 0.017 | 0.017 | 0 | 0.055 | 0.007 |
| budgeted, DB26 | 0.004 | 0.020 | 0.007 | 0.007 | 0.007 | 0.007 | 0 | 0.000 | 0.008 |

## Cost

| Run | Calls/item | Input tokens/item | Output tokens/item | Graph executions/item | Wall (min) |
|---|--:|--:|--:|--:|--:|
| training-free, CK26 | 1.16 | – | – | – | 6.2 |
| uniqgen, CK26 | 1.98 | – | – | – | 35.6 |
| algebra-agent, CK26 | 14.51 | – | – | – | 40.4 |
| budgeted, CK26 | 2.15 | 2,494 | 333 | 6.3 | 21.2 |
| budgeted, DB26 | 2.86 | 1,944 | 317 | 16.4 | 40.5 |

Token counts were not recorded for Ollama runs before 2026-10-25.

## Selection

| | CK26 | DB26 |
|---|--:|--:|
| F1 of chosen candidates | 0.111 | 0.007 |
| F1 of best executed candidate (oracle) | 0.112 | – |
| F1 of best candidate generated (oracle) | 0.113 | – |
| Items with no candidate | 40 | 26 |
| Items answered (not abstained) | 27 | 6 |
| F1 on answered items | 0.411 | 0.111 |
| Executed candidates correct (F1 ≥ 0.5) | – | 1 of 527 |

Mean F1 of executed CK26 candidates ranked by p: top 10% 0.425, top 25% 0.169, top 50% 0.089, all 0.047.

## Errors

| Run | Query errors | Items without a query |
|---|--:|--:|
| budgeted, CK26 | 0 | 40 |
| budgeted, DB26 | 2 (model-written `xsd:datetime(...)`, a malformed projection) | 26 |
