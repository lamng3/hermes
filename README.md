<div align="center">

# 🪽 Hermes

**Ontology-based natural-language querying over knowledge graphs**

[![Manual](https://img.shields.io/badge/docs-manual-8b5cf6)](https://lamng3.com/hermes/)
[![Demo](https://img.shields.io/badge/demo-in%20your%20browser-2a78d6)](https://lamng3.com/hermes/ask.html)
[![Python](https://img.shields.io/badge/python-3.11%2B-3b82f6)](pyproject.toml)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

</div>

Ask a question in English over a knowledge graph; Hermes writes SPARQL from the slice of the ontology the question needs, repairs it, plans it as relational algebra with the graph's statistics, and runs it, pricing every model call and graph query before it spends it.

```bash
pip install -e ".[dev]"
hermes data fetch ck26
hermes ask --graph corporate "Which department does Ms. Brant belong to?"
```

```python
from hermes import Budget, Hermes, KnowledgeGraph

kg = KnowledgeGraph.from_files("ontology.ttl", "data.ttl", schema=["ontology.ttl"])
answer = Hermes(kg, budget=Budget(llm_calls=3, rows=50_000)).ask("Which cities hosted the Winter Olympics?")
answer.sparql, answer.rows, answer.spend.total
```

Try it without installing anything: [ask a question](https://lamng3.com/hermes/ask.html) or [plan a query](https://lamng3.com/hermes/plan.html); Hermes's own Python runs in the page.

## How it works

The pipeline follows NL2KQL ([Tang et al., arXiv 2404.02933](https://arxiv.org/abs/2404.02933)), which translates questions into Kusto queries over a database schema, with the ontology as the schema and the knowledge graph as the data, and adds a query planner and budgets.

| Stage | What it does | Code |
|---|---|---|
| **Refine** | the ontology slice the question needs: top classes, properties and listed values by BM25 or embeddings, what the chosen classes use, and the entities the question names | [`pipeline/refiner.py`](src/hermes/pipeline/refiner.py), [`pipeline/catalog.py`](src/hermes/pipeline/catalog.py) |
| **Examples** | the two closest synthetic question and query pairs, generated from the graph and kept only if they survive a round trip | [`pipeline/synthetic.py`](src/hermes/pipeline/synthetic.py), [`pipeline/fewshot.py`](src/hermes/pipeline/fewshot.py) |
| **Generate** | one model call writes the query | [`pipeline/prompt.py`](src/hermes/pipeline/prompt.py) |
| **Repair** | prefixes, unknown terms by name, guessed entities by label, a type check, an ASK probe per pattern about a named entity; the model again only while the budget allows | [`pipeline/repair.py`](src/hermes/pipeline/repair.py) |
| **Plan** | the query as relational algebra, rewritten: selections pushed down, joins ordered by dynamic programming over C_out, scans pruned | [`planner/`](src/hermes/planner) |
| **Price** | the plan's queries and rows, estimated from the graph's statistics, checked against the budget | [`pricing/budget.py`](src/hermes/pricing/budget.py) |
| **Execute** | scans as single-pattern queries, joins as hash or bind joins, so the plan decides what is fetched | [`planner/executor.py`](src/hermes/planner/executor.py) |

## Results

**The planner** on the gold queries of the Text2SPARQL 2026 benchmarks ([report](results/planner/README.md)): every plan that ran gave the engine's answers, and the optimizer cut the work.

| | CK26 (local) | DB26 (DBpedia endpoint) |
|---|--:|--:|
| plans that ran and gave the engine's answers | 48 of 48 | 41 of 41 |
| intermediate rows, as written → optimized | 1,302,989 → 106,719 | 1,097,107 → 1,060,291 |
| rows fetched, as written → optimized | 56,460 → 48,429 | 622,861 → 588,535 |
| plan seconds, as written → optimized | 4.67 → 1.35 | 68.4 → 29.7 |
| q-error of the row estimate, median / p90 | 1.10 / 5.17 | 1.41 / 10.79 |

On the way, the study found that Oxigraph 0.5.11 evaluates `a / b * c` as `a / (b * c)` (and `8 - 2 - 1` as 7): one CK26 gold answer computed with it is wrong.

**The pipeline**: see [results/](results/) and the [leaderboard](results/LEADERBOARD.md).

<details>
<summary><b>Commands</b></summary>

```bash
hermes ask --graph corporate "Who is Heinrich Hoch's manager?" --calls 3 --rows 50000
hermes ask --files data.ttl --ontology onto.ttl "..."      # your own graph
hermes plan --graph corporate --file query.rq              # a query as relational algebra, optimized and run
hermes fewshot build --graph corporate --count 30          # synthetic examples from the graph
hermes data fetch ck26 db26                                 # benchmarks and their graphs
hermes run configs/ck26/hermes.yaml                         # one benchmark run
hermes sweep configs/sweeps/hermes.yaml                     # Hermes, ablations and baselines
hermes report --write                                       # the leaderboard
hermes serve --config configs/ck26/hermes.yaml              # the Text2SPARQL challenge API
```

The model is local through Ollama by default (`HERMES_OLLAMA_MODEL`), or set `HERMES_LLM_PROVIDER` to `openai`, `anthropic`, `together`, `huggingface`, `vllm` or `modal`.

</details>

<details>
<summary><b>Design</b></summary>

| Pattern | Where |
|---|---|
| Template method | `Hermes.ask` runs the stages in order; a stage is a class with `run` |
| Strategy | retrievers (`Lexical`, `Embeddings`), executors (`PlanExecutor`, `EngineExecutor`), rewrite rules |
| Command | each rewrite rule rewrites a plan and logs what it did |
| Builder | `PromptBuilder` |
| Factory | `retriever(...)`, `executor_for(...)`, `llm.create(...)`, `benchmarks.graph(...)`, `systems.create(...)` |
| Dependency injection | `Hermes(kg, llm, budget, fewshots, retriever, executor, observers, stages)` |
| Decorator | `BudgetedLLM`, `CachedLLM`, `CachedStore` |
| Adapter | `BrowserStore` (Oxigraph in the page), `StoreTriples` (rdflib over single-pattern queries) |
| Repository | `GraphStore`: Oxigraph, rdflib, endpoint, browser; `FewShotStore` |
| Observer | `Recorder`, `Printer` on pipeline events |
| Plugin | systems register by name, or through the `hermes.systems` entry point |

</details>

<details>
<summary><b>Layout</b></summary>

```
src/hermes/pipeline/    the stages: catalog, refiner, few-shots, prompt, repair, and Hermes itself
src/hermes/planner/     relational algebra, estimates, rewrite rules, optimizer, executors
src/hermes/pricing/     graph statistics, model prices, budgets
src/hermes/graph/       stores, knowledge graphs, linking, the browser store
src/hermes/sparql/      parsing, repairs, the verifier
src/hermes/retrieval.py BM25 and embeddings
src/hermes/baselines/   training-free, UniQGen, the algebra agent, the prover, the budgeted planner
configs/                runs and sweeps
data/                   benchmarks, graphs, synthetic few-shots
experiments/            the planner study, scorer check, sweeps on Modal
results/                runs, reports, the leaderboard; earlier runs in results/archive
```

</details>

## License

CC BY 4.0.
