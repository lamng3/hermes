<div align="center">

# 🪽 Hermes

**SPARQL as relational algebra, optimized with the knowledge graph's statistics**

[![Manual](https://img.shields.io/badge/docs-manual-8b5cf6)](https://lamng3.com/hermes/)
[![Leaderboard](https://img.shields.io/badge/leaderboard-CK26%20%C2%B7%20DB26-f59e0b)](results/LEADERBOARD.md)
[![Python](https://img.shields.io/badge/python-3.11%2B-3b82f6)](pyproject.toml)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

[⚡ Quick start](#-quick-start) · [✨ What it is](#-what-it-is) · [📊 Results](#-results) · [🔧 Usage](#-usage) · [📝 Citation](#-citation)

</div>

<p align="center"><b>Lam Nguyen</b><br><sub>Case Western Reserve University</sub></p>

> [!NOTE]
> Hermes treats NL-to-SPARQL as query compilation: a question becomes a plan in
> the SPARQL algebra, checked against the ontology before it reaches the graph.
> Results are preliminary; the [write-up](paper/hermes.pdf) and
> [leaderboard](results/LEADERBOARD.md) are updated as runs finish. Install name
> `hermes-sparql`, import name `hermes`.

## ⚡ Quick start

```bash
git clone https://github.com/lamng3/hermes.git
cd hermes
pip install -e ".[dev]"
hermes data fetch ck26 db26          # download and normalize benchmarks
hermes run configs/ck26/prover.yaml
hermes report                        # leaderboard with official peer scores
```

One question at a time:

```bash
hermes generate --graph corporate --system prover --question "Who is Heinrich Hoch's manager?"
hermes generate --ontology onto.ttl --system prover --question "Which cities hosted the Winter Olympics?"
hermes generate --endpoint https://dbpedia.org/sparql --system algebra-agent --question "Where was Keanu Reeves born?"
```

From Python:

```python
from hermes import Context, KnowledgeGraph, ask

kg = KnowledgeGraph.from_files("onto.ttl", "data.ttl")      # or KnowledgeGraph.from_endpoint(url)
result = ask("Which cities hosted the Winter Olympics?", kg, Context(text="..."), system="prover")
print(result["sparql"])
```

Models: a local [Ollama](https://ollama.com) model by default, hosted APIs
(`provider: openai`, `anthropic`, `huggingface`, `together`, each reading its
usual key variable), or any Hugging Face model on a Modal GPU through
`deploy/modal_vllm.py` (`provider: vllm`). Every completion is cached under
`.cache/llm`, so reruns replay the same outputs. Long sweeps can run on a
machine over SSH or on Modal, and every run bounds its time and memory; see
[experiments/README.md](experiments/README.md#where-to-run).

## ✨ What it is

Translating natural-language questions into SPARQL lets users query knowledge
graphs without learning their schema. Large language models now write
plausible SPARQL, but they write the whole query at once: a query that fails
to parse, joins values of incompatible types, or returns nothing is detected
only after it is complete, and the repair is another guess. Hermes is a
query-planning agent that builds a SPARQL query one operator of the SPARQL
algebra at a time, over a typed formalization of the language. The RDF schema
gives every query variable a type, and a verifier checks each step before it
runs. After each step, execution on the graph tells the planner what that
operator did, the way a query optimizer estimates a rewrite before taking it.
A plan is accepted only when it is well typed and meets a goal stated from the
question.

1. **A typed SPARQL algebra.** Query plans are terms of the SPARQL 1.1
   algebra. The RDF schema induces a type for every variable, and
   well-formedness and typing rules decide, without evaluating a plan,
   whether it is valid (`hermes.verifier`).
2. **Verified query planning.** Actions are algebra operators; after each
   one the planner observes the number of solutions, how it changed, a sample
   of rows, and any error. A plan is accepted only when it is valid and
   satisfies a goal derived from the question (`hermes.algebra`).
3. **The Hermes planner.** A verifier rejects ill-typed steps before they
   reach the graph store, and an LLM planner drives it over local RDF files
   or remote SPARQL endpoints (`prover`, `algebra-agent`).
4. **Budgeted planning.** Compiled like SQL: the model parses the question
   once into a plan with bracketed phrases, which are bound to the ontology,
   type-checked, priced with System R estimates (`hermes.pricing`), and run
   best value for price first under a cap on model calls and executions
   (`budgeted`).
5. **A reproducible evaluation.** Six benchmarks in one normalized format and
   a scorer that reproduces all 5,024 published Text2SPARQL 2026 scores, with
   strict and reproducible-subset variants (`hermes.evaluation`, `data/`).

Earlier runs are in [results/archive](results/archive/).

Install name: `hermes-sparql`. Import name: `hermes`.

## 📊 Results

<div align="center">

| System | Qwen3-4B | Qwen3-8B | Qwen3-14B |
|:--|:--:|:--:|:--:|
| training-free <sub>one draft, one repair</sub> | 0.000 | 0.067 | 0.089 |
| uniqgen <sub>constraint table, Chase & Backchase</sub> | 0.084 | 0.121 | 0.197 |
| algebra-agent <sub>Hermes without the verifier</sub> | **0.103** | **0.178** | 0.231 |
| **prover** <sub>Hermes</sub> | 0.070 | 0.158 | **0.282** |

<sub>Official Text2SPARQL F1 on CK26 (100 items, English and German). Thinking off, temperature 0, up to 15 planning steps · `results/reports/2026-10-11-ck26-qwen3-sizes.md`, from the run folders under `results/ck26/`</sub>

</div>

What each system spends, per item (Qwen3-14B; the 4B and 8B runs are in the
[leaderboard](results/LEADERBOARD.md)):

<div align="center">

| System | LLM calls | Input tokens | Output tokens | Model seconds | GPU minutes, 100 items <sub>4B / 8B / 14B</sub> |
|:--|--:|--:|--:|--:|:--:|
| training-free | 1.06 | 734 | 113 | 5.1 | 1.1 / 1.8 / 3.0 |
| uniqgen | 1.99 | 3,916 | 273 | 18.5 | 1.6 / 2.8 / 2.7 |
| algebra-agent | 14.24 | 29,454 | 280 | 21.8 | 3.6 / 2.1 / 2.5 |
| **prover** | 12.65 | 28,559 | 245 | 22.6 | 1.0 / 2.5 / 2.6 |

<sub>Local vLLM: no per-token price, so cost is GPU time (wall clock; model seconds add up calls made in parallel). Set `HERMES_GPU_HOUR_USD` to put a dollar figure on it, or run a priced model; `hermes.costs`</sub>

</div>

The agents read about 40 times the input of a one-shot draft: each planning
step resends the vocabulary and the trace so far. Output stays small, so input
tokens, and prompt caching for them, decide what a hosted run would cost.

Hermes gains the most from model size (0.070 to 0.282) and is the best system
at 14B. At 4B and 8B, planning without the verifier scores higher, even though
Hermes leaves fewer items without a runnable query (35 against 57 at 4B, 2
against 15 at 8B). Per-run
precision, recall, strict and reproducible-subset scores, cost, and the
official peer submissions are in the [report](results/reports/2026-10-11-ck26-qwen3-sizes.md)
and the [leaderboard](results/LEADERBOARD.md).

## 🔧 Usage

<details>
<summary><b>SPARQL algebra as actions</b></summary>

`hermes.algebra` treats a query as a logical plan built one operator at a
time: `triple` (BGP and join), `filter`, `optional` (left join), `union`,
`minus`, `bind` (extend), `group`, `aggregate`, `having`, `order`, `project`,
`distinct`, `limit`, `offset`, `ask`, `remove`, plus the graph reads `lookup`
and `inspect`. The `Explorer` runs the plan after each action and reports what
happened, the way an optimizer costs a rewrite before taking it:

```
{"op": "triple", "pattern": "?city ex:hosted ?g"}     -> 19 solutions
{"op": "triple", "pattern": "?city ex:country ex:USA"} -> 19 -> 8 solutions
{"op": "triple", "pattern": "?g ex:season \"Winter\""} -> 8 -> 4 solutions
{"op": "distinct"}                                      -> 4 -> 3 solutions
```

`preview(action)` answers "what if?" without committing, and `undo` steps
back. Any policy can drive it: an LLM planner, a search, or a learned
planner. Existing generators are policies in the same space; constraint-guided
search with Chase and Backchase, for example, is a beam search over `remove`
and `triple` actions.

</details>

<details>
<summary><b>The verifier</b></summary>

`hermes.verifier` checks a plan against the RDF schema and a stated goal before
anything runs. `?x a C` gives `?x : C`, and `?x p ?y` gives `?x` the domain of
`p` and `?y` its range, compared up to `rdfs:subClassOf`. Unknown terms,
literals where entities are required, unbound variables, and ungrouped
projections are rejected with a message the planner can act on:

```
{"op": "triple", "pattern": "?c ex:hosts ?g"}
  -> rejected: kernel: unknown property ex:hosts (did you mean ex:hosted?)
```

The planner first states a goal, the shape of the answer the question asks
for (an entity of a class, a number, a yes/no, one or many), and `finish` is
refused while the plan does not meet it. The goal is written by the model
from the question, so the verifier guarantees that a plan has the shape its
goal asks for, not that the goal is the question.

</details>

<details>
<summary><b>Systems</b></summary>

A system is any callable `(question, kg, context) -> {"sparql": ...}`. `kg` is
a `KnowledgeGraph`: a store to run queries on and a vocabulary to prompt with.

| system | what it is | LLM calls |
|---|---|---|
| `prover` | **Hermes**: the planner with the verifier and goals | up to 15 |
| `algebra-agent` | Hermes without the verifier: planning with execution feedback only | up to 15 |
| `budgeted` | Hermes, budgeted: parse, bind, check, price, select; re-parse while the budget lasts | up to `llm_calls` (3) |
| `training-free` | baseline: one prompted draft from the vocabulary, one parse repair | 1-2 |
| `uniqgen` | baseline: UniQGen (Wang et al., [arXiv:2605.00845](https://arxiv.org/abs/2605.00845)), constraint table then Chase & Backchase | 2 |

Register your own with a decorator, or from another package with an entry
point in the `hermes.systems` group:

```python
import hermes

@hermes.systems.register("my-system")
class MySystem:
    def __init__(self, llm=None, depth=3):      # config `params` arrive here
        self.llm = llm
    def __call__(self, question, kg, context):
        return {"sparql": "SELECT * WHERE { ?s ?p ?o } LIMIT 1"}
```

</details>

<details>
<summary><b>Graph stores</b></summary>

Systems and the evaluator only call `store.query(sparql)`, which returns
SPARQL 1.1 JSON results, so the same planner runs on

- `OxigraphStore`: local RDF files in [Oxigraph](https://github.com/oxigraph/oxigraph), the default;
- `LocalStore`: local RDF files in rdflib (`engine: rdflib`);
- `EndpointStore`: any SPARQL endpoint, such as DBpedia or Wikidata.

`CachedStore` wraps them all, memoizing results and counting executions.

</details>

<details>
<summary><b>Benchmarks and evaluation</b></summary>

| benchmark | graph | questions | languages |
|---|---|---|---|
| `ck26` | Text2SPARQL corporate KG (local) | 50 | en, de |
| `db26` | Text2SPARQL DBpedia (organizers' endpoint) | 50 | en, es |
| `ck25`, `db25` | as above, 2025 editions | 50, 100 | en; en, es |
| `qald10` | Wikidata (live endpoint) | 394 | en, de, ru, zh |
| `qald9plus` | DBpedia (live endpoint) | 150 | en, de, ru |

`hermes.evaluation` reproduces the official Text2SPARQL scorer exactly: all
5,024 per-question and average scores published for Text2SPARQL'26 match to
the last digit. Each run reports the official metrics, a strict variant that
fixes the official scorer's ASK and ordering quirks, and the official metrics
on the items whose gold query reproduces on our graph store. See
[data/README.md](data/README.md) for formats, sources, licenses and caveats,
and [experiments/README.md](experiments/README.md) for the protocol.

`hermes serve --config configs/ck26/prover.yaml` exposes any system
behind the Text2SPARQL challenge API, so it can be submitted or scored with
the official client.

</details>

<details>
<summary><b>Design</b></summary>

The structure follows a few standard patterns, so methods plug in without
changes elsewhere:

| pattern | where |
|---|---|
| Strategy, plugin architecture | systems registry, `hermes.systems` entry points |
| Factory | `systems.create`, `llm.create`, `benchmarks.graph`, `store.local_store` |
| Dependency injection | systems receive the LLM and the knowledge graph |
| Repository, adapter | graph stores; benchmark adapters; the Text2SPARQL API |
| Decorator | `CachedStore`, `CachedLLM` |
| Command, builder | algebra actions and `render(plan)` |
| Template method, observer | `Run.evaluate` (predict, execute, score), run observers |

</details>

<details>
<summary><b>Repository layout</b></summary>

```
src/hermes/      source: stores, systems, algebra, benchmarks, evaluation
data/            normalized benchmarks and the graphs they run on (see data/README.md)
configs/         one YAML file per experiment: benchmark x system x model
experiments/     the protocol and the script that reproduces every result
results/         run outputs, official peer scores, and LEADERBOARD.md
tests/           offline tests; no model or network needed
```

</details>

## 📝 Citation

```bibtex
@misc{nguyen2026hermes,
  title  = {Hermes},
  author = {Nguyen, Lam},
  year   = {2026},
  url    = {https://github.com/lamng3/hermes}
}
```

## License

[CC BY 4.0](LICENSE). Copyright (c) 2026 Lam Nguyen. Benchmark files under
`data/benchmarks` keep their own licenses; see data/README.md.
