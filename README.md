# Hermes: Verified Query Planning over the SPARQL Algebra with Language Agents

Official implementation of the Hermes paper.

**Lam Nguyen**, Case Western Reserve University · [Code](https://github.com/lamng3/hermes) · [Documentation](https://lamng3.github.io/hermes-docs/) · [Leaderboard](results/LEADERBOARD.md) · [Paper](paper/hermes.pdf)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Abstract

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

## Contributions

1. **A typed SPARQL algebra.** Query plans are terms of the SPARQL 1.1
   algebra. The RDF schema induces a type for every variable, and
   well-formedness and typing rules decide, without evaluating a plan,
   whether it is valid (`hermes.kernel`).
2. **Verified query planning.** Actions are algebra operators; after each
   one the planner observes the number of solutions, how it changed, a sample
   of rows, and any error. A plan is accepted only when it is valid and
   satisfies a goal derived from the question (`hermes.algebra`).
3. **The Hermes planner.** A verifier rejects ill-typed steps before they
   reach the graph store, and an LLM planner drives it over local RDF files
   or remote SPARQL endpoints (`prover`, `algebra-agent`).
4. **A reproducible evaluation.** Six benchmarks in one normalized format and
   a scorer that reproduces all 5,024 published Text2SPARQL 2026 scores, with
   strict and reproducible-subset variants (`hermes.evaluation`, `data/`).

Preliminary results are in the [paper](paper/hermes.pdf) and the
[leaderboard](results/LEADERBOARD.md).

Install name: `hermes-sparql`. Import name: `hermes`.

## Layout

```
src/hermes/      source: stores, systems, algebra, benchmarks, evaluation
data/            normalized benchmarks and the graphs they run on (see data/README.md)
configs/         one YAML file per experiment: benchmark x system x model
experiments/     the protocol and the script that reproduces every result
results/         run outputs, official peer scores, and LEADERBOARD.md
paper/           VLDB-format paper (`hermes.tex`, built PDF)
tests/           offline tests; no model or network needed
```

## Quick start

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

Models: a local [Ollama](https://ollama.com) model by default (`HERMES_OLLAMA_MODEL`,
`HERMES_OLLAMA_BASE_URL`), or any OpenAI-compatible API with
`HERMES_LLM_PROVIDER=openai`, `HERMES_LLM_API_KEY`, `HERMES_LLM_BASE_URL` and
`HERMES_LLM_MODEL`. Experiment configs set the model explicitly and cache
every completion under `.cache/llm`, so reruns replay the same outputs.

## SPARQL algebra as actions

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

## The verifier

`hermes.kernel` checks a plan against the RDF schema and a stated goal before
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

## Systems

A system is any callable `(question, kg, context) -> {"sparql": ...}`. `kg` is
a `KnowledgeGraph`: a store to run queries on and a vocabulary to prompt with.

| system | what it is | LLM calls |
|---|---|---|
| `prover` | **Hermes**: the planner with the verifier and goals | up to 15 |
| `algebra-agent` | Hermes without the verifier: planning with execution feedback only | up to 15 |
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

## Graph stores

Systems and the evaluator only call `store.query(sparql)`, which returns
SPARQL 1.1 JSON results, so the same planner runs on

- `OxigraphStore`: local RDF files in [Oxigraph](https://github.com/oxigraph/oxigraph), the default;
- `LocalStore`: local RDF files in rdflib (`engine: rdflib`);
- `EndpointStore`: any SPARQL endpoint, such as DBpedia or Wikidata.

`CachedStore` wraps them all, memoizing results and counting executions.

## Benchmarks and evaluation

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

## Design

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

## Citation

```bibtex
@misc{nguyen2026hermes,
  title  = {Hermes: Verified Query Planning over the SPARQL Algebra with Language Agents},
  author = {Nguyen, Lam},
  year   = {2026},
  url    = {https://github.com/lamng3/hermes}
}
```

## License

[CC BY 4.0](LICENSE). Copyright (c) 2026 Lam Nguyen. Benchmark files under
`data/benchmarks` keep their own licenses; see data/README.md.
