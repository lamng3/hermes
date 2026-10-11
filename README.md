# Hermes: Natural Language to SPARQL over Ontologies

**Lam Nguyen** · [Code](https://github.com/lamng3/hermes) · [Documentation](https://lamng3.github.io/hermes-docs/) · [Leaderboard](results/LEADERBOARD.md) · [Paper outline](paper/README.md)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Abstract

Hermes carries a question across to the graph. Named for the Greek messenger
and interpreter, it translates natural-language questions into SPARQL over
ontologies and knowledge graphs. Existing methods
are reimplemented as interchangeable *systems* behind one interface,
benchmarks are converted into one normalized format, and every experiment is
a config whose predictions, metrics and provenance are kept under `results/`.
On top of this foundation we study NL-to-SPARQL as planning over the SPARQL
algebra: a query is built one operator at a time, and execution on the graph
after each operator tells the planner what that operator did, the way a query
optimizer costs a rewrite before taking it. We evaluate on the Text2SPARQL
challenge and on QALD, with a scorer that reproduces the official
Text2SPARQL'26 numbers exactly, so new systems are compared with existing
ones and with the published entries on the same footing.

## Contributions

1. **SPARQL algebra as actions.** Operators of the SPARQL 1.1 algebra as
   symbolic actions, with an explorer that reports cardinality, deltas,
   samples and errors after every step (`hermes.algebra`).
2. **One interface for methods.** Prompt-only generation, UniQGen
   (Chase & Backchase, [arXiv:2605.00845](https://arxiv.org/abs/2605.00845))
   and an algebra agent run as swappable systems over local files or remote
   endpoints.
3. **Normalized benchmarks.** Text2SPARQL CK25, CK26, DB25, DB26, QALD-10 and
   QALD-9-plus in one format, with pinned sources, licenses and documented
   caveats (`data/`).
4. **Comparable evaluation.** The official Text2SPARQL metrics reproduced on
   all 5,024 published scores, plus a strict variant and a reproducible
   subset (`hermes.evaluation`, `experiments/check_scorer.py`).

Install name: `hermes-sparql`. Import name: `hermes`.

## Layout

```
src/hermes/      source: stores, systems, algebra, benchmarks, evaluation
data/            normalized benchmarks and the graphs they run on (see data/README.md)
configs/         one YAML file per experiment: benchmark x system x model
experiments/     the protocol and the script that reproduces every result
results/         run outputs, official peer scores, and LEADERBOARD.md
paper/           the paper outline
tests/           offline tests; no model or network needed
```

## Quick start

```bash
git clone https://github.com/lamng3/hermes.git
cd hermes
pip install -e ".[dev]"
hermes data fetch ck26 db26          # download and normalize benchmarks
hermes run configs/ck26/training-free.yaml
hermes report                        # leaderboard with official peer scores
```

One question at a time:

```bash
hermes generate --graph corporate --question "Who is Heinrich Hoch's manager?" --system algebra-agent
hermes generate --ontology GeoOutage.ttl --question "Which counties have the most outages?"
hermes generate --endpoint https://dbpedia.org/sparql --question "Where was Keanu Reeves born?"
```

From Python:

```python
from hermes import Context, KnowledgeGraph, ask

kg = KnowledgeGraph.from_files("onto.ttl", "data.ttl")      # or KnowledgeGraph.from_endpoint(url)
result = ask("Which counties have the most outages?", kg, Context(text="..."), system="uniqgen")
print(result["sparql"])
```

Models: a local [Ollama](https://ollama.com) model by default (`HERMES_OLLAMA_MODEL`,
`HERMES_OLLAMA_BASE_URL`), or any OpenAI-compatible API with
`HERMES_LLM_PROVIDER=openai`, `HERMES_LLM_API_KEY`, `HERMES_LLM_BASE_URL` and
`HERMES_LLM_MODEL`. Experiment configs set the model explicitly and cache
every completion under `.cache/llm`, so reruns replay the same outputs.

## Systems

A system is any callable `(question, kg, context) -> {"sparql": ...}`. `kg` is
a `KnowledgeGraph`: a store to run queries on and a vocabulary to prompt with.

| system | idea | LLM calls |
|---|---|---|
| `training-free` | one prompted draft from the vocabulary, fixed stages, one parse repair | 1-2 |
| `uniqgen` | UniQGen (Wang et al., [arXiv:2605.00845](https://arxiv.org/abs/2605.00845)): constraint table, then Chase & Backchase beam search against reference answers | 2 |
| `algebra-agent` | an LLM applies SPARQL algebra operators one at a time and reads the effect of each | up to 15 |

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
back. Any policy can drive it: an LLM (`algebra-agent`), a search, or a
learned planner. UniQGen's Chase and Backchase are, in these terms, a beam
search over `remove` and `triple` actions.

## Graph stores

Systems and the evaluator only call `store.query(sparql)`, which returns
SPARQL 1.1 JSON results. That closes the gap between ontology-only and
execution-guided methods: the same system runs on

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

`hermes serve --config configs/ck26/algebra-agent.yaml` exposes any system
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
  title  = {Hermes: Natural Language to SPARQL over Ontologies},
  author = {Nguyen, Lam},
  year   = {2026},
  url    = {https://github.com/lamng3/hermes}
}
```

## License

[CC BY 4.0](LICENSE). Copyright (c) 2026 Lam Nguyen. Benchmark files under
`data/benchmarks` keep their own licenses; see data/README.md.
