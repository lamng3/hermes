<div align="center">

# 🪽 Hermes

**Ontology-based natural-language querying over knowledge graphs**

[![Manual](https://img.shields.io/badge/docs-manual-8b5cf6)](https://lamng3.com/hermes/)
[![Demo](https://img.shields.io/badge/demo-in%20your%20browser-2a78d6)](https://lamng3.com/hermes/ask.html)
[![Python](https://img.shields.io/badge/python-3.11%2B-3b82f6)](pyproject.toml)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

</div>

Ask a question in English over a knowledge graph. Hermes writes SPARQL from the part of the ontology the question needs, repairs it, plans it as relational algebra with the graph's statistics, and runs it, pricing every model call and graph query before spending it.

Try it in your browser, nothing to install: [ask a question](https://lamng3.com/hermes/ask.html) · [plan a query](https://lamng3.com/hermes/plan.html).

## Quick start

```bash
pip install -e ".[dev]"
hermes data fetch ck26
hermes ask --graph corporate "Which department does Ms. Brant belong to?"
```

```python
from hermes import Budget, Hermes, KnowledgeGraph

kg = KnowledgeGraph.from_files("ontology.ttl", "data.ttl", schema=["ontology.ttl"])
answer = Hermes(kg, budget=Budget(llm_calls=3)).ask("Which cities hosted the Winter Olympics?")
print(answer.sparql, answer.rows, answer.spend.total)
```

The model runs locally through Ollama by default; set `HERMES_LLM_PROVIDER` to use OpenAI, Anthropic or another provider.

## How it works

1. **Refine**: pick the classes, properties, values and entities of the ontology the question needs.
2. **Examples**: add the closest question and query pairs, generated from the graph itself.
3. **Generate**: one model call writes the query.
4. **Repair**: fix prefixes, unknown terms and types, and probe facts about named entities; ask the model again only if needed and affordable.
5. **Plan**: translate into relational algebra and rewrite it: push selections down, order joins by cost, prune scans.
6. **Price**: estimate the queries and rows the plan will fetch and check them against the budget.
7. **Execute**: run the plan and return the answer.

## Results

The planner on the gold queries of the Text2SPARQL 2026 benchmarks ([report](results/planner/README.md)):

| | CK26 (local) | DB26 (DBpedia) |
|---|--:|--:|
| plans that ran and returned the engine's answers | 48 of 48 | 41 of 41 |
| intermediate rows, as written → optimized | 1,302,989 → 106,719 | 1,097,107 → 1,060,291 |
| plan seconds, as written → optimized | 4.67 → 1.35 | 68.4 → 29.7 |

The pipeline's accuracy and cost: [leaderboard](results/LEADERBOARD.md).

## Commands

```bash
hermes ask --graph corporate "Who is Heinrich Hoch's manager?"   # answer a question
hermes ask --files data.ttl --ontology onto.ttl "..."            # on your own graph
hermes plan --graph corporate --file query.rq                    # see a query planned and run
hermes fewshot build --graph corporate                           # examples from the graph
hermes sweep configs/sweeps/hermes.yaml && hermes report --write # benchmarks and leaderboard
```

More in the [manual](https://lamng3.com/hermes/).

## License

CC BY 4.0.
