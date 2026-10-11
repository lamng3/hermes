# Experiments

`bash experiments/run_all.sh` reproduces every run in `results/`, then
rewrites `results/LEADERBOARD.md`. Pass config paths to run a subset.

`python experiments/figures.py` (with the `figures` extra installed) rebuilds
the paper's figures in `paper/fig/` from the latest finished run of each
configuration, so the charts follow the results.

## A run

`hermes run configs/<benchmark>/<config>.yaml` does, for each item:

1. **predict**: the system writes SPARQL for the question;
2. **execute**: the query runs on the benchmark's graph store;
3. **score**: the answers are compared with gold (`hermes.evaluation`).

It writes `results/<benchmark>/<config>/<run id>/`:

| file | contents |
|---|---|
| `config.yaml` | the config as run |
| `predictions.jsonl` | per item: SPARQL, error, seconds, official and strict scores, whether gold reproduces, system extras (traces) |
| `metrics.json` | averages: `official`, `strict`, `reproducible`; error count; latency p50/p95 |
| `run.json` | model, LLM calls and cache hits, graph executions, git commit, wall time |

An interrupted run continues where it stopped with `--resume <run id>`.

## Config

```yaml
benchmark: ck26
system: uniqgen
params:            # constructor arguments of the system
  beam_width: 5
llm:
  provider: ollama # or openai (any OpenAI-compatible API)
  model: llama3.1
  cache: .cache/llm
languages: [en]    # optional subset
limit: 20          # optional
examples:          # optional few-shot pool from another benchmark
  benchmark: db25
  limit: 10
leak_gold: true    # ablations only: puts gold answers in context.answers
```

## Rules for comparable numbers

- Report the official metrics on all items for comparison with
  `results/peers/`, and the `strict` and `reproducible` numbers next to
  them. They answer different questions.
- Never tune on the test benchmark. CK25 and CK26 share 49 of 50 queries, so
  neither is a development set for the other.
- Runs with `leak_gold` or with `contaminated_examples > 0` are upper
  bounds, not results.
- Name the model. Peers used larger models (for example Claude, Qwen); a
  local 8B model is a lower bound for a method, not its ceiling.
