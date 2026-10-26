# Experiments

`hermes sweep configs/sweeps/hermes.yaml` runs Hermes, its ablations and two
baselines on CK26 and DB26; `hermes report --write` rewrites
`results/LEADERBOARD.md`. `bash experiments/run_all.sh` reproduces every run
in `results/`; pass config paths to run a subset.

- `planner_study.py`: plans every gold query of a benchmark as written and
  optimized, runs both, and checks their answers against the engine
  (`results/planner/`).
- `check_scorer.py`: checks that the scorer reproduces the official
  Text2SPARQL 2026 scores.
- `fit_selector.py`: fits the budgeted baseline's selector from its logged
  candidates.
- `modal_sweep.py`: runs a sweep on Modal GPUs.

## Where to run

Pick by what the models need; the commands are the same everywhere.

| Where | When | Setup |
|---|---|---|
| Your laptop | small local models (Ollama), short sweeps | `pip install -e .`, `ollama pull llama3.1` |
| A machine over SSH | long sweeps, a workstation or lab GPU box | `git clone`, `pip install -e .`, `hermes data fetch ...`, then `hermes sweep ...` inside `tmux` |
| Modal | Hugging Face models on a GPU, or sweeps that should not hold your machine | `pip install modal`, `modal setup` (once) |

**Hosted APIs.** Name the provider and model; the key comes from the
provider's usual variable.

```yaml
llm: {provider: openai, model: gpt-4o-mini}                  # OPENAI_API_KEY
llm: {provider: anthropic, model: claude-...}                # ANTHROPIC_API_KEY
llm: {provider: huggingface, model: Qwen/Qwen2.5-7B-Instruct} # HF_TOKEN
llm: {provider: together, model: ...}                        # TOGETHER_API_KEY
llm: {provider: modal, model: ..., base_url: <endpoint>/v1}  # MODAL_PROXY_TOKEN, Modal Shared Endpoints
```

Modal Shared Endpoints bill per token; a GPU you deploy yourself (below)
bills compute time, which the Starter plan's monthly credits cover.

**Any Hugging Face model on a Modal GPU.** Deploy a vLLM server once; it
scales to zero when idle, so it costs nothing between runs.

```bash
HERMES_MODEL=Qwen/Qwen2.5-7B-Instruct modal deploy deploy/modal_vllm.py
export HERMES_VLLM_URL=https://<printed url>/v1
# gated models such as Llama: HERMES_HF_SECRET=<your Modal secret with HF_TOKEN>
```

then use `llm: {provider: vllm, model: Qwen/Qwen2.5-7B-Instruct, base_url: ${HERMES_VLLM_URL}}`.

**A whole sweep on Modal.** The sweep runs in a Modal container; results
and the reply cache are kept in Modal Volumes, and rerunning resumes.

```bash
modal run experiments/modal_sweep.py --file configs/sweeps/baselines.yaml --only @qwen
modal volume get hermes-results / results/ --force
```

## Sweeps

`configs/sweeps/baselines.yaml` is the baselines' grid. A sweep lists benchmarks,
systems (an ablation is one more entry with a name), and models, plus shared
settings; every combination becomes a run named `system@model`.

```bash
hermes sweep configs/sweeps/baselines.yaml --list           # what would run
hermes sweep configs/sweeps/baselines.yaml --only @gpt      # a slice
hermes sweep configs/sweeps/baselines.yaml --limit 5        # a smoke test
```

Finished runs are skipped, unfinished runs resume, and a failing
configuration does not stop the others. To redo runs, for example after a fix, pass `--fresh`. It starts
new timestamped runs and leaves the old ones in place. Reports and figures
read the newest finished run. Never delete a run folder: commit finished
runs, so every number can be traced to its predictions.
`results/archive/` records runs whose folders were lost.

## Limits

Every run bounds its time and memory, so one bad query cannot hold a
machine:

```yaml
limits:
  query_seconds: 30    # any single SPARQL query; local queries run in a worker that is killed
  item_seconds: 300    # one question end to end; the item is recorded as an error
```

Endpoint results over 50 MB are refused, and the query cache never keeps
very large results.

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
| `run.json` | model, LLM calls and cache hits, input and output tokens, model and graph seconds, graph executions, limits, the commit the run started from, wall time |

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
