# Paper outline (working)

Working title: **Hermes: NL-to-SPARQL as planning over the SPARQL algebra**

## Claim

Text-to-query systems mostly write the whole query in one shot and repair it
afterwards. Query optimizers work differently: they apply one operator at a
time and cost each rewrite before taking it. We treat NL-to-SPARQL the same
way. The SPARQL algebra is the action space, and execution on the graph is the
feedback after every action. Existing methods become policies in that space;
UniQGen's Chase & Backchase, for example, is a beam search over `remove` and
`triple` actions.

## Research questions

1. Does operator-level execution feedback beat one-shot generation with
   repair, at equal model size and equal LLM-call budget?
2. Which feedback signals matter: cardinality, deltas, samples, or errors?
   (Ablate each in `Observation.summary`.)
3. Can a search policy (beam, MCTS) over the same actions match an LLM policy
   with fewer calls? UniQGen is the search baseline.
4. How far do results on the Text2SPARQL corporate KG transfer to DBpedia and
   Wikidata (DB26, QALD-9-plus, QALD-10)?

## Taxonomy (adapted from the NL2SQL survey, Liu et al., arXiv:2408.05109)

| stage | NL2SQL survey | SPARQL counterpart in Hermes |
|---|---|---|
| pre-processing | schema linking, DB content retrieval, example selection | `linking` (label index, exact labels), k-hop facts, `inspect`; `context_guided` stage |
| translation | encoding, decoding strategy, multi-step and agentic generation | `training-free` (one shot), `uniqgen` (constraint search), `algebra-agent` (operator policy) |
| post-processing | self-correction, consistency, execution-guided selection, N-best rerank | `guardrails`, parse repair, `Explorer.preview`, Chase & Backchase acceptance |
| evaluation | EX, EM, VES, robustness | official Text2SPARQL metrics, strict variant, reproducible subset, latency, calls |

## Baselines (existing methods, reimplemented in Hermes)

- Prompt-only (`training-free`), the Prompt-Only baseline of the UniQGen paper.
- UniQGen (`uniqgen`), with an LLM oracle and with a gold oracle (ablation).
- To add as systems: SPARQLGEN (one-shot with subgraph), KB-BINDER (draft and
  bind), DIN-SQL-style decomposition, DAIL-SQL-style example selection.
- Official Text2SPARQL'26 entries, from published scores (`results/peers`).

## Experiments

- Main table: CK26 and DB26, official metrics on all items, against the
  official leaderboard. Strict and reproducible columns next to them.
- Model scaling: the same systems on an 8B local model and on API models.
- Ablations: feedback signals; max steps; with and without `lookup` and
  `inspect`; preview-and-choose against commit-only.
- Cost: LLM calls, graph executions, p50/p95 latency (all in `run.json`).
- Error analysis: by Text2SPARQL feature tags (COUNT, GROUP, ORDER, ASK, ...).

## Threats to validity (see data/README.md)

The CK25/CK26 overlap, the official ASK and nDCG quirks, engine-dependent CK
gold sets, and drift of the live endpoints that QALD uses.
