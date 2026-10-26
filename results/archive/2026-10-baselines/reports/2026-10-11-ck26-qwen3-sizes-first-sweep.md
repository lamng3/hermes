# CK26 · Qwen3 sizes · first sweep (superseded)

Run on 2026-10-11 and superseded by the rerun on commit 38f7cc4. The run folders were deleted; the numbers below were recorded before that. The raw model replies are in the Modal volume `hermes-llm-cache`.

## Config

| | |
|---|---|
| Sweep | `configs/sweeps/qwen3-sizes.yaml` |
| Code | d5d2bca – 6971a0c |
| Benchmark | CK26 (100 items, en + de) |
| Models | Qwen/Qwen3-4B, -8B, -14B; thinking off |
| Serving | vLLM 0.29.0, Modal L40S, bf16 |
| Concurrency | 16 |
| Max output tokens | none |
| Limits | 30 s per query, 300 s per item |
| training-free | defaults |
| uniqgen | beam_width 5, max_depth 5 |
| algebra-agent | max_steps 15 |
| prover | max_steps 15 |

## Accuracy

| Run | P | R | F1 | F1-nDCG | Strict | Repro. | Exact | F1 en | F1 de |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| training-free@qwen3-4b | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0.001 | 0.000 |
| training-free@qwen3-8b | 0.093 | 0.068 | 0.067 | 0.068 | 0.067 | 0.078 | 5 | 0.099 | 0.035 |
| training-free@qwen3-14b | 0.111 | 0.091 | 0.089 | 0.090 | 0.089 | 0.103 | 4 | 0.091 | 0.086 |
| uniqgen@qwen3-4b | 0.084 | 0.154 | 0.078 | 0.078 | 0.078 | 0.082 | 5 | 0.124 | 0.031 |
| uniqgen@qwen3-8b | 0.142 | 0.232 | 0.114 | 0.114 | 0.114 | 0.125 | 8 | 0.160 | 0.067 |
| uniqgen@qwen3-14b | 0.201 | 0.404 | 0.194 | 0.193 | 0.194 | 0.213 | 11 | 0.228 | 0.159 |
| algebra-agent@qwen3-4b | 0.027 | 0.079 | 0.028 | 0.028 | 0.028 | 0.027 | 2 | 0.016 | 0.040 |
| algebra-agent@qwen3-8b | 0.183 | 0.349 | 0.192 | 0.192 | 0.192 | 0.198 | 12 | 0.179 | 0.205 |
| algebra-agent@qwen3-14b | 0.258 | 0.274 | 0.236 | 0.237 | 0.236 | 0.254 | 17 | 0.279 | 0.194 |
| prover@qwen3-4b | 0.077 | 0.220 | 0.083 | 0.085 | 0.083 | 0.094 | 5 | 0.079 | 0.087 |
| prover@qwen3-8b | 0.152 | 0.395 | 0.151 | 0.151 | 0.151 | 0.165 | 7 | 0.156 | 0.145 |
| prover@qwen3-14b | 0.270 | 0.421 | 0.263 | 0.263 | 0.263 | 0.281 | 15 | 0.279 | 0.247 |

These are the official Text2SPARQL averages. Strict is the strict-scoring variant, Repro. is the subset whose gold queries reproduce on Oxigraph, and Exact counts items with F1 = 1.

## Cost

| Run | Wall (min) | p50 (s) | p95 (s) | LLM calls | Input tok. | Output tok. | Graph exec. | Rejections |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| training-free@qwen3-4b | 2.3 | 2.4 | 4.9 | 97 | 18,445 | 7,567 | 89 | – |
| training-free@qwen3-8b | 2.2 | 0.9 | 3.5 | 8 | 1,489 | 478 | 174 | – |
| training-free@qwen3-14b | 2.3 | 2.9 | 9.4 | 104 | 70,531 | 7,167 | 147 | – |
| uniqgen@qwen3-4b | 2.5 | 3.3 | 120.1 | 158 | 364,012 | 10,518 | 863 | – |
| uniqgen@qwen3-8b | 4.5 | 5.7 | 124.5 | 171 | 348,397 | 7,920 | 658 | – |
| uniqgen@qwen3-14b | 3.0 | 10.2 | 125.4 | 192 | 384,981 | 9,663 | 1,241 | – |
| algebra-agent@qwen3-4b | 3.5 | 7.5 | 122.1 | 1,208 | 2,127,834 | 19,185 | 379 | – |
| algebra-agent@qwen3-8b | 1.5 | 13.1 | 20.7 | 1,312 | 2,688,222 | 28,608 | 886 | – |
| algebra-agent@qwen3-14b | 2.4 | 21.0 | 33.6 | 1,408 | 2,884,105 | 27,307 | 879 | – |
| prover@qwen3-4b | 1.1 | 10.1 | 12.9 | 1,454 | 2,868,710 | 34,647 | 539 | 44 |
| prover@qwen3-8b | 1.8 | 16.2 | 21.4 | 1,300 | 2,962,156 | 27,925 | 896 | 54 |
| prover@qwen3-14b | 2.5 | 24.0 | 31.7 | 1,257 | 2,842,681 | 24,196 | 940 | 32 |

LLM calls and tokens leave out replies served from the cache. That is why training-free@qwen3-8b, a rerun, shows only 8 calls. Rejections are plans the verifier refused.

## Errors (items out of 100)

| Run | Total | ValueError | QueryError | ReadTimeout | Other |
|---|--:|--:|--:|--:|---|
| training-free@qwen3-4b | 80 | 79 | – | 1 | – |
| training-free@qwen3-8b | 16 | 5 | 9 | 2 | – |
| training-free@qwen3-14b | 15 | 2 | 12 | 1 | – |
| uniqgen@qwen3-4b | 36 | 24 | 1 | 10 | IndexError 1 |
| uniqgen@qwen3-8b | 32 | 8 | 5 | 17 | IndexError 2 |
| uniqgen@qwen3-14b | 15 | 2 | 7 | 6 | – |
| algebra-agent@qwen3-4b | 73 | 57 | – | 16 | – |
| algebra-agent@qwen3-8b | 15 | 15 | – | – | – |
| algebra-agent@qwen3-14b | 26 | 25 | – | – | HTTP 500: 1 |
| prover@qwen3-4b | 35 | 33 | – | – | TypeError 1, AttributeError 1 |
| prover@qwen3-8b | 4 | 3 | – | – | HTTP 500: 1 |
| prover@qwen3-14b | 5 | 5 | – | – | – |

The 38f7cc4 changes affect these errors:

| Error | Change |
|---|---|
| IndexError | UniQGen with no bound variable now raises ValueError |
| TypeError, AttributeError | a malformed action now counts as an invalid step |
| HTTP 500 | now retried |
| ReadTimeout | replies now capped at 2,048 tokens |
