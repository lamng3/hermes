# CK26, Qwen3 4B/8B/14B: first sweep (superseded)

This is the first run of `configs/sweeps/qwen3-sizes.yaml` on Modal, on 2026-10-11.
It was superseded by a rerun after fixes in 38f7cc4. Its run folders
(`predictions.jsonl`, `metrics.json`, `run.json`) were deleted locally and from the
`hermes-results` volume before that rerun, and no copy of them exists. This record keeps
what survives for audit.

- **Code:** commits d5d2bca to 6971a0c. Runs started after 6971a0c (11:45 local) have
  the rdflib parse lock; the 8B training-free and prover runs were rerun under it.
- **Serving:** vLLM 0.29.0 on a Modal L40S in bf16, with thinking off, concurrency 16,
  and limits of 30 s per query and 300 s per item. Replies had no length cap.
- **Raw model replies:** these are still in the `hermes-llm-cache` Modal volume, under
  cache keys without generation settings. Replaying commit 6971a0c against that cache
  rebuilds the per-item predictions for every item that got its replies.
- **Why it was superseded:** UniQGen raised `IndexError` when no variable was bound.
  Malformed actions raised `TypeError`/`AttributeError` instead of counting as invalid
  steps. vLLM 500s were not retried. With no output cap, some requests hit
  `ReadTimeout`. Qwen3-4B training-free mostly omitted `SELECT` (`ValueError`), which is
  a model failure, not a bug.

## Per-run summary (CK26, 100 items)

The table was copied from the analysis printed at 2026-10-11T17:48Z, before the deletion.

- **Columns:**
  - P / R / F1 / Fn: official averages (Fn is `set_F_ndcg`).
  - strict: the strict variant. repro: the reproducible subset. exact: items with F1 = 1.
  - err: items with an error.
  - wall: run wall time. p50 / p95: per-item seconds.
  - calls / in / out: LLM calls and input/output tokens.
  - gexec: graph executions. rej: verifier rejections.

| run | P | R | F1 | Fn | strict | repro | exact | err | errors | wall | p50 | p95 | calls | in | out | gexec | rej | F1 en / de |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| training-free@qwen3-4b | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 80 | ValueError 79, ReadTimeout 1 | 2.3m | 2.4 | 4.9 | 97 | 18445 | 7567 | 89 | - | 0.001 / 0.000 |
| training-free@qwen3-8b | 0.093 | 0.068 | 0.067 | 0.068 | 0.067 | 0.078 | 5 | 16 | ValueError 5, QueryError 9, ReadTimeout 2 | 2.2m | 0.9 | 3.5 | 8 | 1489 | 478 | 174 | - | 0.099 / 0.035 |
| training-free@qwen3-14b | 0.111 | 0.091 | 0.089 | 0.090 | 0.089 | 0.103 | 4 | 15 | ValueError 2, QueryError 12, ReadTimeout 1 | 2.3m | 2.9 | 9.4 | 104 | 70531 | 7167 | 147 | - | 0.091 / 0.086 |
| uniqgen@qwen3-4b | 0.084 | 0.154 | 0.078 | 0.078 | 0.078 | 0.082 | 5 | 36 | ValueError 24, IndexError 1, QueryError 1, ReadTimeout 10 | 2.5m | 3.3 | 120.1 | 158 | 364012 | 10518 | 863 | - | 0.124 / 0.031 |
| uniqgen@qwen3-8b | 0.142 | 0.232 | 0.114 | 0.114 | 0.114 | 0.125 | 8 | 32 | ValueError 8, IndexError 2, QueryError 5, ReadTimeout 17 | 4.5m | 5.7 | 124.5 | 171 | 348397 | 7920 | 658 | - | 0.160 / 0.067 |
| uniqgen@qwen3-14b | 0.201 | 0.404 | 0.194 | 0.193 | 0.194 | 0.213 | 11 | 15 | ValueError 2, QueryError 7, ReadTimeout 6 | 3.0m | 10.2 | 125.4 | 192 | 384981 | 9663 | 1241 | - | 0.228 / 0.159 |
| algebra-agent@qwen3-4b | 0.027 | 0.079 | 0.028 | 0.028 | 0.028 | 0.027 | 2 | 73 | ReadTimeout 16, ValueError 57 | 3.5m | 7.5 | 122.1 | 1208 | 2127834 | 19185 | 379 | - | 0.016 / 0.040 |
| algebra-agent@qwen3-8b | 0.183 | 0.349 | 0.192 | 0.192 | 0.192 | 0.198 | 12 | 15 | ValueError 15 | 1.5m | 13.1 | 20.7 | 1312 | 2688222 | 28608 | 886 | - | 0.179 / 0.205 |
| algebra-agent@qwen3-14b | 0.258 | 0.274 | 0.236 | 0.237 | 0.236 | 0.254 | 17 | 26 | ValueError 25, HTTPStatusError 1 | 2.4m | 21.0 | 33.6 | 1408 | 2884105 | 27307 | 879 | - | 0.279 / 0.194 |
| prover@qwen3-4b | 0.077 | 0.220 | 0.083 | 0.085 | 0.083 | 0.094 | 5 | 35 | TypeError 1, ValueError 33, AttributeError 1 | 1.1m | 10.1 | 12.9 | 1454 | 2868710 | 34647 | 539 | 44 | 0.079 / 0.087 |
| prover@qwen3-8b | 0.152 | 0.395 | 0.151 | 0.151 | 0.151 | 0.165 | 7 | 4 | ValueError 3, HTTPStatusError 1 | 1.8m | 16.2 | 21.4 | 1300 | 2962156 | 27925 | 896 | 54 | 0.156 / 0.145 |
| prover@qwen3-14b | 0.270 | 0.421 | 0.263 | 0.263 | 0.263 | 0.281 | 15 | 5 | ValueError 5 | 2.5m | 24.0 | 31.7 | 1257 | 2842681 | 24196 | 940 | 32 | 0.279 / 0.247 |

These are the items that crashed on the bugs fixed in 38f7cc4:

    uniqgen@qwen3-4b         ck26:4-en   IndexError: list index out of range
    uniqgen@qwen3-8b         ck26:4-en   IndexError: list index out of range
    uniqgen@qwen3-8b         ck26:7-en   IndexError: list index out of range
    prover@qwen3-4b          ck26:3-de   TypeError: argument of type 'NoneType' is not iterable
    prover@qwen3-4b          ck26:29-en  AttributeError: 'list' object has no attribute 'startswith'
    algebra-agent@qwen3-14b  ck26:47-de  HTTPStatusError: 500 Internal Server Error
    prover@qwen3-8b          ck26:48-de  HTTPStatusError: 500 Internal Server Error
