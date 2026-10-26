"""What a run costs: list prices for hosted models, GPU time for local ones.

A run already counts its model calls, input and output tokens, and the
seconds spent in the model (run.json). `run_cost` turns those into money when
the model has a price, and says how much GPU time it took when it runs
locally (vLLM, Ollama), where there is no per-token price to charge. GPU time
is the run's wall time: model seconds add up the latency of calls made in
parallel, so they overstate how long the GPU was busy.

Prices are dollars per million tokens, input then output. Override or extend
them with HERMES_PRICES, a JSON object of the same shape
(`{"my-model": [0.5, 1.5]}`). Set HERMES_GPU_HOUR_USD to price local runs by
GPU time instead.
"""

import json
import os

PRICES = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
LOCAL_PROVIDERS = {"vllm", "ollama"}


def prices():
    table = dict(PRICES)
    table.update({k: tuple(v) for k, v in json.loads(os.environ.get("HERMES_PRICES", "{}")).items()})
    return table


def run_cost(run):
    """Cost of one run from its run.json fields: dollars if priced, and per-item figures."""
    items = run.get("items") or 0
    calls = run.get("llm_model_calls") or run.get("llm_calls") or 0
    tokens_in = run.get("input_tokens") or 0
    tokens_out = run.get("output_tokens") or 0
    seconds = run.get("llm_seconds") or 0.0
    model = (run.get("model") or "").split("/")[-1].lower()
    price = prices().get(model)
    usd, basis = None, "unpriced"
    if price is not None:
        usd, basis = (tokens_in * price[0] + tokens_out * price[1]) / 1e6, "list price"
    gpu_hours = None
    if price is None and run.get("provider") in LOCAL_PROVIDERS:
        gpu_hours = (run.get("wall_seconds") or 0.0) / 3600
        rate = os.environ.get("HERMES_GPU_HOUR_USD")
        if rate:
            usd, basis = gpu_hours * float(rate), f"GPU time at ${float(rate):g}/h"
        else:
            basis = f"local, {gpu_hours * 60:.1f} GPU-min"
    per = (lambda x: x / items) if items else (lambda x: 0.0)
    return {
        "cost_usd": None if usd is None else round(usd, 4),
        "cost_basis": basis,
        "gpu_hours": None if gpu_hours is None else round(gpu_hours, 4),
        "calls_per_item": round(per(calls), 2),
        "input_tokens_per_item": round(per(tokens_in)),
        "output_tokens_per_item": round(per(tokens_out)),
        "llm_seconds_per_item": round(per(seconds), 2),
        "cost_usd_per_item": None if usd is None or not items else round(usd / items, 5),
    }
