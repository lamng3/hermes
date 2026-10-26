"""Per-item limits and metering that work in worker threads.

A run may answer several questions at once, so that a model server such as
vLLM can batch their requests. Two things are kept per thread:

- a deadline: `check()` raises ItemTimeout once the current item's time
  budget is spent. Model clients and graph stores call it before every
  request, so an item stops at its next step;
- a meter: model clients and stores add their calls, tokens, seconds, and
  executions to the current item's meter, so per-item cost stays exact when
  items overlap.
"""

import contextlib
import threading
import time

_local = threading.local()


class ItemTimeout(Exception):
    """An item ran past its time limit."""


@contextlib.contextmanager
def meter():
    """Count this thread's model and graph usage; yields the running totals."""
    _local.meter = {
        "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "llm_seconds": 0.0,
        "graph_executions": 0, "graph_seconds": 0.0,
    }
    try:
        yield _local.meter
    finally:
        _local.meter = None


@contextlib.contextmanager
def deadline(seconds):
    """Give this thread a time budget; check() raises once it is spent."""
    _local.deadline = time.monotonic() + seconds if seconds else None
    try:
        yield
    finally:
        _local.deadline = None


def check():
    deadline = getattr(_local, "deadline", None)
    if deadline is not None and time.monotonic() > deadline:
        raise ItemTimeout()


def remaining(default):
    """Seconds left for the current item, capped at `default`."""
    deadline = getattr(_local, "deadline", None)
    if deadline is None:
        return default
    return max(0.1, min(default, deadline - time.monotonic()))


def add(**amounts):
    meter = getattr(_local, "meter", None)
    if meter is not None:
        for key, value in amounts.items():
            meter[key] += value
