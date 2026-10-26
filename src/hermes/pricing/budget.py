"""Budgets: price every model call and graph query before it is spent.

A `Price` is what an action costs: model calls, input and output tokens,
dollars when the model has a list price, and graph work (queries sent, rows
fetched). Every action is priced before it runs (`llm_price` from the prompt's
length, `plan_price` from the planner's estimates), checked against the
`Budget`, and recorded with its actual cost in a `Spend`, so a run can report
what each stage was expected to cost and what it did cost.

`BudgetedLLM` decorates any model client: it prices the call, refuses it with
`BudgetExceeded` when it would break the budget, and meters what it used.
"""

from dataclasses import dataclass, field, fields

from hermes.llm import LLM, SYSTEM
from hermes.pricing.prices import prices

CHARS_PER_TOKEN = 4.0  # a rough rate for English and SPARQL
EXPECTED_OUTPUT = 400  # tokens a reply is priced at, before it is seen


@dataclass
class Price:
    llm_calls: float = 0
    input_tokens: float = 0
    output_tokens: float = 0
    usd: float = 0.0  # 0 when the model has no list price
    queries: float = 0
    rows: float = 0

    def __add__(self, other):
        return Price(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))

    @property
    def tokens(self):
        return self.input_tokens + self.output_tokens

    def to_dict(self):
        return {f.name: round(getattr(self, f.name), 6 if f.name == "usd" else 1)
                for f in fields(self)}


@dataclass
class Budget:
    """Caps on a run; None means no cap."""

    llm_calls: int = None
    tokens: int = None
    usd: float = None
    queries: int = None
    rows: int = None

    def refuses(self, spent, next_price):
        """Why `next_price` would break the budget after `spent`, or None."""
        total = spent + next_price
        for name, used in (("llm_calls", total.llm_calls), ("tokens", total.tokens),
                           ("usd", total.usd), ("queries", total.queries), ("rows", total.rows)):
            cap = getattr(self, name)
            if cap is not None and used > cap:
                return f"{name.replace('_', ' ')} would reach {used:,.4g}, over the cap of {cap:,}"
        return None


class BudgetExceeded(RuntimeError):
    def __init__(self, stage, reason, price):
        super().__init__(f"{stage}: {reason}")
        self.stage, self.reason, self.price = stage, reason, price


@dataclass
class Entry:
    stage: str
    estimated: Price
    actual: Price


@dataclass
class Spend:
    """What a run spent, stage by stage, against what each stage was priced at."""

    budget: Budget = field(default_factory=Budget)
    entries: list = field(default_factory=list)

    @property
    def total(self):
        total = Price()
        for entry in self.entries:
            total = total + entry.actual
        return total

    @property
    def estimated(self):
        total = Price()
        for entry in self.entries:
            total = total + entry.estimated
        return total

    def check(self, stage, price):
        reason = self.budget.refuses(self.total, price)
        if reason:
            raise BudgetExceeded(stage, reason, price)

    def affordable(self, price):
        return self.budget.refuses(self.total, price) is None

    def record(self, stage, estimated, actual):
        self.entries.append(Entry(stage, estimated, actual))

    def to_dict(self):
        return {"total": self.total.to_dict(), "estimated": self.estimated.to_dict(),
                "stages": [{"stage": e.stage, "estimated": e.estimated.to_dict(),
                            "actual": e.actual.to_dict()} for e in self.entries]}


def llm_price(model, prompt, system=SYSTEM, output_tokens=EXPECTED_OUTPUT):
    """A call's price before it is made: tokens from the prompt's length, dollars from the list."""
    input_tokens = (len(prompt) + len(system)) / CHARS_PER_TOKEN
    return Price(1, input_tokens, output_tokens, _usd(model, input_tokens, output_tokens))


def plan_price(optimized):
    """A plan's price before it runs: the queries and rows the planner expects it to fetch."""
    from hermes.planner.estimate import fetched

    queries, rows = fetched(optimized.query.root, optimized.estimator)
    return Price(queries=queries, rows=rows)


def _usd(model, input_tokens, output_tokens):
    price = prices().get((model or "").split("/")[-1].lower())
    if price is None:
        return 0.0
    return (input_tokens * price[0] + output_tokens * price[1]) / 1e6


class BudgetedLLM(LLM):
    """Decorator: price each call, refuse it over budget, and record what it used."""

    def __init__(self, llm, spend, stage="model"):
        self.llm = llm
        self.spend = spend
        self.stage = stage

    @property
    def provider(self):
        return self.llm.provider

    @property
    def model(self):
        return self.llm.model

    @property
    def usage(self):
        return self.llm.usage

    def price(self, prompt, system=SYSTEM):
        return llm_price(self.model, prompt, system)

    def complete(self, prompt, system=SYSTEM):
        estimated = self.price(prompt, system)
        self.spend.check(self.stage, estimated)
        before = dict(self.llm.usage)
        reply = self.llm.complete(prompt, system)
        after = self.llm.usage
        used_in = after["input_tokens"] - before["input_tokens"]
        used_out = after["output_tokens"] - before["output_tokens"]
        if not used_in:  # replayed from a cache, or a client that does not count: estimate
            used_in, used_out = estimated.input_tokens, len(reply) / CHARS_PER_TOKEN
        actual = Price(1, used_in, used_out, _usd(self.model, used_in, used_out))
        self.spend.record(self.stage, estimated, actual)
        return reply
