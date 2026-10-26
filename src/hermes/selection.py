"""Selection: a learned estimate that a candidate query answers its question.

Every candidate is described by a few features, known before it runs (how
well its phrases bound, what the verifier said, what pricing expects it to
return) or after (whether it returned anything, and as many rows as the goal
asks for). A logistic model turns them into a probability. The default weights
are set by hand; `Scorer.fit` learns them from logged candidates whose answers
were scored against gold, as experiments/fit_selector.py does.
"""

import json
import math
from pathlib import Path

FEATURES = (
    "binding",           # mean log score of the slot bindings (0 is a perfect match)
    "warnings",          # verifier warnings, such as a variable with two unrelated classes
    "open_goals",        # goals the plan does not meet
    "estimated_empty",   # pricing expects no rows
    "estimated_excess",  # one answer is asked for, pricing expects many
    "executed",          # the candidate has run
    "empty",             # it ran and returned nothing
    "excess",            # one answer was asked for and it returned many
)
DEFAULT = {
    "bias": 1.0, "binding": 1.5, "warnings": -0.5, "open_goals": -1.0,
    "estimated_empty": -1.0, "estimated_excess": -0.3,
    "executed": 0.8, "empty": -3.0, "excess": -0.8,
}


class Scorer:
    def __init__(self, weights=None):
        self.weights = {**DEFAULT, **(weights or {})}

    @classmethod
    def load(cls, weights):
        """From a dict, a JSON file path, or None for the defaults."""
        if isinstance(weights, (str, Path)):
            weights = json.loads(Path(weights).read_text(encoding="utf-8"))
        return cls(weights)

    def logit(self, features):
        return self.weights["bias"] + sum(
            self.weights.get(name, 0.0) * float(features.get(name, 0.0)) for name in FEATURES)

    def probability(self, features):
        return _sigmoid(self.logit(features))

    def fit(self, examples, epochs=500, rate=0.1, l2=0.01):
        """Logistic regression by gradient descent on (features, label) pairs."""
        examples = list(examples)
        if not examples:
            return self
        names = ("bias", *FEATURES)
        weights = {name: self.weights.get(name, 0.0) for name in names}
        for _epoch in range(epochs):
            gradient = dict.fromkeys(names, 0.0)
            for features, label in examples:
                row = {"bias": 1.0, **{n: float(features.get(n, 0.0)) for n in FEATURES}}
                error = _sigmoid(sum(weights[n] * row[n] for n in names)) - float(label)
                for name in names:
                    gradient[name] += error * row[name]
            for name in names:
                penalty = 0.0 if name == "bias" else l2 * weights[name]
                weights[name] -= rate * (gradient[name] / len(examples) + penalty)
        self.weights = weights
        return self

    def to_dict(self):
        return {name: round(value, 4) for name, value in self.weights.items()}


def _sigmoid(value):
    if value < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))
