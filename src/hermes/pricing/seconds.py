"""Seconds per unit of plan cost, fitted to the executions a store actually ran."""

import threading


class SecondsModel:
    """seconds = a + b * C_out, fitted by least squares to observed executions."""

    def __init__(self, intercept=0.005, slope=1e-6, minimum=3):
        self.prior = (intercept, slope)
        self.minimum = minimum
        self.points = []
        self._lock = threading.Lock()

    def observe(self, cost, seconds):
        with self._lock:
            self.points.append((float(cost), float(seconds)))

    def coefficients(self):
        with self._lock:
            points = list(self.points)
        if len(points) < self.minimum:
            return self.prior
        n = len(points)
        mean_x = sum(x for x, _ in points) / n
        mean_y = sum(y for _, y in points) / n
        spread = sum((x - mean_x) ** 2 for x, _ in points)
        if spread == 0:
            return (mean_y, 0.0)
        slope = max(0.0, sum((x - mean_x) * (y - mean_y) for x, y in points) / spread)
        return (max(0.0, mean_y - slope * mean_x), slope)

    def predict(self, cost):
        intercept, slope = self.coefficients()
        return intercept + slope * cost

    def nominal(self, cost):
        """Seconds at the prior rate: depends only on the plan, so choices that use it
        are the same on every machine and every run."""
        intercept, slope = self.prior
        return intercept + slope * cost
