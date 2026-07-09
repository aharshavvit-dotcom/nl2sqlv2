"""Early stopping for neural model training.

Monitors a metric across epochs and signals when training should halt.
"""

from __future__ import annotations


class EarlyStopping:
    """Signals when training should stop due to lack of improvement.

    Parameters
    ----------
    patience:
        Number of epochs without improvement before stopping.
    metric_name:
        Name of the metric to monitor.
    mode:
        ``"max"`` (higher is better) or ``"min"`` (lower is better).
    min_delta:
        Minimum improvement to count as an actual improvement.
    regression_threshold:
        Maximum allowed drop from the best value before halting immediately.
    """

    def __init__(
        self,
        patience: int = 2,
        metric_name: str = "loss",
        mode: str = "min",
        min_delta: float = 0.0,
        regression_threshold: float = 0.50,
    ) -> None:
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.min_delta = min_delta
        self.regression_threshold = regression_threshold
        self._best: float | None = None
        self._counter: int = 0

    def step(self, metrics: dict[str, float]) -> bool:
        """Check whether training should stop.

        Returns ``True`` when training should be halted.
        """
        value = float(metrics.get(
            self.metric_name,
            metrics.get("overall_slot_accuracy",
            metrics.get("loss", 0.0)),
        ))
        if self._best is None:
            self._best = value
            self._counter = 0
            return False

        # Abort immediately if a significant regression is detected
        if self.mode == "max" and value < self._best - self.regression_threshold:
            print(f"Early Stopping: Significant regression detected! Metric {self.metric_name} fell from best {self._best:.4f} to {value:.4f} (limit: -{self.regression_threshold})")
            return True
        elif self.mode == "min" and value > self._best + self.regression_threshold:
            print(f"Early Stopping: Significant regression detected! Metric {self.metric_name} rose from best {self._best:.4f} to {value:.4f} (limit: +{self.regression_threshold})")
            return True

        improved = (
            (value > self._best + self.min_delta) if self.mode == "max"
            else (value < self._best - self.min_delta)
        )
        if improved:
            self._best = value
            self._counter = 0
            return False

        self._counter += 1
        return self._counter >= self.patience

    @property
    def counter(self) -> int:
        return self._counter

    @property
    def best_value(self) -> float | None:
        return self._best
