"""Improvement Tracker — tracks metrics across self-training iterations.

Records per-iteration metrics, detects convergence, and produces an
improvement report summarising the full self-improvement run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# Key metrics that the tracker monitors
TRACKED_METRICS = [
    "intent_accuracy",
    "base_table_accuracy",
    "metric_accuracy",
    "dimension_accuracy",
    "filter_accuracy",
    "date_filter_accuracy",
    "overall_slot_accuracy",
    "sql_validation_rate",
    "exact_match_rate",
    "match_score_mean",
]


@dataclass
class ImprovementReport:
    """Final report summarising a complete self-improvement run."""

    iterations: list[dict[str, Any]] = field(default_factory=list)
    best_iteration: int = 0
    best_metrics: dict[str, float] = field(default_factory=dict)
    total_improvement: dict[str, float] = field(default_factory=dict)
    converged: bool = False
    convergence_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ImprovementTracker:
    """Tracks per-iteration metrics and detects convergence."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._history_path = self.output_dir / "improvement_history.json"
        self._history: list[dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_iteration(self, iteration: int, metrics: dict[str, Any]) -> None:
        """Record metrics for a single self-training iteration."""

        entry = {"iteration": iteration, **{k: float(metrics.get(k, 0.0)) for k in TRACKED_METRICS}}
        # Also store any extra metrics the caller provides
        for key, value in metrics.items():
            if key not in entry:
                try:
                    entry[key] = float(value)
                except (TypeError, ValueError):
                    entry[key] = value
        self._history.append(entry)
        self._save()

    def get_improvement(self, metric_name: str) -> float:
        """Return percentage improvement from iteration 0 to the latest.

        Returns 0.0 if there are fewer than 2 iterations or the baseline
        value is 0.
        """

        if len(self._history) < 2:
            return 0.0
        baseline = float(self._history[0].get(metric_name, 0.0))
        latest = float(self._history[-1].get(metric_name, 0.0))
        if baseline == 0.0:
            return latest  # can't compute %
        return (latest - baseline) / abs(baseline)

    def should_stop(self, min_improvement: float = 0.005) -> bool:
        """True if the latest iteration didn't improve enough to continue.

        Compares the primary metric (``overall_slot_accuracy``) between the
        last two iterations.  Returns False if there are fewer than 2 entries.
        """

        if len(self._history) < 2:
            return False
        prev = float(self._history[-2].get("overall_slot_accuracy", 0.0))
        curr = float(self._history[-1].get("overall_slot_accuracy", 0.0))
        improvement = curr - prev
        return improvement < min_improvement

    def generate_report(self) -> ImprovementReport:
        """Produce the final improvement report."""

        if not self._history:
            return ImprovementReport(convergence_reason="no_iterations_recorded")

        # Find the best iteration by overall_slot_accuracy
        best_idx = 0
        best_score = -1.0
        for idx, entry in enumerate(self._history):
            score = float(entry.get("overall_slot_accuracy", 0.0))
            if score > best_score:
                best_score = score
                best_idx = idx

        best_metrics = {
            k: float(self._history[best_idx].get(k, 0.0))
            for k in TRACKED_METRICS
        }

        # Compute total improvement (last vs first)
        total_improvement: dict[str, float] = {}
        if len(self._history) >= 2:
            for k in TRACKED_METRICS:
                baseline = float(self._history[0].get(k, 0.0))
                latest = float(self._history[-1].get(k, 0.0))
                total_improvement[k] = round(latest - baseline, 6)

        # Determine convergence
        converged = self.should_stop()
        if converged:
            convergence_reason = "improvement_below_threshold"
        elif len(self._history) >= 2:
            convergence_reason = "completed_all_iterations"
        else:
            convergence_reason = "single_iteration"

        return ImprovementReport(
            iterations=list(self._history),
            best_iteration=best_idx,
            best_metrics=best_metrics,
            total_improvement=total_improvement,
            converged=converged,
            convergence_reason=convergence_reason,
        )

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    @property
    def iteration_count(self) -> int:
        return len(self._history)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self._history_path.write_text(
            json.dumps(self._history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load(self) -> None:
        if self._history_path.exists():
            try:
                self._history = json.loads(self._history_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._history = []
