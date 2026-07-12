"""Evaluation Framework — Gate 5.

Component/QueryIR/SQL metrics, generalization evaluator with complexity slices,
statistical reporter for multi-seed analysis, and test-set access guard.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ── Metric Definitions ───────────────────────────────────────────────

@dataclass
class MetricResult:
    """A single evaluation metric result."""
    name: str
    value: float
    count: int = 0
    confidence_interval: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "count": self.count,
            "confidence_interval": self.confidence_interval,
            "metadata": self.metadata,
        }


def exact_match_accuracy(
    predictions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    key: str = "source_sql",
) -> MetricResult:
    """Exact string match between predicted and gold SQL."""
    correct = sum(
        1 for p, g in zip(predictions, gold)
        if _normalize_sql(p.get(key, "")) == _normalize_sql(g.get(key, ""))
    )
    total = len(gold)
    return MetricResult(
        name="exact_match",
        value=correct / total if total > 0 else 0.0,
        count=total,
    )


def component_accuracy(
    predictions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
) -> dict[str, MetricResult]:
    """Per-component accuracy (tables, columns, predicates, aggregations)."""
    components = ["required_tables", "select_items", "group_by", "order_by"]
    results: dict[str, MetricResult] = {}

    for comp in components:
        correct = 0
        total = len(gold)
        for p, g in zip(predictions, gold):
            pred_val = _extract_component(p, comp)
            gold_val = _extract_component(g, comp)
            if pred_val == gold_val:
                correct += 1
        results[comp] = MetricResult(
            name=f"component_{comp}",
            value=correct / total if total > 0 else 0.0,
            count=total,
        )

    return results


def _extract_component(record: dict[str, Any], component: str) -> Any:
    """Extract a component from a prediction/gold record."""
    ir = record.get("query_ir", record)
    if isinstance(ir, dict):
        return json.dumps(ir.get(component, []), sort_keys=True, default=str)
    return ""


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.lower().strip().split())


# ── Complexity Slicing ───────────────────────────────────────────────

class ComplexitySlice:
    """Evaluates metrics on subsets defined by query complexity."""

    COMPLEXITY_LEVELS = {
        "simple": lambda r: _query_complexity(r) == "simple",
        "moderate": lambda r: _query_complexity(r) == "moderate",
        "complex": lambda r: _query_complexity(r) == "complex",
    }

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        gold: list[dict[str, Any]],
    ) -> dict[str, MetricResult]:
        """Evaluate exact match per complexity slice."""
        results: dict[str, MetricResult] = {}

        for slice_name, predicate in self.COMPLEXITY_LEVELS.items():
            indices = [i for i, g in enumerate(gold) if predicate(g)]
            if not indices:
                results[slice_name] = MetricResult(name=f"em_{slice_name}", value=0.0, count=0)
                continue
            sliced_pred = [predictions[i] for i in indices]
            sliced_gold = [gold[i] for i in indices]
            results[slice_name] = exact_match_accuracy(sliced_pred, sliced_gold)
            results[slice_name].name = f"em_{slice_name}"
            results[slice_name].metadata["slice"] = slice_name

        return results


def _query_complexity(record: dict[str, Any]) -> str:
    """Classify a query's complexity based on structural features."""
    ir = record.get("query_ir", record)
    if not isinstance(ir, dict):
        return "simple"

    score = 0
    if ir.get("joins"):
        score += len(ir["joins"])
    if ir.get("group_by"):
        score += 1
    if ir.get("having"):
        score += 1
    if ir.get("ctes"):
        score += 2
    if ir.get("set_operations"):
        score += 2
    if ir.get("where"):
        score += 1

    if score == 0:
        return "simple"
    elif score <= 2:
        return "moderate"
    return "complex"


# ── Multi-Seed Statistical Reporter ──────────────────────────────────

@dataclass
class SeedRunResult:
    """Result from a single seed run."""
    seed: int
    metrics: dict[str, float]


class StatisticalReporter:
    """Reports metrics across multiple seeds with confidence intervals.

    Uses bootstrap or t-distribution confidence intervals to report
    the stability of results across random seeds.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self._runs: list[SeedRunResult] = []

    def add_run(self, result: SeedRunResult) -> None:
        self._runs.append(result)

    def report(self) -> dict[str, Any]:
        """Generate a statistical summary across all seed runs."""
        if not self._runs:
            return {"error": "No runs to report"}

        all_metrics = set()
        for run in self._runs:
            all_metrics.update(run.metrics.keys())

        summaries: dict[str, dict[str, Any]] = {}
        for metric in sorted(all_metrics):
            values = [run.metrics.get(metric, 0.0) for run in self._runs]
            summaries[metric] = self._summarize(metric, values)

        return {
            "num_seeds": len(self._runs),
            "seeds": [r.seed for r in self._runs],
            "confidence_level": self.confidence_level,
            "metrics": summaries,
        }

    def _summarize(self, name: str, values: list[float]) -> dict[str, Any]:
        n = len(values)
        mean = statistics.mean(values)
        if n < 2:
            return {"mean": mean, "std": 0.0, "ci": (mean, mean), "n": n}

        std = statistics.stdev(values)
        # t-distribution CI
        t_val = _t_critical(n - 1, self.confidence_level)
        margin = t_val * std / math.sqrt(n)
        ci = (mean - margin, mean + margin)

        return {
            "mean": round(mean, 6),
            "std": round(std, 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "ci": (round(ci[0], 6), round(ci[1], 6)),
            "n": n,
        }


def _t_critical(df: int, confidence: float) -> float:
    """Approximate t-critical value using normal approximation for large df."""
    # Exact values for small df
    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
    }
    if df in t_table:
        return t_table[df]
    if df > 30:
        return 1.96  # z-approximation
    # Linear interpolation
    keys = sorted(t_table.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= df < keys[i + 1]:
            frac = (df - keys[i]) / (keys[i + 1] - keys[i])
            return t_table[keys[i]] * (1 - frac) + t_table[keys[i + 1]] * frac
    return 1.96


# ── Test-Set Access Guard ────────────────────────────────────────────

class FrozenSplitGuard:
    """Prevents accidental access to the frozen test set during training.

    Usage:
        guard = TestSetAccessGuard("frozen_semantic_test")
        guard.check_access("train")  # OK
        guard.check_access("frozen_semantic_test")  # Raises
    """

    def __init__(
        self,
        frozen_splits: list[str] | None = None,
        max_access_count: int = 0,
    ) -> None:
        self._frozen = set(frozen_splits or ["frozen_semantic_test", "unseen_database_test"])
        self._max_access = max_access_count
        self._access_log: dict[str, int] = defaultdict(int)

    def check_access(self, split_name: str, purpose: str = "") -> None:
        """Check if access to a split is allowed."""
        if split_name in self._frozen:
            self._access_log[split_name] += 1
            if self._access_log[split_name] > self._max_access:
                raise FrozenSplitAccessError(
                    f"Access to frozen split {split_name!r} is not allowed "
                    f"(purpose: {purpose or 'unspecified'}). "
                    f"Access count: {self._access_log[split_name]}."
                )

    def allow_evaluation_access(self, split_name: str) -> None:
        """Explicitly allow a one-time evaluation access."""
        self._max_access = max(self._max_access, self._access_log.get(split_name, 0) + 1)

    @property
    def access_log(self) -> dict[str, int]:
        return dict(self._access_log)

    def is_frozen(self, split_name: str) -> bool:
        return split_name in self._frozen


class FrozenSplitAccessError(RuntimeError):
    pass


# ── Confidence Separation ────────────────────────────────────────────

@dataclass
class ConfidenceScores:
    """Separated confidence scores for a prediction."""
    model_confidence: float      # How confident the model is in its output
    capability_coverage: float   # What fraction of required capabilities are supported
    safety_score: float         # Safety assessment (1.0 = fully safe)

    @property
    def overall(self) -> float:
        """Combined confidence: product of all three scores."""
        return self.model_confidence * self.capability_coverage * self.safety_score

    @property
    def is_actionable(self) -> bool:
        """Whether the prediction should be acted on."""
        return (
            self.model_confidence >= 0.5
            and self.capability_coverage >= 0.8
            and self.safety_score >= 0.9
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "model_confidence": self.model_confidence,
            "capability_coverage": self.capability_coverage,
            "safety_score": self.safety_score,
            "overall": self.overall,
            "is_actionable": self.is_actionable,
        }


__all__ = [
    "CheckpointMetrics",
    "ComplexitySlice",
    "ConfidenceScores",
    "MetricResult",
    "SeedRunResult",
    "StatisticalReporter",
    "FrozenSplitAccessError",
    "FrozenSplitGuard",
    "component_accuracy",
    "exact_match_accuracy",
]
