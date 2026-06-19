from __future__ import annotations

import random
from typing import Any

from .model_selector import _hard_blockers


class PromotionPolicy:
    def can_promote(
        self,
        challenger_metrics: dict[str, Any],
        champion_metrics: dict[str, Any] | None,
        thresholds: dict[str, Any],
        bootstrap_iterations: int = 1000,
    ) -> dict[str, Any]:
        minimums = thresholds.get("minimums") or thresholds
        blocking = _hard_blockers(challenger_metrics, minimums)
        champion_metrics = champion_metrics or {}
        min_improvement = float(minimums.get("model_promotion_min_improvement", 0.0))
        statistical_report = self._bootstrap_report(
            challenger_metrics.get("per_example") or [],
            champion_metrics.get("per_example") or [],
            iterations=bootstrap_iterations,
        ) if champion_metrics else {"bootstrap_iterations": bootstrap_iterations, "metrics": {}, "paired_examples": 0}
        if champion_metrics:
            statistically_checked = bool(statistical_report.get("metrics"))
            regressions = [name for name, item in statistical_report.get("metrics", {}).items() if item.get("regression_detected")]
            blocking.extend(f"{name}_statistical_regression" for name in regressions)
            if not statistically_checked and challenger_metrics.get("simple_query_pass_rate", 0.0) < champion_metrics.get("simple_query_pass_rate", 0.0):
                blocking.append("simple_query_pass_rate_regression")
            if not statistically_checked and challenger_metrics.get("unnecessary_join_rate", 0.0) > champion_metrics.get("unnecessary_join_rate", 0.0):
                blocking.append("unnecessary_join_regression")
            if not statistically_checked and challenger_metrics.get("gold_comparison_score", 0.0) + min_improvement < champion_metrics.get("gold_comparison_score", 0.0):
                blocking.append("gold_comparison_regression")
            if not statistically_checked and challenger_metrics.get("unseen_db_sql_validation_rate", 0.0) + min_improvement < champion_metrics.get("unseen_db_sql_validation_rate", 0.0):
                blocking.append("unseen_db_regression")
        return {
            "can_promote": not blocking,
            "blocking_issues": list(dict.fromkeys(blocking)),
            "warnings": [] if champion_metrics else ["No current champion; challenger may become initial champion if hard gates pass."],
            "statistical_report": statistical_report,
        }

    @staticmethod
    def _bootstrap_report(
        challenger_rows: list[dict[str, Any]],
        champion_rows: list[dict[str, Any]],
        iterations: int = 1000,
        seed: int = 42,
    ) -> dict[str, Any]:
        challenger = {str(row.get("example_id", index)): row for index, row in enumerate(challenger_rows)}
        champion = {str(row.get("example_id", index)): row for index, row in enumerate(champion_rows)}
        ids = sorted(set(challenger) & set(champion))
        report: dict[str, Any] = {"bootstrap_iterations": iterations, "paired_examples": len(ids), "metrics": {}}
        if not ids:
            return report
        metric_fields = {
            "intent_macro_f1": ("intent_correct", True),
            "base_table_accuracy": ("base_table_correct", True),
            "join_decision_macro_f1": ("join_correct", True),
            "sql_validation_rate": ("sql_valid", True),
            "execution_match_rate": ("execution_match", True),
            "unnecessary_join_rate": ("unnecessary_join", False),
            "wrong_table_rate": ("wrong_table", False),
        }
        rng = random.Random(seed)
        for metric, (field, higher_is_better) in metric_fields.items():
            available = [item for item in ids if field in challenger[item] and field in champion[item]]
            if not available:
                continue
            def delta(sample: list[str]) -> float:
                c = sum(float(challenger[item][field]) for item in sample) / len(sample)
                b = sum(float(champion[item][field]) for item in sample) / len(sample)
                return c - b if higher_is_better else b - c
            point = delta(available)
            deltas = sorted(delta([rng.choice(available) for _ in available]) for _ in range(max(1, iterations)))
            p05 = _percentile(deltas, 5)
            report["metrics"][metric] = {
                "point_delta": point,
                "delta_p05": p05,
                "delta_p50": _percentile(deltas, 50),
                "delta_p95": _percentile(deltas, 95),
                "higher_is_better": higher_is_better,
                "regression_detected": p05 < 0.0,
            }
        return report


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)
