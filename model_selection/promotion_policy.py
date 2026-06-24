from __future__ import annotations

import random
from typing import Any

from .model_selector import _hard_blockers


PROMOTION_CRITICAL_METRICS = {
    "intent_macro_f1": True,
    "base_table_accuracy": True,
    "join_decision_macro_f1": True,
    "router_macro_f1": True,
    "sql_validation_rate": True,
    "execution_match_rate": True,
    "simple_query_pass_rate": True,
    "gold_comparison_score": True,
    "unseen_db_sql_validation_rate": True,
    "unnecessary_join_rate": False,
    "wrong_table_rate": False,
    "unsafe_sql_count": False,
}


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
        point_fallback_checks: dict[str, Any] = {}
        statistical_checks: dict[str, Any] = {}
        warnings: list[str] = []
        if champion_metrics:
            bootstrap_metrics = statistical_report.get("metrics", {})
            for metric_name, higher_is_better in PROMOTION_CRITICAL_METRICS.items():
                if metric_name in bootstrap_metrics:
                    check = {
                        **bootstrap_metrics[metric_name],
                        "metric_name": metric_name,
                        "statistical_check_available": True,
                    }
                    statistical_checks[metric_name] = check
                    if check.get("regression_detected"):
                        blocking.append(f"{metric_name}_statistical_regression")
                    continue
                fallback = self._point_estimate_check(
                    metric_name,
                    challenger_metrics,
                    champion_metrics,
                    higher_is_better=higher_is_better,
                    min_improvement=min_improvement,
                )
                if fallback is None:
                    warnings.append(f"No promotion regression check available for {metric_name}")
                    continue
                point_fallback_checks[metric_name] = fallback
                if fallback["regression_detected"]:
                    blocking.append(f"{metric_name}_regression")
        else:
            warnings.append("No current champion; challenger may become initial champion if hard gates pass.")
        return {
            "can_promote": not blocking,
            "blocking_issues": list(dict.fromkeys(blocking)),
            "warnings": warnings,
            "statistical_report": statistical_report,
            "statistical_checks": statistical_checks,
            "point_estimate_fallback_checks": point_fallback_checks,
            "blocking_regressions": [item for item in list(dict.fromkeys(blocking)) if item.endswith("_regression")],
        }

    @staticmethod
    def _point_estimate_check(
        metric_name: str,
        challenger_metrics: dict[str, Any],
        champion_metrics: dict[str, Any],
        higher_is_better: bool,
        min_improvement: float,
    ) -> dict[str, Any] | None:
        if metric_name not in challenger_metrics or metric_name not in champion_metrics:
            return None
        challenger = float(challenger_metrics.get(metric_name) or 0.0)
        champion = float(champion_metrics.get(metric_name) or 0.0)
        point_delta = challenger - champion if higher_is_better else champion - challenger
        if higher_is_better:
            regression = challenger + min_improvement < champion
        else:
            regression = challenger > champion
        return {
            "metric_name": metric_name,
            "statistical_check_available": False,
            "challenger": challenger,
            "champion": champion,
            "higher_is_better": higher_is_better,
            "point_delta": point_delta,
            "regression_detected": regression,
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
            "router_macro_f1": ("router_correct", True),
            "sql_validation_rate": ("sql_valid", True),
            "execution_match_rate": ("execution_match", True),
            "simple_query_pass_rate": ("simple_query_pass", True),
            "gold_comparison_score": ("gold_comparison_score", True),
            "unseen_db_sql_validation_rate": ("unseen_db_sql_valid", True),
            "unnecessary_join_rate": ("unnecessary_join", False),
            "wrong_table_rate": ("wrong_table", False),
            "unsafe_sql_count": ("unsafe_sql", False),
        }
        rng = random.Random(seed)
        for metric, (field, higher_is_better) in metric_fields.items():
            available = [item for item in ids if field in challenger[item] and field in champion[item]
                         and challenger[item][field] is not None and champion[item][field] is not None]
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
                "metric_name": metric,
                "statistical_check_available": True,
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
