from __future__ import annotations

from typing import Any

from .model_selector import _hard_blockers


class PromotionPolicy:
    def can_promote(self, challenger_metrics: dict[str, Any], champion_metrics: dict[str, Any] | None, thresholds: dict[str, Any]) -> dict[str, Any]:
        minimums = thresholds.get("minimums") or thresholds
        blocking = _hard_blockers(challenger_metrics, minimums)
        champion_metrics = champion_metrics or {}
        min_improvement = float(minimums.get("model_promotion_min_improvement", 0.0))
        if champion_metrics:
            if challenger_metrics.get("simple_query_pass_rate", 0.0) < champion_metrics.get("simple_query_pass_rate", 0.0):
                blocking.append("simple_query_pass_rate_regression")
            if challenger_metrics.get("unnecessary_join_rate", 0.0) > champion_metrics.get("unnecessary_join_rate", 0.0):
                blocking.append("unnecessary_join_regression")
            if challenger_metrics.get("gold_comparison_score", 0.0) + min_improvement < champion_metrics.get("gold_comparison_score", 0.0):
                blocking.append("gold_comparison_regression")
            if challenger_metrics.get("unseen_db_sql_validation_rate", 0.0) + min_improvement < champion_metrics.get("unseen_db_sql_validation_rate", 0.0):
                blocking.append("unseen_db_regression")
        return {
            "can_promote": not blocking,
            "blocking_issues": list(dict.fromkeys(blocking)),
            "warnings": [] if champion_metrics else ["No current champion; challenger may become initial champion if hard gates pass."],
        }
