from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .model_candidate import ModelCandidate


class ModelSelector:
    def select_best(self, candidates: list[ModelCandidate], thresholds: dict[str, Any]) -> dict[str, Any]:
        minimums = thresholds.get("minimums") or thresholds
        accepted = []
        rejected = []
        blocking_issues = []
        for candidate in candidates:
            issues = _hard_blockers(candidate.metrics, minimums)
            if issues:
                rejected.append({"model": asdict(candidate), "blocking_issues": issues})
            else:
                accepted.append(candidate)
        if not accepted:
            blocking_issues.append("No candidate passed hard blockers.")
            return {"selected_model": None, "rejected_models": rejected, "selection_reason": "all_candidates_rejected", "blocking_issues": blocking_issues, "warnings": []}
        selected = sorted(accepted, key=lambda item: _selection_score(item.metrics), reverse=True)[0]
        return {
            "selected_model": asdict(selected),
            "rejected_models": rejected,
            "selection_reason": f"highest quality score {_selection_score(selected.metrics):.4f}",
            "blocking_issues": [],
            "warnings": [],
        }


def _selection_score(metrics: dict[str, Any]) -> float:
    positive = [
        "query_ir_validity_rate",
        "sql_validation_rate",
        "gold_comparison_score",
        "execution_match_rate",
        "structure_match_rate",
        "unseen_db_sql_validation_rate",
        "simple_query_pass_rate",
        "analytics_query_pass_rate",
        "overall_slot_accuracy",
    ]
    negative = ["unnecessary_join_rate", "wrong_table_rate"]
    score = sum(float(metrics.get(key, 0.0)) for key in positive)
    score -= sum(float(metrics.get(key, 0.0)) for key in negative)
    return score


def _hard_blockers(metrics: dict[str, Any], minimums: dict[str, Any]) -> list[str]:
    issues = []
    if float(metrics.get("unsafe_sql_count", metrics.get("unsafe_sql_count_max", 0)) or 0) > float(minimums.get("unsafe_sql_count_max", 0)):
        issues.append("unsafe_sql_count")
    if float(metrics.get("no_select_star_rate", 1.0) or 0.0) < float(minimums.get("no_select_star_rate", 1.0)):
        issues.append("select_star")
    if float(metrics.get("unnecessary_join_rate", metrics.get("unnecessary_join_rate_max", 0.0)) or 0.0) > float(minimums.get("unnecessary_join_rate_max", 0.05)):
        issues.append("unnecessary_join_rate")
    if float(metrics.get("wrong_table_rate", metrics.get("wrong_table_rate_max", 0.0)) or 0.0) > float(minimums.get("wrong_table_rate_max", 0.15)):
        issues.append("wrong_table_rate")
    if float(metrics.get("sql_validation_rate", 0.0) or 0.0) < float(minimums.get("sql_validation_rate", 0.90)):
        issues.append("sql_validation_rate")
    if float(metrics.get("simple_query_pass_rate", 1.0) or 1.0) < float(minimums.get("simple_query_pass_rate", 0.0)):
        issues.append("simple_query_pass_rate_regressed")
    return issues

