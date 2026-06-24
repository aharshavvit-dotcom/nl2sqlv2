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
        warnings: list[str] = []
        # Multi-seed variance check (when available and valid for governance)
        multi_seed_report = selected.metadata.get("multi_seed_report") if selected.metadata else None
        if multi_seed_report and isinstance(multi_seed_report, dict):
            is_valid_training_governance = multi_seed_report.get("is_valid_for_training_variance_governance", False)
            is_valid_eval_stability = multi_seed_report.get("is_valid_for_evaluation_stability", False)
            # Support both flat metric_std and nested metrics.*.std shapes
            variance = multi_seed_report.get("metric_std")
            if variance is None:
                variance = {
                    name: values.get("std")
                    for name, values in multi_seed_report.get("metrics", {}).items()
                    if isinstance(values, dict) and "std" in values
                }
            if is_valid_training_governance:
                # True training variance governance warnings
                high_variance = [
                    f"{metric}: std={std:.4f}" for metric, std in variance.items()
                    if isinstance(std, (int, float)) and std > 0.05
                ]
                if high_variance:
                    warnings.append(
                        "training_variance_warning: High metric variance across seeds: " + ", ".join(high_variance)
                    )
            elif is_valid_eval_stability:
                # Evaluation-only stability warnings (informational, not governance)
                high_variance = [
                    f"{metric}: std={std:.4f}" for metric, std in variance.items()
                    if isinstance(std, (int, float)) and std > 0.05
                ]
                if high_variance:
                    warnings.append(
                        "evaluation_stability_warning: High prediction stability variance: " + ", ".join(high_variance)
                    )
            else:
                mode = multi_seed_report.get("mode", "unknown")
                seeds_evaluated = multi_seed_report.get("seeds_evaluated", 0)
                warnings.append(
                    f"multi_seed_variance_not_available: mode={mode}, "
                    f"seeds_evaluated={seeds_evaluated}, "
                    f"is_valid_for_training_variance_governance=false"
                )
        return {
            "selected_model": asdict(selected),
            "rejected_models": rejected,
            "selection_reason": f"highest quality score {_selection_score(selected.metrics):.4f}",
            "blocking_issues": [],
            "multi_seed_report": multi_seed_report,
            "warnings": warnings,
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

