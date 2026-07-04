from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from .model_candidate import ModelCandidate


class ModelSelector:
    def select_best(self, candidates: list[ModelCandidate], thresholds: dict[str, Any]) -> dict[str, Any]:
        minimums = thresholds.get("minimums") or thresholds
        accepted = []
        rejected = []
        blocking_issues = []
        for candidate in candidates:
            issues = _eligibility_issues(candidate)
            if candidate.evaluation_mode != "real_model_predictions":
                issues.append("evaluation_mode_not_real_model_predictions")
            if not candidate.eligible_for_promotion:
                issues.append("candidate_not_eligible_for_promotion")
            if (candidate.metadata or {}).get("quality_gate_passed") is False:
                issues.append("quality_gate_not_passed")
            issues.extend(_hard_blockers(candidate.metrics, minimums))
            if issues:
                rejected.append({"model": asdict(candidate), "blocking_issues": list(dict.fromkeys(issues))})
            else:
                accepted.append(candidate)
        if not accepted:
            blocking_issues.append("No candidate passed hard blockers.")
            return {
                "selected_model": None,
                "eligible_candidates": [],
                "ineligible_candidates": [
                    {"name": item["model"]["name"], "reason": item["blocking_issues"][0], "reasons": item["blocking_issues"]}
                    for item in rejected
                ],
                "rejected_models": rejected,
                "selection_reason": "all_candidates_rejected",
                "selection_blocked": True,
                "selection_blocked_reason": "no_eligible_candidate",
                "blocking_issues": blocking_issues,
                "warnings": [],
            }
        selected = sorted(accepted, key=lambda item: _selection_score(item.metrics), reverse=True)[0]
        warnings: list[str] = []
        # Multi-seed variance check (when available and valid for governance)
        multi_seed_report = selected.metadata.get("multi_seed_report") if selected.metadata else None
        if multi_seed_report and isinstance(multi_seed_report, dict):
            mode = multi_seed_report.get("mode", "unknown")
            is_valid_training_governance = bool(
                mode == "full_retrain_multi_seed"
                and multi_seed_report.get("is_valid_for_training_variance_governance", False)
            )
            is_valid_eval_stability = bool(
                multi_seed_report.get("is_valid_for_evaluation_stability", False)
                or multi_seed_report.get("evaluation_stability_available", False)
                or mode == "evaluation_only_stability"
            )
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
                        "evaluation_stability_warning: High metric variance across evaluation-only seed reruns: "
                        + ", ".join(high_variance)
                    )
                if multi_seed_report.get("evaluation_stability_interpretation"):
                    warnings.append(
                        "evaluation_stability_interpretation: "
                        + str(multi_seed_report.get("evaluation_stability_interpretation"))
                    )
            else:
                seeds_evaluated = multi_seed_report.get("seeds_evaluated", 0)
                warnings.append(
                    f"multi_seed_variance_not_available: mode={mode}, "
                    f"seeds_evaluated={seeds_evaluated}, "
                    f"is_valid_for_training_variance_governance=false"
                )
        predicted_sql_execution = _predicted_sql_summary(selected.metrics, selected.metadata or {}, thresholds)
        # Phase 4+5: Use metric_sample_counts for variance warnings
        metric_sample_counts = multi_seed_report.get("metric_sample_counts", {}) if multi_seed_report else {}
        return {
            "selected_model": asdict(selected),
            "eligible_candidates": [asdict(item) for item in accepted],
            "ineligible_candidates": [
                {"name": item["model"]["name"], "reason": item["blocking_issues"][0], "reasons": item["blocking_issues"]}
                for item in rejected
            ],
            "rejected_models": rejected,
            "selection_reason": f"highest quality score {_selection_score(selected.metrics):.4f}",
            "blocking_issues": [],
            "multi_seed_report": multi_seed_report,
            "metric_sample_counts": metric_sample_counts,
            "predicted_sql_execution": predicted_sql_execution,
            "warnings": warnings,
            "selection_blocked": False,
            "selection_blocked_reason": None,
        }


def _eligibility_issues(candidate: ModelCandidate) -> list[str]:
    issues: list[str] = []
    if candidate.model_artifact_source not in {"model_bundle", "artifact_dirs", "model_bundle_candidate"}:
        issues.append("gold_or_oracle_report" if candidate.model_artifact_source == "gold_baseline" else "stale_report")
    metadata = candidate.metadata or {}
    enforce_freshness = bool(
        metadata.get("enforce_freshness")
        or candidate.candidate_bundle_id
        or candidate.manifest_bundle_id
        or metadata.get("candidate_bundle_generated_at")
    )
    if enforce_freshness:
        if not candidate.candidate_bundle_id or not candidate.manifest_bundle_id:
            issues.append("bundle_id_missing")
        elif candidate.candidate_bundle_id != candidate.manifest_bundle_id:
            issues.append("bundle_id_mismatch")
        manifest_generated = metadata.get("candidate_bundle_generated_at")
        if not candidate.generated_at:
            issues.append("report_generated_at_missing")
        elif manifest_generated and _timestamp(candidate.generated_at) < _timestamp(str(manifest_generated)):
            issues.append("stale_report")
    return issues


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)


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
    if float(metrics.get("unsafe_sql_count", metrics.get("unsafe_sql_count_max", 0)) or 0) > float(_minimum(minimums, "unsafe_sql_count_max", 0)):
        issues.append("unsafe_sql_count")
    if float(metrics.get("no_select_star_rate", 1.0) or 0.0) < float(_minimum(minimums, "no_select_star_rate", 1.0)):
        issues.append("select_star")
    if float(metrics.get("unnecessary_join_rate", metrics.get("unnecessary_join_rate_max", 0.0)) or 0.0) > float(_minimum(minimums, "unnecessary_join_rate_max", 0.05)):
        issues.append("unnecessary_join_rate")
    if float(metrics.get("wrong_table_rate", metrics.get("wrong_table_rate_max", 0.0)) or 0.0) > float(_minimum(minimums, "wrong_table_rate_max", 0.15)):
        issues.append("wrong_table_rate")
    if float(metrics.get("sql_validation_rate", 0.0) or 0.0) < float(_minimum(minimums, "sql_validation_rate", 0.90)):
        issues.append("sql_validation_rate")
    if float(metrics.get("simple_query_pass_rate", 1.0) or 1.0) < float(_minimum(minimums, "simple_query_pass_rate", 0.0)):
        issues.append("simple_query_pass_rate_regressed")
    if bool(minimums.get("controlled_predicted_sql_required", False)):
        if float(metrics.get("controlled_predicted_sql_safe_sql_rate", 1.0) or 0.0) < 1.0:
            issues.append("controlled_predicted_sql_safe_sql_rate")
        if int(metrics.get("controlled_predicted_sql_unsafe_sql_count", 0) or 0) > 0:
            issues.append("controlled_predicted_sql_unsafe_sql_count")
    return issues


def _minimum(minimums: dict[str, Any], key: str, default: Any) -> Any:
    value = minimums.get(key, default)
    if isinstance(value, dict):
        return value.get("production_min", value.get("warning_min", value.get("smoke_min", default)))
    return value


def _predicted_sql_summary(
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    report = metadata.get("controlled_predicted_sql_report") if metadata else None
    values = dict(metrics)
    if isinstance(report, dict):
        values.setdefault(
            "controlled_predicted_sql_execution_match_rate",
            report.get("predicted_execution_match_rate", report.get("predicted_result_value_match_rate")),
        )
        values.setdefault("controlled_predicted_sql_execution_success_rate", report.get("predicted_execution_success_rate"))
        values.setdefault("controlled_predicted_sql_row_count_match_rate", report.get("predicted_row_count_match_rate"))
        values.setdefault("controlled_predicted_sql_safe_sql_rate", report.get("predicted_safe_sql_rate"))
        values.setdefault(
            "controlled_predicted_sql_unsafe_sql_count",
            report.get("unsafe_sql_count", report.get("predicted_unsafe_sql_count")),
        )
    available = any(
        key in values and values.get(key) is not None
        for key in [
            "controlled_predicted_sql_execution_match_rate",
            "controlled_predicted_sql_safe_sql_rate",
            "controlled_predicted_sql_unsafe_sql_count",
        ]
    )
    minimums = thresholds.get("minimums") or thresholds
    blocking = bool(minimums.get("controlled_predicted_sql_required", False))
    return {
        "available": available,
        "execution_match_rate": values.get("controlled_predicted_sql_execution_match_rate"),
        "execution_success_rate": values.get("controlled_predicted_sql_execution_success_rate"),
        "row_count_match_rate": values.get("controlled_predicted_sql_row_count_match_rate"),
        "safe_sql_rate": values.get("controlled_predicted_sql_safe_sql_rate"),
        "unsafe_sql_count": values.get("controlled_predicted_sql_unsafe_sql_count"),
        "blocking": blocking,
    }
