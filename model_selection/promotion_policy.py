from __future__ import annotations

import random
from typing import Any

from .model_selector import _hard_blockers, _timestamp, safe_float


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

PREDICTED_SQL_METRICS = {
    "controlled_predicted_sql_execution_match_rate": True,
    "controlled_predicted_sql_execution_success_rate": True,
    "controlled_predicted_sql_row_count_match_rate": True,
    "controlled_predicted_sql_safe_sql_rate": True,
    "controlled_predicted_sql_unsafe_sql_count": False,
}


class PromotionPolicy:
    def can_promote(
        self,
        challenger_metrics: dict[str, Any],
        champion_metrics: dict[str, Any] | None,
        thresholds: dict[str, Any],
        bootstrap_iterations: int = 1000,
        challenger_metadata: dict[str, Any] | None = None,
        champion_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        minimums = thresholds.get("minimums") or thresholds
        blocking = _hard_blockers(challenger_metrics, minimums)
        metadata_supplied = challenger_metadata is not None
        challenger_metadata = challenger_metadata or {}
        evaluation_mode = challenger_metadata.get("evaluation_mode")
        if evaluation_mode and evaluation_mode != "real_model_predictions":
            blocking.append("evaluation_mode_not_real_model_predictions")
        if challenger_metadata.get("eligible_for_promotion") is False:
            blocking.append("candidate_not_eligible_for_promotion")
        if challenger_metadata.get("quality_gate_passed") is False:
            blocking.append("quality_gate_not_passed")
        artifact_source = challenger_metadata.get("model_artifact_source", "legacy_cache")
        if metadata_supplied and artifact_source not in {"model_bundle", "artifact_dirs", "model_bundle_candidate"}:
            blocking.append("ineligible_model_artifact_source")
        if challenger_metadata.get("enforce_freshness"):
            candidate_bundle_id = challenger_metadata.get("candidate_bundle_id")
            manifest_bundle_id = challenger_metadata.get("manifest_bundle_id")
            if not candidate_bundle_id or not manifest_bundle_id:
                blocking.append("bundle_id_missing")
            elif candidate_bundle_id != manifest_bundle_id:
                blocking.append("bundle_id_mismatch")
            pipeline_run_id = challenger_metadata.get("pipeline_run_id")
            if not pipeline_run_id:
                blocking.append("pipeline_run_id_missing")
            generated_at = challenger_metadata.get("generated_at")
            manifest_generated_at = challenger_metadata.get("candidate_bundle_generated_at")
            if not generated_at or (
                manifest_generated_at and _timestamp(str(generated_at)) < _timestamp(str(manifest_generated_at))
            ):
                blocking.append("stale_report")
        if bool(minimums.get("controlled_predicted_sql_required", False)):
            unsafe_cnt = safe_float(challenger_metrics.get("controlled_predicted_sql_unsafe_sql_count"), 0.0) or 0.0
            if int(unsafe_cnt) > 0:
                blocking.append("controlled_predicted_sql_unsafe_sql_count")
            safe_rate = safe_float(challenger_metrics.get("controlled_predicted_sql_safe_sql_rate"), 1.0)
            if safe_rate is None or safe_rate < 1.0:
                blocking.append("controlled_predicted_sql_safe_sql_rate")
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
        predicted_sql_execution = self._predicted_sql_summary(challenger_metrics, champion_metrics)
        return {
            "can_promote": not blocking,
            "blocking_issues": list(dict.fromkeys(blocking)),
            "warnings": warnings,
            "statistical_report": statistical_report,
            "statistical_checks": statistical_checks,
            "point_estimate_fallback_checks": point_fallback_checks,
            "predicted_sql_execution": predicted_sql_execution,
            "blocking_regressions": [item for item in list(dict.fromkeys(blocking)) if item.endswith("_regression")],
            "evaluation_mode": evaluation_mode or "legacy_unspecified",
            "model_artifact_source": challenger_metadata.get("model_artifact_source", "legacy_cache"),
            "eligible_for_promotion": not blocking,
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
        challenger = safe_float(challenger_metrics.get(metric_name))
        champion = safe_float(champion_metrics.get(metric_name))
        if challenger is None or champion is None:
            return None
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

    @staticmethod
    def _predicted_sql_summary(
        challenger_metrics: dict[str, Any],
        champion_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        checks = {}
        for metric_name, higher_is_better in PREDICTED_SQL_METRICS.items():
            check = PromotionPolicy._point_estimate_check(
                metric_name,
                challenger_metrics,
                champion_metrics,
                higher_is_better=higher_is_better,
                min_improvement=0.0,
            ) if champion_metrics else None
            if check is not None:
                checks[metric_name] = check
        available = any(metric in challenger_metrics for metric in PREDICTED_SQL_METRICS)

        # Phase 8: Per-case comparison by stable case_id
        per_case_comparison = _compare_predicted_sql_per_case(
            challenger_metrics.get("controlled_predicted_sql_cases"),
            champion_metrics.get("controlled_predicted_sql_cases") if champion_metrics else None,
        )

        return {
            "available": available,
            "execution_match_rate": challenger_metrics.get("controlled_predicted_sql_execution_match_rate"),
            "safe_sql_rate": challenger_metrics.get("controlled_predicted_sql_safe_sql_rate"),
            "unsafe_sql_count": challenger_metrics.get("controlled_predicted_sql_unsafe_sql_count"),
            "blocking": False,
            "deltas": checks,
            "per_case_comparison": per_case_comparison,
        }


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _compare_predicted_sql_per_case(
    challenger_cases: list[dict[str, Any]] | None,
    champion_cases: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Phase 8: Compare predicted-SQL results by stable case_id between champion and challenger."""
    if not challenger_cases:
        return {"available": False, "reason": "no_challenger_cases"}
    if not champion_cases:
        return {"available": False, "reason": "no_champion_cases"}

    # Index by case_id
    challenger_by_id = {
        str(c.get("case_id") or c.get("example_id", i)): c
        for i, c in enumerate(challenger_cases)
    }
    champion_by_id = {
        str(c.get("case_id") or c.get("example_id", i)): c
        for i, c in enumerate(champion_cases)
    }
    common_ids = sorted(set(challenger_by_id) & set(champion_by_id))
    if not common_ids:
        return {"available": False, "reason": "no_common_case_ids"}

    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    unchanged: int = 0
    
    for case_id in common_ids:
        chall = challenger_by_id[case_id]
        champ = champion_by_id[case_id]
        chall_match = bool(chall.get("final_execution_match") or chall.get("predicted_result_value_match"))
        champ_match = bool(champ.get("final_execution_match") or champ.get("predicted_result_value_match"))
        if chall_match and not champ_match:
            improvements.append({"case_id": case_id, "question": chall.get("question")})
        elif not chall_match and champ_match:
            regressions.append({"case_id": case_id, "question": chall.get("question")})
        else:
            unchanged += 1

    # Phase 9: Bootstrap percentiles
    bootstrap_iterations = 1000
    p05, p50, p95 = 0.0, 0.0, 0.0
    point_delta = 0.0
    
    if common_ids:
        rng = random.Random(42)
        def delta(sample: list[str]) -> float:
            c = sum(bool(challenger_by_id[i].get("final_execution_match") or challenger_by_id[i].get("predicted_result_value_match")) for i in sample) / len(sample)
            b = sum(bool(champion_by_id[i].get("final_execution_match") or champion_by_id[i].get("predicted_result_value_match")) for i in sample) / len(sample)
            return c - b
            
        point_delta = delta(common_ids)
        if len(common_ids) >= 10:  # Only bootstrap if enough cases
            deltas = sorted(delta([rng.choice(common_ids) for _ in common_ids]) for _ in range(bootstrap_iterations))
            p05 = _percentile(deltas, 5)
            p50 = _percentile(deltas, 50)
            p95 = _percentile(deltas, 95)
        else:
            p05 = point_delta
            p50 = point_delta
            p95 = point_delta

    return {
        "available": True,
        "common_cases": len(common_ids),
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "execution_match_delta": point_delta,
        "delta_p05": p05,
        "delta_p50": p50,
        "delta_p95": p95,
        "regression_detected": p05 < 0.0,
        "statistical_check_available": len(common_ids) >= 10,
        "reason": "" if len(common_ids) >= 10 else "insufficient_common_cases",
        "minimum_cases_required": 10,
    }
