from __future__ import annotations

from typing import Any


CRITICAL_METRICS = [
    "query_ir_validity_rate",
    "sql_validation_rate",
    "unsafe_sql_count",
    "unnecessary_join_rate",
    "wrong_table_rate",
    "simple_query_pass_rate",
]

# Semantic metrics that block promotion when below threshold.
# These use applicability-aware denominators — a metric is only checked
# when its applicable_cases count meets the configured minimum.
SEMANTIC_CRITICAL_METRICS = [
    "simple_query_semantic_pass_rate",
    "projection_exact_match_rate",
    "filter_column_accuracy_rate",
    "filter_value_accuracy_rate",
    "dimension_column_accuracy_rate",
]

POLICY_ONLY_THRESHOLDS = {
    "controlled_predicted_sql_required",
}


class ModelQualityGate:
    def evaluate(self, evaluation_report: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
        minimums = {
            **(thresholds.get("minimums") or (thresholds if "classification_metrics" not in thresholds else {})),
            **(thresholds.get("classification_metrics") or {}),
            **(thresholds.get("calibration_metrics") or {}),
            **_semantic_thresholds(thresholds),
        }
        mode = _quality_gate_mode(evaluation_report)
        metrics, present = self._extract_metrics(evaluation_report)
        failed_checks: list[dict[str, Any]] = []
        warnings: list[str] = []
        missing_metrics: list[str] = []

        execution_status = self._execution_status(evaluation_report)
        feedback_config = evaluation_report.get("feedback_regression") or {}
        source_failures, source_warnings = self._evaluation_source_checks(evaluation_report)
        failed_checks.extend(source_failures)
        warnings.extend(source_warnings)

        require_non_degenerate = bool(
            ((thresholds.get("calibration") or {}).get("require_non_degenerate_confidence") or {}).get(
                "production", False
            )
        )

        for key, expected in minimums.items():
            metric_key = _threshold_metric_name(key)
            if isinstance(expected, dict):
                expected = _threshold_value_for_mode(expected, mode)
                if expected is None:
                    warnings.append(f"Skipping {key}; no threshold configured for mode={mode}.")
                    continue
            if metric_key == "model_promotion_min_improvement" and metric_key not in metrics:
                warnings.append("Skipping model_promotion_min_improvement; it is a promotion policy threshold, not an evaluation metric.")
                continue
            if metric_key in POLICY_ONLY_THRESHOLDS:
                continue
            if metric_key in {"execution_match_rate", "final_sql_execution_accuracy"} and not execution_status.get("available"):
                if mode in {"production", "release"} and execution_status.get("required"):
                    failed_checks.append({
                        "metric": "execution_unavailable",
                        "actual": execution_status.get("unavailable_reason") or "unavailable",
                        "expected": "execution_available",
                        "comparison": "==",
                    })
                else:
                    warnings.append(
                        f"execution_unavailable: {execution_status.get('unavailable_reason') or 'not_configured'}; "
                        f"{key} threshold skipped for mode={mode}."
                    )
                continue
            if metric_key == "sql_structure_match_rate" and mode in {"debug", "baseline"}:
                warnings.append(f"Skipping {key}; structure matching is advisory for mode={mode}.")
                continue
            if metric_key == "feedback_regression_pass_rate":
                feedback_required = bool(
                    feedback_config.get("enabled", False)
                    and feedback_config.get("required_for_production", False)
                    and mode in {"production", "release"}
                )
                if not feedback_required:
                    warnings.append(f"feedback_regression_not_required_for_mode: mode={mode}")
                    continue
            if mode == "debug" and metric_key not in {
                "query_ir_validity_rate",
                "sql_validation_rate",
                "unsafe_sql_count",
                "no_select_star_rate",
            }:
                actual = metrics.get(metric_key)
                if actual is None:
                    warnings.append(f"Debug advisory metric missing: {metric_key}")
                else:
                    passed = actual <= expected if key.endswith("_max") else actual >= expected
                    if not passed:
                        warnings.append(
                            f"debug_threshold_warning: {metric_key}={actual} expected "
                            f"{'<=' if key.endswith('_max') else '>='} {expected}"
                        )
                continue
            if key.endswith("_production"):
                base_metric = _threshold_metric_name(key.removesuffix("_production"))
                actual = metrics.get(base_metric)
                if mode in {"production", "release"} and base_metric == "simple_query_pass_rate":
                    if actual is None:
                        missing_metrics.append(base_metric)
                        failed_checks.append({
                            "metric": key,
                            "actual": "missing",
                            "expected": expected,
                            "comparison": ">=",
                        })
                    elif isinstance(actual, (int, float)) and actual < expected:
                        failed_checks.append({
                            "metric": key,
                            "actual": actual,
                            "expected": expected,
                            "comparison": ">=",
                        })
                elif actual is not None and isinstance(actual, (int, float)) and actual < expected:
                    warnings.append(
                        f"production_threshold_warning: {base_metric}={actual:.4f} "
                        f"below production target {expected}"
                    )
                continue
            actual = metrics.get(metric_key)
            if actual is None:
                missing_metrics.append(metric_key)
                failed_checks.append({
                    "metric": key,
                    "actual": "missing",
                    "expected": expected,
                    "comparison": "<=" if key.endswith("_max") else ">=",
                })
                continue
            passed = actual <= expected if key.endswith("_max") else actual >= expected
            if not passed:
                failed_checks.append({"metric": key, "actual": actual, "expected": expected, "comparison": "<=" if key.endswith("_max") else ">="})

        for metric in CRITICAL_METRICS:
            if metric not in present:
                missing_metrics.append(metric)
                if mode == "debug" and metric not in {"unsafe_sql_count", "sql_validation_rate"}:
                    warnings.append(f"Debug advisory critical metric missing: {metric}")
                else:
                    failed_checks.append({
                        "metric": metric,
                        "actual": "missing",
                        "expected": "present",
                        "comparison": "exists",
                    })

        contribution = evaluation_report.get("dataset_contribution_report")
        if evaluation_report.get("dataset_contribution_report_required") and not contribution:
            missing_metrics.append("dataset_contribution_report_exists")
            failed_checks.append({
                "metric": "dataset_contribution_report",
                "actual": "missing",
                "expected": "present",
                "comparison": "exists",
            })
        if contribution:
            metrics["dataset_contribution_report_exists"] = True
            metrics["leakage_check_passed"] = bool(contribution.get("leakage_check_passed", False))
            metrics["full_training_dataset_minimums_passed"] = bool(
                contribution.get("full_training_dataset_minimums_passed", True)
            )
            if not contribution.get("leakage_check_passed", False):
                failed_checks.append({
                    "metric": "dataset_leakage_check",
                    "actual": False,
                    "expected": True,
                    "comparison": "==",
                })
            requested = set(contribution.get("datasets_requested") or [])
            by_dataset = contribution.get("by_dataset") or {}
            for dataset_name in ["spider", "bird-mini"]:
                if dataset_name in requested and int((by_dataset.get(dataset_name) or {}).get("converted_to_queryir", 0)) <= 0:
                    failed_checks.append({
                        "metric": f"{dataset_name}_usable_examples",
                        "actual": 0,
                        "expected": "> 0",
                        "comparison": ">",
                    })
            for item in contribution.get("minimum_failures") or []:
                failed_checks.append({
                    "metric": f"{item.get('dataset')}_minimum_contribution",
                    "actual": item.get("converted_to_queryir", 0),
                    "expected": item.get("minimum_required", 0),
                    "comparison": ">=",
                })
        elif evaluation_report.get("dataset_contribution_report_required"):
            metrics["dataset_contribution_report_exists"] = False
            metrics["leakage_check_passed"] = False
            metrics["full_training_dataset_minimums_passed"] = False

        missing_metrics = sorted(set(missing_metrics))

        # --- Sklearn and Calibration model metadata check (Review #15) ---
        sklearn_info = evaluation_report.get("sklearn_info") or {}
        if mode in {"production", "release"}:
            if not sklearn_info.get("retrieval_sklearn_metadata_valid", True):
                failed_checks.append({
                    "metric": "retrieval_sklearn_metadata_valid",
                    "actual": False,
                    "expected": True,
                    "comparison": "==",
                })
            if not sklearn_info.get("retrieval_checksums_valid", True):
                failed_checks.append({
                    "metric": "retrieval_checksums_valid",
                    "actual": False,
                    "expected": True,
                    "comparison": "==",
                })
            if not sklearn_info.get("calibration_metadata_valid", True):
                failed_checks.append({
                    "metric": "calibration_metadata_valid",
                    "actual": False,
                    "expected": True,
                    "comparison": "==",
                })

        # Controlled fixture validation check
        controlled_fixture = evaluation_report.get("controlled_fixture_evaluation") or {}
        if isinstance(controlled_fixture, dict) and controlled_fixture:
            fixture_summary = controlled_fixture.get("summary") or {}
            if not fixture_summary.get("execution_success_rate", 0.0) == 1.0:
                warnings.append("Controlled gold-SQL fixture validation did not achieve 100% execution success")
            if controlled_fixture.get("measures_model_predictions", True) is False:
                warnings.append("Controlled fixture evaluation validates gold SQL, not model-predicted SQL")
        elif evaluation_report.get("controlled_fixture_required", False):
            warnings.append("Controlled fixture evaluation is required but no report was found")

        controlled_predicted = evaluation_report.get("controlled_predicted_sql_execution") or {}
        if isinstance(controlled_predicted, dict) and controlled_predicted:
            controlled_required = bool(
                evaluation_report.get("controlled_predicted_sql_required", False)
            )
            if controlled_predicted.get("error") and evaluation_report.get("controlled_predicted_sql_required", False):
                failed_checks.append({
                    "metric": "controlled_predicted_sql_execution",
                    "actual": controlled_predicted.get("error"),
                    "expected": "report without error",
                    "comparison": "==",
                })
            if controlled_predicted.get("central_sql_validator_used") is False:
                target = failed_checks if evaluation_report.get("controlled_predicted_sql_required", False) else warnings
                message = "controlled_predicted_sql_missing_central_sql_validator"
                if target is failed_checks:
                    failed_checks.append({
                        "metric": message,
                        "actual": False,
                        "expected": True,
                        "comparison": "==",
                    })
                else:
                    warnings.append(message)
            if controlled_required and controlled_predicted.get("passed") is not True:
                failed_checks.append({
                    "metric": "controlled_predicted_sql_passed",
                    "actual": {
                        "passed": controlled_predicted.get("passed", False),
                        "cases_total": controlled_predicted.get("cases_total", 0),
                        "predictions_generated": controlled_predicted.get("predictions_generated", 0),
                        "abstention_count": controlled_predicted.get("abstention_count", 0),
                        "execution_match_rate": controlled_predicted.get("predicted_execution_match_rate"),
                    },
                    "expected": True,
                    "comparison": "==",
                })
            if (
                controlled_required
                and int(controlled_predicted.get("predicted_unsafe_sql_count", controlled_predicted.get("unsafe_sql_count", 0)) or 0) > 0
            ):
                failed_checks.append({
                    "metric": "controlled_predicted_sql_unsafe_sql_count",
                    "actual": controlled_predicted.get("predicted_unsafe_sql_count", controlled_predicted.get("unsafe_sql_count", 0)),
                    "expected": 0,
                    "comparison": "<=",
                })
        elif evaluation_report.get("controlled_predicted_sql_required", False):
            failed_checks.append({
                "metric": "controlled_predicted_sql_execution",
                "actual": "missing",
                "expected": "present",
                "comparison": "exists",
            })

        calibration = evaluation_report.get("test_performance", {}).get("calibration", {})
        if calibration.get("calibration_degenerate") is True:
            issue = {
                "metric": "calibration_degenerate",
                "actual": True,
                "expected": False,
                "comparison": "==",
            }
            if require_non_degenerate and mode in {"production", "release"}:
                failed_checks.append(issue)
            else:
                warnings.append("calibration_degenerate: confidence thresholds are disabled")

        selection = evaluation_report.get("model_selection_report") or evaluation_report.get("model_selection") or {}
        if evaluation_report.get("model_selection_required") and not selection:
            failed_checks.append({
                "metric": "model_selection_freshness",
                "actual": "missing",
                "expected": "fresh_eligible_candidate",
                "comparison": "==",
            })
        if selection and (
            selection.get("selection_blocked") is True
            or selection.get("selected_model") is None
            or selection.get("model_selection_stale") is True
        ):
            target = failed_checks if mode in {"production", "release"} else None
            if target is not None:
                target.append({
                    "metric": "model_selection_freshness",
                    "actual": selection.get("selection_blocked_reason") or "stale_or_ineligible",
                    "expected": "fresh_eligible_candidate",
                    "comparison": "==",
                })
            else:
                warnings.append("model_selection_stale_or_ineligible")

        # --- Semantic metric checks (applicability-aware) ---
        semantic_eval = (
            evaluation_report.get("test_performance", {}).get("summary", {})
            .get("semantic_evaluation", {})
        ) or evaluation_report.get("summary", {}).get("semantic_evaluation", {})
        semantic_thresholds = thresholds.get("semantic") or {}
        min_applicable = int(semantic_thresholds.get("minimum_applicable_cases", 50))
        for metric_name in SEMANTIC_CRITICAL_METRICS:
            # Get applicability-aware metric from semantic_evaluation
            if metric_name == "simple_query_semantic_pass_rate":
                actual = (
                    semantic_eval.get("simple_query_semantic_pass_rate")
                    or metrics.get("simple_query_semantic_pass_rate")
                )
                if actual is not None:
                    metrics[metric_name] = actual
                    present.add(metric_name)
            else:
                # These have applicability-aware sub-structures
                base_name = metric_name.removesuffix("_rate")
                sub = semantic_eval.get(base_name) or {}
                actual = sub.get("value")
                applicable_cases = int(sub.get("applicable_cases", 0))
                if actual is not None:
                    metrics[metric_name] = actual
                    present.add(metric_name)
                    # Only enforce gate when we have sufficient support
                    if applicable_cases < min_applicable and mode in {"production", "release"}:
                        warnings.append(
                            f"{metric_name}: only {applicable_cases} applicable cases "
                            f"(minimum {min_applicable}); threshold not enforced"
                        )
                        continue
            threshold_config = semantic_thresholds.get(metric_name)
            if threshold_config is None:
                continue
            expected = _threshold_value_for_mode(threshold_config, mode) if isinstance(threshold_config, dict) else threshold_config
            if expected is None:
                continue
            actual_val = metrics.get(metric_name)
            if actual_val is not None and isinstance(actual_val, (int, float)) and actual_val < expected:
                if mode in {"production", "release"}:
                    failed_checks.append({
                        "metric": metric_name,
                        "actual": actual_val,
                        "expected": expected,
                        "comparison": ">=",
                    })
                else:
                    warnings.append(
                        f"semantic_threshold_warning: {metric_name}={actual_val:.4f} "
                        f"below target {expected}"
                    )

        # --- Safe-but-wrong SQL blocking ---
        safe_but_wrong = metrics.get("controlled_predicted_sql_safe_but_wrong_sql_rate")
        safe_but_wrong_max = float(
            (semantic_thresholds.get("safe_but_wrong_sql_rate_max") or {}).get("production_max", 0.30)
            if isinstance(semantic_thresholds.get("safe_but_wrong_sql_rate_max"), dict)
            else semantic_thresholds.get("safe_but_wrong_sql_rate_max", 0.30)
        )
        if (
            safe_but_wrong is not None
            and isinstance(safe_but_wrong, (int, float))
            and safe_but_wrong > safe_but_wrong_max
            and mode in {"production", "release"}
        ):
            failed_checks.append({
                "metric": "controlled_predicted_sql_safe_but_wrong_sql_rate",
                "actual": safe_but_wrong,
                "expected": safe_but_wrong_max,
                "comparison": "<=",
            })

        sql_safe = bool(
            metrics.get("sql_validation_rate", 0.0) >= 0.90
            and metrics.get("post_abstention_unsafe_sql_count", metrics.get("unsafe_sql_count", 0)) == 0
        )
        semantic_match_ready = bool(
            metrics.get("controlled_predicted_sql_execution_match_rate", 0.0) >= 0.70
            and metrics.get("controlled_predicted_sql_result_value_match_rate", 0.0) >= 0.70
            and (safe_but_wrong is None or safe_but_wrong <= safe_but_wrong_max)
        )
        simple_query_ready = bool(metrics.get("simple_query_pass_rate", 0.0) >= 0.95)
        semantic_grounding_ready = bool(
            metrics.get("projection_exact_match_rate", 0.0) >= 0.70
            and metrics.get("filter_column_accuracy_rate", 0.0) >= 0.70
            and metrics.get("filter_value_accuracy_rate", 0.0) >= 0.70
            and metrics.get("dimension_column_accuracy_rate", 0.0) >= 0.65
        )
        bundle_ready = bool(
            not evaluation_report.get("model_selection_required")
            or (
                selection
                and not selection.get("selection_blocked", False)
                and not selection.get("model_selection_stale", False)
                and selection.get("candidate_bundle_id") == selection.get("manifest_bundle_id")
            )
        )
        failed_checks = _dedupe_failed_checks(failed_checks)
        passed = not failed_checks
        return {
            "passed": passed,
            "quality_gate_mode": mode,
            "eligible_for_promotion": bool(passed and mode in {"production", "release"}),
            "failed_checks": failed_checks,
            "blocking_failures": failed_checks,
            "warnings": warnings,
            "missing_metrics": missing_metrics,
            "metrics": metrics,
            "execution_aware_evaluation": execution_status,
            "feedback_regression": {
                "enabled": bool(feedback_config.get("enabled", False)),
                "required_for_production": bool(feedback_config.get("required_for_production", False)),
                "available": "feedback_regression_pass_rate" in present,
            },
            "production_readiness_summary": {
                "sql_safe": sql_safe,
                "semantic_match_ready": semantic_match_ready,
                "simple_query_ready": simple_query_ready,
                "semantic_grounding_ready": semantic_grounding_ready,
                "bundle_ready": bundle_ready,
                "promotion_ready": bool(
                    passed and mode in {"production", "release"} and bundle_ready
                ),
            },
        }

    @staticmethod
    def _extract_metrics(report: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
        test_summary = report.get("test_performance", {}).get("summary", {})
        classification = report.get("test_performance", {}).get("classification_metrics", {})
        unseen_summary = report.get("unseen_db_performance", {}).get("summary", {})
        summary = report.get("summary", {})
        metrics = {
            **{key: value for key, value in summary.items() if isinstance(value, (int, float, bool))},
            **{key: value for key, value in test_summary.items() if isinstance(value, (int, float, bool))},
        }
        present = set(metrics)
        _copy_metric(metrics, present, "query_ir_validity_rate", test_summary, report)
        _copy_metric(metrics, present, "sql_validation_rate", test_summary, report)
        if "simple_query_pass_rate" in test_summary or "simple_query_pass_rate" in report:
            _copy_metric(metrics, present, "simple_query_pass_rate", test_summary, report)
        elif (
            _quality_gate_mode(report) not in {"production", "release"}
            and bool(report.get("allow_intent_accuracy_simple_query_fallback", True))
            and "intent_accuracy_rate" in test_summary
        ):
            metrics["simple_query_pass_rate"] = test_summary["intent_accuracy_rate"]
            present.add("simple_query_pass_rate")
            metrics["simple_query_pass_rate_fallback_used"] = True
        _copy_metric(metrics, present, "no_select_star_rate", report)
        _copy_metric(metrics, present, "unsafe_sql_count", summary, report)
        _copy_metric(metrics, present, "unnecessary_join_rate", test_summary, summary, report)
        _copy_metric(metrics, present, "wrong_table_rate", test_summary, summary, report)
        if "sql_validation_rate" in unseen_summary:
            metrics["unseen_db_sql_validation_rate"] = unseen_summary["sql_validation_rate"]
            present.add("unseen_db_sql_validation_rate")
        elif "unseen_db_sql_validation_rate" in report:
            metrics["unseen_db_sql_validation_rate"] = report["unseen_db_sql_validation_rate"]
            present.add("unseen_db_sql_validation_rate")
        _copy_metric(metrics, present, "feedback_regression_pass_rate", report)
        _copy_metric(metrics, present, "gold_comparison_score", summary, report)
        _copy_metric(metrics, present, "sql_structure_match_rate", summary, report)
        _copy_metric(metrics, present, "execution_match_rate", summary, report)
        _copy_metric(metrics, present, "model_promotion_min_improvement", report)
        predicted_sql = report.get("controlled_predicted_sql_execution") or {}
        if isinstance(predicted_sql, dict) and predicted_sql:
            predicted_metric_map = {
                "controlled_predicted_sql_execution_match_rate": "predicted_execution_match_rate",
                "controlled_predicted_sql_execution_success_rate": "predicted_execution_success_rate",
                "controlled_predicted_sql_row_count_match_rate": "predicted_row_count_match_rate",
                "controlled_predicted_sql_safe_sql_rate": "predicted_safe_sql_rate",
                "controlled_predicted_sql_unsafe_sql_count": "predicted_unsafe_sql_count",
                "controlled_predicted_sql_result_value_match_rate": "predicted_result_value_match_rate",
                "controlled_predicted_sql_safe_but_wrong_sql_rate": "safe_but_wrong_sql_rate",
            }
            for output_name, source_name in predicted_metric_map.items():
                value = predicted_sql.get(source_name)
                if value is None and output_name.endswith("_unsafe_sql_count"):
                    value = predicted_sql.get("unsafe_sql_count")
                if isinstance(value, (int, float, bool)):
                    metrics[output_name] = value
                    present.add(output_name)
        classification_map = {
            "intent_accuracy": ("intent", "accuracy"),
            "intent_macro_f1": ("intent", "macro_f1"),
            "base_table_accuracy": ("base_table", "accuracy"),
            "base_table_macro_f1": ("base_table", "macro_f1"),
            "join_decision_macro_f1": ("join_decision", "macro_f1"),
            "router_accuracy": ("router", "accuracy"),
            "router_macro_f1": ("router", "macro_f1"),
        }
        for metric_name, (section, field) in classification_map.items():
            value = (classification.get(section) or {}).get(field, test_summary.get(metric_name))
            if isinstance(value, (int, float)):
                metrics[metric_name] = value
                present.add(metric_name)
        final_execution = test_summary.get("execution_match_rate", test_summary.get("structural_sql_match_rate"))
        if isinstance(final_execution, (int, float)):
            metrics["final_sql_execution_accuracy"] = final_execution
            present.add("final_sql_execution_accuracy")
        calibration = report.get("test_performance", {}).get("calibration", {})
        for name in ["expected_calibration_error", "brier_score", "calibration_sample_count"]:
            source_name = "sample_count" if name == "calibration_sample_count" else name
            if isinstance(calibration.get(source_name), (int, float)):
                metrics[name] = calibration[source_name]
                present.add(name)
        for name in [
            "confidence_unique_value_count",
            "confidence_std",
            "confidence_bucket_coverage_count",
            "calibration_degenerate",
        ]:
            if isinstance(calibration.get(name), (int, float, bool)):
                metrics[name] = calibration[name]
                present.add(name)
        return metrics, present

    @staticmethod
    def _evaluation_source_checks(report: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
        failed: list[dict[str, Any]] = []
        warnings: list[str] = []
        sections = [("generic_model_evaluation_report", report)]
        if isinstance(report.get("test_performance"), dict):
            sections.append(("test_performance", report["test_performance"]))
        if isinstance(report.get("unseen_db_performance"), dict):
            sections.append(("unseen_db_performance", report["unseen_db_performance"]))
        for name, section in sections:
            mode = section.get("evaluation_mode")
            gold_replay = bool(section.get("gold_replay_used", False) or section.get("gold_replay_baseline", False))
            valid = section.get("is_valid_for_quality_gate")
            predictor_used = section.get("predictor_used")
            rows_evaluated = section.get("rows_evaluated", 0)
            real_preds = section.get("real_predictions_generated", 0)
            artifact_source = section.get("model_artifact_source")
            if mode in {"explicit_gold_replay_baseline", "explicit_oracle_upper_bound"} or gold_replay or valid is False:
                failed.append({
                    "metric": f"{name}_valid_evaluation_source",
                    "actual": {
                        "evaluation_mode": mode,
                        "gold_replay_used": gold_replay,
                        "is_valid_for_quality_gate": valid,
                    },
                    "expected": {
                        "evaluation_mode": "real_model_predictions",
                        "gold_replay_used": False,
                        "is_valid_for_quality_gate": True,
                    },
                    "comparison": "==",
                })
            elif mode is None and section is not report:
                warnings.append(f"{name} does not declare evaluation_mode; future gates will require real_model_predictions metadata.")
            if mode == "real_model_predictions" and predictor_used is False:
                failed.append({
                    "metric": f"{name}_predictor_used",
                    "actual": False,
                    "expected": True,
                    "comparison": "==",
                })
            if mode == "real_model_predictions" and isinstance(rows_evaluated, int) and rows_evaluated == 0:
                failed.append({
                    "metric": f"{name}_rows_evaluated",
                    "actual": 0,
                    "expected": "> 0",
                    "comparison": ">",
                })
            if mode == "real_model_predictions" and isinstance(real_preds, int) and real_preds == 0:
                failed.append({
                    "metric": f"{name}_real_predictions_generated",
                    "actual": 0,
                    "expected": "> 0",
                    "comparison": ">",
                })
            # Neural-only fallback warning
            if artifact_source == "neural_only_artifact_dirs":
                warnings.append(
                    f"{name} used neural-only artifact dirs. This may not represent "
                    "full bundle runtime performance and does not load bundle calibration."
                )
        return failed, warnings

    @staticmethod
    def _execution_status(report: dict[str, Any]) -> dict[str, Any]:
        execution = report.get("execution_aware_evaluation")
        if isinstance(execution, dict):
            summary = execution.get("summary") or execution
            available_count = int(summary.get("execution_available", 0) or 0)
            unavailable = bool(summary.get("execution_unavailable", available_count == 0))
            return {
                "enabled": bool(execution.get("enabled", True)),
                "required": bool(execution.get("required", summary.get("execution_required", False))),
                "available": available_count > 0 and not unavailable,
                "execution_available": available_count,
                "unavailable": unavailable,
                "unavailable_reason": summary.get("execution_unavailable_reason") or execution.get("reason") or "not_configured",
                "status": summary.get("execution_status") or (
                    "execution_unavailable" if unavailable else "execution_available_but_failed"
                ),
            }
        return {
            "enabled": "execution_match_rate" in report or "execution_match_rate" in report.get("summary", {}),
            "required": False,
            "available": False,
            "execution_available": 0,
            "unavailable": True,
            "unavailable_reason": "not_configured",
            "status": "execution_unavailable",
        }


def _threshold_metric_name(key: str) -> str:
    if key.endswith("_min"):
        return key[:-4]
    if key.endswith("_max"):
        return key[:-4]
    return key


def _semantic_thresholds(thresholds: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    controlled_mapping = {
        "min_safe_sql_rate": "controlled_predicted_sql_safe_sql_rate_min",
        "min_execution_success_rate": "controlled_predicted_sql_execution_success_rate_min",
        "min_execution_match_rate": "controlled_predicted_sql_execution_match_rate_min",
        "min_row_count_match_rate": "controlled_predicted_sql_row_count_match_rate_min",
        "min_result_value_match_rate": "controlled_predicted_sql_result_value_match_rate_min",
        "max_safe_but_wrong_sql_rate": "controlled_predicted_sql_safe_but_wrong_sql_rate_max",
    }
    for source, target in controlled_mapping.items():
        if source in (thresholds.get("controlled_predicted_sql") or {}):
            values[target] = thresholds["controlled_predicted_sql"][source]
    for source, threshold in (thresholds.get("linking") or {}).items():
        target = source.removeprefix("min_") + "_min" if source.startswith("min_") else source
        values[target] = threshold
    calibration = thresholds.get("calibration") or {}
    if "max_expected_calibration_error" in calibration:
        values["expected_calibration_error_max"] = calibration["max_expected_calibration_error"]
    return values


def _quality_gate_mode(report: dict[str, Any]) -> str:
    mode = str(
        report.get("quality_gate_mode")
        or report.get("mode")
        or report.get("training_mode")
        or ""
    ).lower()
    aliases = {
        "full": "production",
        "smoke": "debug",
        "dev": "debug",
        "development": "debug",
    }
    if mode in {"debug", "baseline", "production", "release", *aliases}:
        return aliases.get(mode, mode)
    pipeline_name = str(report.get("pipeline_name") or report.get("pipeline") or "").lower()
    if "smoke" in pipeline_name:
        return "debug"
    return "baseline"


def _threshold_value_for_mode(threshold: dict[str, Any], mode: str) -> Any:
    if mode in {"production", "release"}:
        return threshold.get("production_min", threshold.get("production_max", threshold.get("min", threshold.get("max"))))
    if mode == "debug":
        return threshold.get("smoke_min", threshold.get("smoke_max", threshold.get("warning_min", threshold.get("warning_max"))))
    return threshold.get("baseline_min", threshold.get("baseline_max", threshold.get("warning_min", threshold.get("warning_max"))))


def _copy_metric(metrics: dict[str, Any], present: set[str], name: str, *sources: dict[str, Any]) -> None:
    if name in metrics:
        present.add(name)
        return
    max_alias = f"{name}_max"
    min_alias = f"{name}_min"
    for source in sources:
        if not isinstance(source, dict):
            continue
        if name in source and isinstance(source[name], (int, float, bool)):
            metrics[name] = source[name]
            present.add(name)
            return
        if max_alias in source and isinstance(source[max_alias], (int, float, bool)):
            metrics[name] = source[max_alias]
            present.add(name)
            return
        if min_alias in source and isinstance(source[min_alias], (int, float, bool)):
            metrics[name] = source[min_alias]
            present.add(name)
            return


def _dedupe_failed_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for check in checks:
        signature = (
            check.get("metric"),
            repr(check.get("actual")),
            repr(check.get("expected")),
            check.get("comparison"),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(check)
    return deduped
