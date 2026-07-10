from __future__ import annotations

from quality_gates.model_quality_gate import ModelQualityGate


THRESHOLDS = {
    "minimums": {
        "query_ir_validity_rate": 0.90,
        "sql_validation_rate": 0.90,
        "simple_query_pass_rate": 0.95,
        "no_select_star_rate": 1.00,
        "unsafe_sql_count_max": 0,
        "unnecessary_join_rate_max": 0.05,
        "wrong_table_rate_max": 0.15,
        "unseen_db_sql_validation_rate": 0.80,
        "feedback_regression_pass_rate": 0.95,
    }
}


def test_quality_gate_fails_bad_metrics() -> None:
    report = {
        "test_performance": {"summary": {"query_ir_validity_rate": 0.5, "sql_validation_rate": 0.5, "intent_accuracy_rate": 0.5, "unnecessary_join_rate": 0.2, "wrong_table_rate": 0.3}},
        "unseen_db_performance": {"summary": {"sql_validation_rate": 0.4}},
        "no_select_star_rate": 0.8,
        "unsafe_sql_count": 1,
        "feedback_regression_pass_rate": 0.5,
    }

    result = ModelQualityGate().evaluate(report, THRESHOLDS)

    assert result["passed"] is False
    assert result["failed_checks"]


def test_quality_gate_passes_good_metrics() -> None:
    report = {
        "test_performance": {"summary": {"query_ir_validity_rate": 0.99, "sql_validation_rate": 0.99, "intent_accuracy_rate": 0.98, "unnecessary_join_rate": 0.0, "wrong_table_rate": 0.0}},
        "unseen_db_performance": {"summary": {"sql_validation_rate": 0.95}},
        "no_select_star_rate": 1.0,
        "unsafe_sql_count": 0,
        "feedback_regression_pass_rate": 1.0,
        "dataset_contribution_report_required": True,
        "dataset_contribution_report": {
            "datasets_requested": ["wikisql"],
            "leakage_check_passed": True,
            "full_training_dataset_minimums_passed": True,
            "by_dataset": {"wikisql": {"converted_to_queryir": 10}},
        },
    }

    result = ModelQualityGate().evaluate(report, THRESHOLDS)

    assert result["passed"] is True


def test_production_simple_query_threshold_is_blocking() -> None:
    report = {
        "quality_gate_mode": "production",
        "test_performance": {
            "summary": {
                "query_ir_validity_rate": 1.0,
                "sql_validation_rate": 1.0,
                "simple_query_pass_rate": 0.90,
                "unnecessary_join_rate": 0.0,
                "wrong_table_rate": 0.0,
                "unsafe_sql_count": 0,
            }
        },
        "unseen_db_performance": {"summary": {"sql_validation_rate": 1.0}},
        "no_select_star_rate": 1.0,
        "unsafe_sql_count": 0,
        "feedback_regression_pass_rate": 1.0,
    }

    result = ModelQualityGate().evaluate(report, THRESHOLDS)

    assert result["passed"] is False
    assert any(check["metric"] == "simple_query_pass_rate" for check in result["failed_checks"])


def test_production_missing_simple_query_does_not_use_intent_accuracy() -> None:
    report = {
        "quality_gate_mode": "production",
        "test_performance": {
            "summary": {
                "query_ir_validity_rate": 1.0,
                "sql_validation_rate": 1.0,
                "intent_accuracy_rate": 1.0,
                "unnecessary_join_rate": 0.0,
                "wrong_table_rate": 0.0,
                "unsafe_sql_count": 0,
            }
        },
        "unseen_db_performance": {"summary": {"sql_validation_rate": 1.0}},
        "no_select_star_rate": 1.0,
        "unsafe_sql_count": 0,
        "feedback_regression_pass_rate": 1.0,
    }

    result = ModelQualityGate().evaluate(report, THRESHOLDS)

    assert result["passed"] is False
    assert "simple_query_pass_rate" in result["missing_metrics"]


def test_smoke_simple_query_uses_flat_080_threshold() -> None:
    report = {
        "quality_gate_mode": "smoke",
        "test_performance": {
            "summary": {
                "query_ir_validity_rate": 1.0,
                "sql_validation_rate": 1.0,
                "simple_query_pass_rate": 0.81,
                "unnecessary_join_rate": 0.0,
                "wrong_table_rate": 0.0,
                "unsafe_sql_count": 0,
            }
        },
        "unseen_db_performance": {"summary": {"sql_validation_rate": 1.0}},
        "no_select_star_rate": 1.0,
        "unsafe_sql_count": 0,
        "feedback_regression_pass_rate": 1.0,
    }
    thresholds = {"minimums": {**THRESHOLDS["minimums"], "simple_query_pass_rate": 0.80}}

    result = ModelQualityGate().evaluate(report, thresholds)

    assert result["passed"] is True


def test_quality_gate_fails_missing_critical_metric() -> None:
    report = {
        "test_performance": {"summary": {"query_ir_validity_rate": 0.99, "sql_validation_rate": 0.99}},
        "no_select_star_rate": 1.0,
        "feedback_regression_pass_rate": 1.0,
    }

    result = ModelQualityGate().evaluate(report, THRESHOLDS)

    assert result["passed"] is False
    assert "unsafe_sql_count" in result["missing_metrics"]


def test_production_blocks_safe_but_semantically_wrong_sql_and_degenerate_confidence() -> None:
    report = {
        "quality_gate_mode": "production",
        "test_performance": {
            "summary": {
                "query_ir_validity_rate": 1.0,
                "sql_validation_rate": 1.0,
                "simple_query_pass_rate": 1.0,
                "unnecessary_join_rate": 0.0,
                "wrong_table_rate": 0.0,
                "unsafe_sql_count": 0,
                "filter_column_accuracy_rate": 0.9,
                "filter_value_accuracy_rate": 0.9,
                "dimension_column_accuracy_rate": 0.9,
            },
            "calibration": {
                "expected_calibration_error": 0.02,
                "calibration_degenerate": True,
            },
        },
        "no_select_star_rate": 1.0,
        "unsafe_sql_count": 0,
        "controlled_predicted_sql_execution": {
            "predicted_safe_sql_rate": 1.0,
            "predicted_execution_success_rate": 1.0,
            "predicted_execution_match_rate": 0.2,
            "predicted_row_count_match_rate": 0.5,
            "predicted_result_value_match_rate": 0.2,
            "safe_but_wrong_sql_rate": 0.8,
            "central_sql_validator_used": True,
            "passed": False,
        },
    }
    thresholds = {
        "minimums": THRESHOLDS["minimums"],
        "controlled_predicted_sql": {
            "min_execution_match_rate": {"production_min": 0.6},
            "max_safe_but_wrong_sql_rate": {"production_max": 0.4},
        },
        "calibration": {"require_non_degenerate_confidence": {"production": True}},
    }

    result = ModelQualityGate().evaluate(report, thresholds)
    failed = {item["metric"] for item in result["failed_checks"]}

    assert "controlled_predicted_sql_execution_match_rate_min" in failed
    assert "controlled_predicted_sql_safe_but_wrong_sql_rate_max" in failed
    assert "calibration_degenerate" in failed


def test_quality_gate_rejects_gold_replay_report() -> None:
    report = {
        "evaluation_mode": "explicit_gold_replay_baseline",
        "gold_replay_used": True,
        "is_valid_for_quality_gate": False,
        "test_performance": {
            "evaluation_mode": "explicit_gold_replay_baseline",
            "gold_replay_used": True,
            "is_valid_for_quality_gate": False,
            "summary": {
                "query_ir_validity_rate": 1.0,
                "sql_validation_rate": 1.0,
                "intent_accuracy_rate": 1.0,
                "unnecessary_join_rate": 0.0,
                "wrong_table_rate": 0.0,
                "unsafe_sql_count": 0,
            },
        },
        "unseen_db_performance": {"summary": {"sql_validation_rate": 1.0}},
        "no_select_star_rate": 1.0,
        "unsafe_sql_count": 0,
        "feedback_regression_pass_rate": 1.0,
    }

    result = ModelQualityGate().evaluate(report, THRESHOLDS)

    assert result["passed"] is False
    assert any(check["metric"].endswith("valid_evaluation_source") for check in result["failed_checks"])


def test_quality_gate_rejects_zero_predictions() -> None:
    """Quality gate must reject a report with real_predictions_generated=0."""
    report = {
        "evaluation_mode": "real_model_predictions",
        "gold_replay_used": False,
        "predictor_used": True,
        "is_valid_for_quality_gate": True,
        "rows_evaluated": 0,
        "real_predictions_generated": 0,
        "test_performance": {
            "evaluation_mode": "real_model_predictions",
            "gold_replay_used": False,
            "predictor_used": True,
            "is_valid_for_quality_gate": False,
            "rows_evaluated": 0,
            "real_predictions_generated": 0,
            "summary": {
                "query_ir_validity_rate": 0.0,
                "sql_validation_rate": 0.0,
                "intent_accuracy_rate": 0.0,
                "unnecessary_join_rate": 0.0,
                "wrong_table_rate": 0.0,
                "unsafe_sql_count": 0,
            },
        },
        "unseen_db_performance": {"summary": {"sql_validation_rate": 0.0}},
        "no_select_star_rate": 1.0,
        "unsafe_sql_count": 0,
        "feedback_regression_pass_rate": 1.0,
    }

    result = ModelQualityGate().evaluate(report, THRESHOLDS)

    assert result["passed"] is False
    failed_metrics = [check["metric"] for check in result["failed_checks"]]
    assert any("real_predictions_generated" in m or "rows_evaluated" in m or "valid_evaluation_source" in m for m in failed_metrics)


def test_quality_gate_rejects_predictor_not_used() -> None:
    """Quality gate must reject a report with predictor_used=False."""
    report = {
        "evaluation_mode": "real_model_predictions",
        "gold_replay_used": False,
        "predictor_used": False,
        "is_valid_for_quality_gate": False,
        "rows_evaluated": 10,
        "real_predictions_generated": 10,
        "test_performance": {
            "evaluation_mode": "real_model_predictions",
            "gold_replay_used": False,
            "predictor_used": False,
            "is_valid_for_quality_gate": False,
            "rows_evaluated": 10,
            "real_predictions_generated": 10,
            "summary": {
                "query_ir_validity_rate": 0.99,
                "sql_validation_rate": 0.99,
                "intent_accuracy_rate": 0.98,
                "unnecessary_join_rate": 0.0,
                "wrong_table_rate": 0.0,
                "unsafe_sql_count": 0,
            },
        },
        "unseen_db_performance": {"summary": {"sql_validation_rate": 0.95}},
        "no_select_star_rate": 1.0,
        "unsafe_sql_count": 0,
        "feedback_regression_pass_rate": 1.0,
    }

    result = ModelQualityGate().evaluate(report, THRESHOLDS)

    assert result["passed"] is False
    failed_metrics = [check["metric"] for check in result["failed_checks"]]
    assert any("predictor_used" in m or "valid_evaluation_source" in m for m in failed_metrics)


def _mode_report(mode: str) -> dict:
    return {
        "quality_gate_mode": mode,
        "test_performance": {"summary": {
            "query_ir_validity_rate": 1.0,
            "sql_validation_rate": 1.0,
            "simple_query_pass_rate": 1.0,
            "unnecessary_join_rate": 0.0,
            "wrong_table_rate": 0.0,
            "unsafe_sql_count": 0,
        }},
        "unseen_db_performance": {"summary": {"sql_validation_rate": 1.0}},
        "no_select_star_rate": 1.0,
        "unsafe_sql_count": 0,
    }


def test_execution_unavailable_is_warning_in_baseline() -> None:
    report = _mode_report("baseline")
    report["execution_aware_evaluation"] = {
        "enabled": True,
        "required": True,
        "summary": {
            "execution_available": 0,
            "execution_unavailable": True,
            "execution_unavailable_reason": "no_database_connection",
            "execution_match_rate": None,
        },
    }
    thresholds = {"minimums": {
        **THRESHOLDS["minimums"],
        "execution_match_rate_min": 0.60,
    }}
    result = ModelQualityGate().evaluate(report, thresholds)
    assert not any(item["metric"] == "execution_unavailable" for item in result["failed_checks"])
    assert any("execution_unavailable" in warning for warning in result["warnings"])


def test_execution_unavailable_blocks_production_when_required() -> None:
    report = _mode_report("production")
    report["feedback_regression"] = {"enabled": False, "required_for_production": False}
    report["execution_aware_evaluation"] = {
        "enabled": True,
        "required": True,
        "summary": {
            "execution_available": 0,
            "execution_unavailable": True,
            "execution_unavailable_reason": "no_database_connection",
            "execution_match_rate": None,
        },
    }
    thresholds = {"minimums": {
        **THRESHOLDS["minimums"],
        "execution_match_rate_min": 0.60,
    }}
    result = ModelQualityGate().evaluate(report, thresholds)
    assert any(item["metric"] == "execution_unavailable" for item in result["failed_checks"])


def test_execution_unavailable_failure_is_deduplicated() -> None:
    report = _mode_report("production")
    report["feedback_regression"] = {"enabled": False, "required_for_production": False}
    report["execution_aware_evaluation"] = {
        "enabled": True,
        "required": True,
        "summary": {
            "execution_available": 0,
            "execution_unavailable": True,
            "execution_unavailable_reason": "no_database_connection",
        },
    }
    thresholds = {
        "minimums": {
            **THRESHOLDS["minimums"],
            "execution_match_rate_min": 0.60,
        },
        "classification_metrics": {
            "final_sql_execution_accuracy_min": {"production_min": 0.70},
        },
    }

    result = ModelQualityGate().evaluate(report, thresholds)

    assert sum(item["metric"] == "execution_unavailable" for item in result["failed_checks"]) == 1


def test_missing_feedback_blocks_only_when_production_required() -> None:
    thresholds = {"minimums": {**THRESHOLDS["minimums"]}}
    production = _mode_report("production")
    production["feedback_regression"] = {
        "enabled": True,
        "required_for_production": True,
    }
    blocked = ModelQualityGate().evaluate(production, thresholds)
    assert any(item["metric"] == "feedback_regression_pass_rate" for item in blocked["failed_checks"])

    debug = _mode_report("debug")
    debug["feedback_regression"] = {
        "enabled": True,
        "required_for_production": True,
    }
    advisory = ModelQualityGate().evaluate(debug, thresholds)
    assert not any(item["metric"] == "feedback_regression_pass_rate" for item in advisory["failed_checks"])
    assert any("feedback_regression" in warning for warning in advisory["warnings"])


def test_feedback_report_metric_contributes_when_present() -> None:
    report = _mode_report("production")
    report["feedback_regression"] = {"enabled": True, "required_for_production": True}
    report["feedback_regression_pass_rate"] = 1.0
    result = ModelQualityGate().evaluate(report, THRESHOLDS)
    assert result["metrics"]["feedback_regression_pass_rate"] == 1.0


def test_controlled_predicted_sql_required_is_policy_not_metric() -> None:
    report = _mode_report("production")
    report["feedback_regression"] = {"enabled": False, "required_for_production": False}
    thresholds = {
        "minimums": {
            **THRESHOLDS["minimums"],
            "controlled_predicted_sql_required": False,
        }
    }

    result = ModelQualityGate().evaluate(report, thresholds)

    failed = {item["metric"] for item in result["failed_checks"]}
    assert "controlled_predicted_sql_required" not in failed
    assert "controlled_predicted_sql_required" not in result["missing_metrics"]
    assert result["passed"] is True


def test_required_controlled_predicted_sql_blocks_zero_prediction_report() -> None:
    report = _mode_report("production")
    report["feedback_regression"] = {"enabled": False, "required_for_production": False}
    report["controlled_predicted_sql_required"] = True
    report["controlled_predicted_sql_execution"] = {
        "passed": False,
        "cases_total": 12,
        "predictions_generated": 0,
        "abstention_count": 12,
        "predicted_execution_match_rate": 0.0,
        "predicted_unsafe_sql_count": 0,
        "central_sql_validator_used": True,
    }

    result = ModelQualityGate().evaluate(report, THRESHOLDS)

    failure = next(
        item for item in result["failed_checks"]
        if item["metric"] == "controlled_predicted_sql_passed"
    )
    assert failure["actual"]["predictions_generated"] == 0
    assert failure["actual"]["abstention_count"] == 12
