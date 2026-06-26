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
