from __future__ import annotations

from copy import deepcopy

import pytest

from dataset_training.dataset_evaluator import DatasetScaleEvaluator


def test_dataset_scale_evaluator_reports_core_metrics() -> None:
    gold = {"intent": "show_records", "template_id": "show_records", "base_table": "users", "required_tables": ["users"], "joins": [], "metrics": [], "dimensions": [], "filters": [], "date_filters": []}
    bad = deepcopy(gold)
    bad["joins"] = [{"condition": "assignments.user_id = users.id"}]
    row = {
        "example_id": "ex1",
        "dataset_name": "mock",
        "db_id": "db1",
        "complexity": "simple",
        "question": "list users",
        "query_ir": gold,
        "predicted_query_ir": bad,
        "ir_validation": {"is_valid": True},
        "sql_validation": {"is_valid": True},
        "confidence": 0.95,
        "prediction_latency_ms": 12.0,
        "retrieval_scores": [0.9, 0.7],
    }

    report = DatasetScaleEvaluator().evaluate_model("mock_model", [row])

    assert report["summary"]["intent_accuracy_rate"] == 1.0
    assert report["summary"]["join_accuracy_rate"] == 0.0
    assert report["summary"]["unnecessary_join_rate"] == 1.0
    assert report["summary"]["sql_validation_rate"] == 1.0
    assert report["by_intent"]["show_records"]["total_examples"] == 1
    assert report["classification_metrics"]["intent"]["macro_f1"] == 1.0
    assert report["classification_metrics"]["join_decision"]["macro_f1"] == 0.0
    assert report["confusion_matrices"]["join_decision"]["no_join_required"]["unnecessary_join"] == 1
    assert report["percentiles"]["prediction_latency_ms_p95"] == 12.0
    assert report["percentiles"]["retrieval_margin_p50"] == pytest.approx(0.2)
    assert report["calibration"]["sample_count"] == 1
    assert report["evaluation_mode"] == "real_model_predictions"
    assert report["is_valid_for_quality_gate"] is True


def test_dataset_scale_evaluator_refuses_implicit_gold_replay() -> None:
    row = {
        "example_id": "ex1",
        "question": "list users",
        "query_ir": {"intent": "show_records", "base_table": "users", "joins": []},
    }

    with pytest.raises(ValueError, match="requires real predicted_query_ir"):
        DatasetScaleEvaluator().evaluate_model("mock_model", [row])


def test_dataset_scale_evaluator_labels_explicit_gold_replay() -> None:
    row = {
        "example_id": "ex1",
        "question": "list users",
        "query_ir": {"intent": "show_records", "base_table": "users", "joins": []},
    }

    report = DatasetScaleEvaluator().evaluate_model(
        "debug_gold",
        [row],
        evaluation_mode="explicit_gold_replay_baseline",
    )

    assert report["gold_replay_used"] is True
    assert report["is_valid_for_quality_gate"] is False


def test_evaluator_invalid_when_zero_real_predictions() -> None:
    """Even with predicted_query_ir present, if real_predictions_generated sums to 0 the report is invalid."""
    row = {
        "example_id": "ex1",
        "question": "list users",
        "query_ir": {"intent": "show_records", "base_table": "users", "joins": []},
        "predicted_query_ir": {"intent": "show_records", "base_table": "users", "joins": []},
        "ir_validation": {"is_valid": True},
        "sql_validation": {"is_valid": True},
        "confidence": 0.9,
        "prediction_latency_ms": 10.0,
    }

    report = DatasetScaleEvaluator().evaluate_model(
        "mock_model",
        [row],
        evaluation_mode="real_model_predictions",
        predictor_used=False,
    )

    assert report["is_valid_for_quality_gate"] is False


def test_evaluator_invalid_when_predictor_not_used() -> None:
    """When predictor_used=False is explicitly passed, the report must be invalid."""
    row = {
        "example_id": "ex1",
        "question": "list users",
        "query_ir": {"intent": "show_records", "base_table": "users", "joins": []},
        "predicted_query_ir": {"intent": "show_records", "base_table": "users", "joins": []},
        "ir_validation": {"is_valid": True},
        "sql_validation": {"is_valid": True},
        "confidence": 0.95,
        "prediction_latency_ms": 12.0,
    }

    report = DatasetScaleEvaluator().evaluate_model(
        "mock_model",
        [row],
        evaluation_mode="real_model_predictions",
        predictor_used=False,
    )

    assert report["predictor_used"] is False
    assert report["is_valid_for_quality_gate"] is False


def test_evaluator_invalid_when_rows_evaluated_zero() -> None:
    """Empty rows list → is_valid_for_quality_gate must be False."""
    report = DatasetScaleEvaluator().evaluate_model(
        "mock_model",
        [],
        evaluation_mode="real_model_predictions",
        predictor_used=True,
    )

    assert report["rows_evaluated"] == 0
    assert report["is_valid_for_quality_gate"] is False


def test_per_example_contains_bootstrap_promotion_fields() -> None:
    """per_example must contain simple_query_pass, gold_comparison_score, unseen_db_sql_valid
    for bootstrap promotion to work (see promotion_policy.py)."""
    gold = {"intent": "show_records", "base_table": "users", "joins": []}
    row = {
        "example_id": "ex1",
        "question": "list users",
        "query_ir": gold,
        "predicted_query_ir": gold,
        "ir_validation": {"is_valid": True},
        "sql_validation": {"is_valid": True},
        "confidence": 0.95,
        "prediction_latency_ms": 10.0,
    }

    # Gold schema mode → unseen_db_sql_valid should be None
    report = DatasetScaleEvaluator().evaluate_model("mock_model", [row], schema_mode="gold")
    pe = report["per_example"][0]
    assert "simple_query_pass" in pe
    assert "gold_comparison_score" in pe
    assert "unseen_db_sql_valid" in pe
    assert pe["unseen_db_sql_valid"] is None  # Not unseen_db mode

    # Unseen-DB schema mode → unseen_db_sql_valid should be bool
    report_unseen = DatasetScaleEvaluator().evaluate_model("mock_model", [row], schema_mode="unseen_db")
    pe_unseen = report_unseen["per_example"][0]
    assert isinstance(pe_unseen["unseen_db_sql_valid"], bool)
    assert pe_unseen["gold_comparison_score"] >= 0.0
