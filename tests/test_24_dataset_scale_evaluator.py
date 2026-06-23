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
