from __future__ import annotations

from copy import deepcopy

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
    }

    report = DatasetScaleEvaluator().evaluate_model("mock_model", [row])

    assert report["summary"]["intent_accuracy_rate"] == 1.0
    assert report["summary"]["join_accuracy_rate"] == 0.0
    assert report["summary"]["unnecessary_join_rate"] == 1.0
    assert report["summary"]["sql_validation_rate"] == 1.0
    assert report["by_intent"]["show_records"]["total_examples"] == 1
