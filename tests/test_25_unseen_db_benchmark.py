"""
Purpose: Protects ir unit behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from pathlib import Path

from dataset_training.dataset_evaluator import DatasetScaleEvaluator
from dataset_training.reporting import save_report_pair


def test_unseen_db_metrics_and_report_written(tmp_path: Path) -> None:
    gold = {"intent": "show_records", "template_id": "show_records", "base_table": "users", "required_tables": ["users"], "joins": [], "metrics": [], "dimensions": [], "filters": [], "date_filters": []}
    pred = {**gold, "base_table": "assignments", "joins": [{"condition": "assignments.user_id = users.id"}]}
    rows = [{"example_id": "u1", "dataset_name": "mock", "db_id": "unseen_db", "query_ir": gold, "predicted_query_ir": pred, "sql_validation": {"is_valid": True}}]

    report = DatasetScaleEvaluator().evaluate_model("mock", rows, schema_mode="unseen_db")
    save_report_pair(tmp_path / "unseen_db_benchmark_report.json", report, "Unseen DB Benchmark Report")

    assert report["summary"]["wrong_table_rate"] == 1.0
    assert report["summary"]["unnecessary_join_rate"] == 1.0
    assert report["test_source"] == "real_model_predictions"
    assert report["is_valid_for_quality_gate"] is True
    assert (tmp_path / "unseen_db_benchmark_report.json").exists()
    assert (tmp_path / "unseen_db_benchmark_report.md").exists()


def test_gold_replay_benchmark_invalid_for_quality_gate(tmp_path: Path) -> None:
    """GoldReplayBenchmarkRunner must always mark output as invalid for quality gates."""
    from dataset_training.benchmark_runner import GoldReplayBenchmarkRunner

    gold = {"intent": "show_records", "template_id": "show_records", "base_table": "users",
            "required_tables": ["users"], "joins": [], "metrics": [], "dimensions": [],
            "filters": [], "date_filters": []}
    row = {"example_id": "u1", "dataset_name": "mock", "db_id": "db1",
           "question": "list users", "query_ir": gold}
    input_path = tmp_path / "input.jsonl"
    import json
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output_path = tmp_path / "gold_replay_report.json"

    report = GoldReplayBenchmarkRunner().run(input_path, output_path)

    assert report["gold_replay_used"] is True
    assert report["is_valid_for_quality_gate"] is False
    assert report["evaluation_mode"] == "explicit_gold_replay_baseline"
