from __future__ import annotations

from pathlib import Path

from dataset_training.utils import write_jsonl
from execution_eval.execution_matcher import ExecutionMatcher
from training.run_execution_aware_evaluation import evaluate_rows


class FakeConnector:
    def __init__(self):
        self.calls = 0

    def execute(self, sql: str):
        self.calls += 1
        return {"success": True, "columns": ["id"], "rows": [{"id": 1}]}


def test_unsafe_predicted_sql_is_not_executed() -> None:
    connector = FakeConnector()
    result = ExecutionMatcher().evaluate_example(
        "DROP TABLE users",
        "SELECT users.id FROM users LIMIT 100",
        {"tables": {"users": {"columns": {"id": {}}}}},
        connector,
        "sqlite",
    )

    assert result["executed"] is False
    assert connector.calls == 0


def test_predicted_and_gold_sql_results_compared() -> None:
    result = ExecutionMatcher().evaluate_example(
        "SELECT users.id FROM users LIMIT 100",
        "SELECT users.id FROM users LIMIT 100",
        {"tables": {"users": {"columns": {"id": {}}}}},
        FakeConnector(),
        "sqlite",
    )

    assert result["executed"] is True
    assert result["execution_match"] is True


def test_structure_comparison_runs_when_execution_unavailable_and_report_written(tmp_path: Path) -> None:
    rows = [{"example_id": "1", "question": "list users", "predicted_sql": "SELECT users.id FROM users LIMIT 100", "gold_sql": "SELECT users.id FROM users LIMIT 100", "query_ir": {"intent": "show_records"}}]
    report = evaluate_rows(rows)
    output = tmp_path / "predictions.jsonl"
    write_jsonl(output, rows)

    assert report["summary"]["total_examples"] == 1
    assert report["summary"]["structure_match_rate"] == 1.0
