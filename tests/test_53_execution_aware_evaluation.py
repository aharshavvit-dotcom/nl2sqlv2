from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dataset_training.utils import write_jsonl
from execution_eval.execution_matcher import ExecutionMatcher
from training.run_execution_aware_evaluation import evaluate_controlled_predicted_sql, evaluate_rows


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


def test_controlled_predicted_sql_uses_central_validator_for_valid_select(tmp_path: Path, monkeypatch) -> None:
    _patch_fake_model(monkeypatch, "SELECT id FROM users LIMIT 100")
    report = evaluate_controlled_predicted_sql(
        model_artifact_dir=_artifact_dir(tmp_path),
        fixture_sql_path=_fixture_sql(tmp_path),
        fixture_cases_path=_fixture_cases(tmp_path),
        config={},
    )

    case = report["cases"][0]
    assert report["central_sql_validator_used"] is True
    assert case["sql_validation_passed"] is True
    assert case["safe_sql"] is True
    assert case["predicted_execution_success"] is True


def test_controlled_predicted_sql_blocks_unsafe_before_execution(tmp_path: Path, monkeypatch) -> None:
    _patch_fake_model(monkeypatch, "DROP TABLE users")
    report = evaluate_controlled_predicted_sql(
        model_artifact_dir=_artifact_dir(tmp_path),
        fixture_sql_path=_fixture_sql(tmp_path),
        fixture_cases_path=_fixture_cases(tmp_path),
        config={},
    )

    case = report["cases"][0]
    assert report["central_sql_validator_used"] is True
    assert case["sql_validation_passed"] is False
    assert case["predicted_execution_success"] is False
    assert case["blocked_statement_reason"] in {"non_select_statement", "blocked_keyword"}
    assert report["predicted_unsafe_sql_count"] == 1


def _patch_fake_model(monkeypatch, sql: str) -> None:
    from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel

    class FakeModel:
        def predict(self, question, schema):
            return SimpleNamespace(sql=sql)

    monkeypatch.setattr(RetrievalNL2SQLModel, "load", staticmethod(lambda *_args, **_kwargs: FakeModel()))


def _artifact_dir(tmp_path: Path) -> Path:
    path = tmp_path / "bundle"
    path.mkdir()
    return path


def _fixture_sql(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.sql"
    path.write_text(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"
        "INSERT INTO users (id, name) VALUES (1, 'Ada');",
        encoding="utf-8",
    )
    return path


def _fixture_cases(tmp_path: Path) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"example_id":"c1","question":"list users","gold_sql":"SELECT id FROM users LIMIT 100","expected_row_count":1}\n',
        encoding="utf-8",
    )
    return path
