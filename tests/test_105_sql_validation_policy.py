from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from validation.sql_validator import SQLValidator, policy_failure_type, root_cause_hint
from training.run_execution_aware_evaluation import evaluate_controlled_predicted_sql
from training.evaluate_generic_models import _apply_sql_safety


def test_select_star_blocked_by_validator():
    validator = SQLValidator()
    # SELECT * is blocked by policy
    res = validator.validate("SELECT * FROM users LIMIT 10")
    assert res["checks"]["parse"] is True
    assert res["checks"]["no_select_star"] is False
    assert res["is_valid"] is False


def test_missing_limit_blocked():
    validator = SQLValidator()
    # LIMIT is required by policy
    res = validator.validate("SELECT name FROM users")
    assert res["checks"]["parse"] is True
    assert res["checks"]["limit_present"] is False
    assert res["is_valid"] is False


def test_valid_select_passes():
    validator = SQLValidator()
    res = validator.validate("SELECT name FROM users LIMIT 10")
    assert res["checks"]["parse"] is True
    assert res["checks"]["no_select_star"] is True
    assert res["checks"]["limit_present"] is True
    assert res["is_valid"] is True


def test_controlled_predicted_sql_blocks_unsafe_sql_before_execution(tmp_path, monkeypatch):
    # Setup mock schema SQL and cases JSONL
    sql_path = tmp_path / "schema.sql"
    sql_path.write_text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);", encoding="utf-8")
    
    cases_path = tmp_path / "cases.jsonl"
    case = {
        "example_id": "x1",
        "question": "Show all users",
        "gold_sql": "SELECT name FROM users LIMIT 5",
        "expected_row_count": 0,
    }
    cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    
    # Mock model prediction that returns unsafe SQL "SELECT * FROM users"
    mock_model = MagicMock()
    mock_prediction = MagicMock()
    mock_prediction.sql = "SELECT * FROM users"  # Fails policy (SELECT *)
    mock_model.predict.return_value = mock_prediction
    
    # Mock RetrievalNL2SQLModel.load to return mock_model
    import retriever.retrieval_nl2sql_model as retrieval_nl2sql_model
    monkeypatch.setattr(retrieval_nl2sql_model.RetrievalNL2SQLModel, "load", lambda *args, **kwargs: mock_model)
    
    # Run the evaluation
    report = evaluate_controlled_predicted_sql(
        model_artifact_dir=tmp_path,  # must exist
        fixture_sql_path=sql_path,
        fixture_cases_path=cases_path,
        bundle_id="bundle-123",
        pipeline_run_id="run-456",
        candidate_bundle_dir=str(tmp_path),
    )
    
    assert report["evaluation_type"] == "controlled_predicted_sql_execution"
    assert "cases" in report
    
    case_results = report["cases"]
    assert len(case_results) == 1
    
    result_entry = case_results[0]
    
    # Verify exact policy failure classifications
    assert result_entry["production_sql_valid"] is False
    assert result_entry["blocked_by_production_policy"] is True
    assert "no_select_star" in result_entry["production_policy_blocks"]
    assert result_entry["policy_failure_type"] == "select_star_blocked"
    assert result_entry["failure_category"] == "production_sql_validation_failed"
    assert result_entry["fixture_execution_allowed"] is False
    assert result_entry["sqlite_execution_success"] is False
    
    # Verify failure breakdown has been updated
    assert "failure_breakdown" in report
    breakdown = report["failure_breakdown"]
    assert breakdown.get("production_sql_validation_failed", 0) == 1
    assert report["policy_failure_type_counts"]["select_star_blocked"] == 1


def _evaluate_prediction(tmp_path, monkeypatch, predicted_sql):
    sql_path = tmp_path / "schema.sql"
    sql_path.write_text(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"
        "INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob');",
        encoding="utf-8",
    )
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps({
        "case_id": "policy_case",
        "example_id": "x1",
        "question": "List users",
        "gold_sql": "SELECT id, name FROM users ORDER BY id LIMIT 10",
        "expected_row_count": 2,
    }) + "\n", encoding="utf-8")
    model = MagicMock()
    model.predict.return_value.sql = predicted_sql
    import retriever.retrieval_nl2sql_model as retrieval_nl2sql_model
    monkeypatch.setattr(
        retrieval_nl2sql_model.RetrievalNL2SQLModel,
        "load",
        lambda *_args, **_kwargs: model,
    )
    return evaluate_controlled_predicted_sql(
        model_artifact_dir=tmp_path,
        fixture_sql_path=sql_path,
        fixture_cases_path=cases_path,
    )


def test_missing_limit_maps_to_limit_policy_failed(tmp_path, monkeypatch):
    report = _evaluate_prediction(tmp_path, monkeypatch, "SELECT id, name FROM users")
    case = report["cases"][0]
    assert case["policy_failure_type"] == "limit_policy_failed"
    assert report["policy_failure_type_counts"]["limit_policy_failed"] == 1


@pytest.mark.parametrize("sql", [
    "DELETE FROM users WHERE id = 1",
    "UPDATE users SET name = 'X' WHERE id = 1",
    "INSERT INTO users (id, name) VALUES (3, 'X')",
])
def test_mutating_statement_maps_to_non_select(sql):
    validation = SQLValidator().validate(sql)
    assert policy_failure_type(validation) == "non_select_statement"


def test_sqlite_execution_error_has_no_policy_failure_type(tmp_path, monkeypatch):
    report = _evaluate_prediction(
        tmp_path,
        monkeypatch,
        "SELECT unknown_runtime_function(id) AS value FROM users LIMIT 10",
    )
    case = report["cases"][0]
    assert case["production_sql_valid"] is True
    assert case["sqlite_execution_success"] is False
    assert case["policy_failure_type"] is None
    assert case["failure_category"] == "sqlite_execution_error"


def test_value_mismatch_has_no_policy_failure_type(tmp_path, monkeypatch):
    report = _evaluate_prediction(
        tmp_path,
        monkeypatch,
        "SELECT id, name FROM users WHERE id = 1 LIMIT 10",
    )
    case = report["cases"][0]
    assert case["sqlite_execution_success"] is True
    assert case["final_execution_match"] is False
    assert case["policy_failure_type"] is None


def test_missing_limit_is_repaired_and_revalidated():
    result = SQLValidator().validate_with_repair("SELECT id FROM users")
    assert result["repair_attempted"] is True
    assert result["repair_succeeded"] is True
    assert "LIMIT 100" in result["final_sql"].upper()
    assert result["final_validation"]["is_valid"] is True


def test_select_star_repair_requires_schema_columns():
    validator = SQLValidator()
    schema = {"tables": {"users": {"columns": {
        "id": {"type": "integer"},
        "name": {"type": "text"},
    }}}}
    repaired = validator.validate_with_repair("SELECT * FROM users", schema=schema)
    assert repaired["repair_succeeded"] is True
    assert "*" not in repaired["final_sql"]
    assert "id" in repaired["final_sql"] and "name" in repaired["final_sql"]

    unavailable = validator.validate_with_repair("SELECT * FROM users", schema=None)
    assert unavailable["repair_succeeded"] is False
    assert unavailable["final_sql"] is None


def test_unsafe_dml_is_never_repaired():
    result = SQLValidator().validate_with_repair("DELETE FROM users WHERE id = 1")
    assert result["repair_attempted"] is False
    assert result["repair_succeeded"] is False
    assert result["final_sql"] is None


@pytest.mark.parametrize("alias", ["Home (1st leg)", "#", "1st Leg", "Country/Region", "Rd."])
def test_invalid_display_alias_is_quoted_then_revalidated(alias):
    sql = f'SELECT "t"."value" AS {alias} FROM "t" LIMIT 10'
    result = SQLValidator().validate_with_repair(
        sql,
        schema={"tables": {"t": {"columns": {"value": {"type": "text"}}}}},
    )
    assert result["repair_attempted"] is True
    assert result["repair_succeeded"] is True
    assert f'AS "{alias}"' in result["final_sql"]
    assert result["final_validation"]["is_valid"] is True


def test_blocked_word_inside_quoted_identifier_is_not_unsafe():
    result = SQLValidator().validate('SELECT "drop" FROM "events" LIMIT 10')
    assert result["checks"]["no_blocked_keywords"] is True
    assert result["is_valid"] is True


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ('SELECT "t"."x" AS Home (1st leg) FROM "t" LIMIT 10', "unquoted_alias"),
        ('SELECT "t".Rd. FROM "t" LIMIT 10', "malformed_identifier"),
        ('DROP TABLE users', "unsafe_keyword"),
        ('SELECT ( FROM "t" LIMIT 10', "parse_error"),
        ('SELECT "t".name FROM "t"', "unknown"),
    ],
)
def test_sql_failure_root_cause_hints(sql, expected):
    validation = SQLValidator().validate(sql)
    assert root_cause_hint(sql, validation) == expected


def test_generic_sql_safety_counts_failures_and_abstains(tmp_path):
    rows = [
        {
            "example_id": "repairable",
            "question": "list ids",
            "predicted_sql": "SELECT id FROM users",
            "schema": {"tables": {"users": {"columns": {"id": {"type": "integer"}}}}},
            "ir_validation": {"is_valid": True},
        },
        {
            "example_id": "unsafe",
            "question": "delete users",
            "predicted_sql": "DELETE FROM users",
            "schema": {"tables": {"users": {"columns": {"id": {"type": "integer"}}}}},
            "ir_validation": {"is_valid": True},
        },
    ]
    summary = _apply_sql_safety(rows)
    assert summary["repair_attempt_count"] == 1
    assert summary["repair_success_count"] == 1
    assert summary["failure_breakdown"]["non_select_statement"] == 1
    assert summary["invalid_sql_count"] == 1
    assert summary["unsafe_sql_abstention_count"] == 1
    assert summary["post_abstention_unsafe_sql_count"] == 0
    assert rows[1]["predicted_sql"] is None
    assert rows[1]["abstention_reason"] == "unsafe_sql"
    assert summary["failures"][0]["invalid_sql"] is True
    assert summary["failures"][0]["unsafe_sql"] is True
    from dataset_training.utils import write_jsonl
    output = tmp_path / "unsafe_sql_examples.jsonl"
    write_jsonl(output, summary["failures"])
    saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert saved[0]["policy_failure_type"] == "non_select_statement"
