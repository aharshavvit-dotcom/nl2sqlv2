from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from validation.sql_validator import SQLValidator
from training.run_execution_aware_evaluation import evaluate_controlled_predicted_sql


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
    assert result_entry["fixture_execution_allowed"] is False
    assert result_entry["sqlite_execution_success"] is False
    
    # Verify failure breakdown has been updated
    assert "failure_breakdown" in report
    breakdown = report["failure_breakdown"]
    assert breakdown.get("production_sql_validation_failed", 0) == 1
