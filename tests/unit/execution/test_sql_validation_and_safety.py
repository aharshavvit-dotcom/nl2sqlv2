"""
Purpose: Verifies execution safety behaviour consolidated from fragmented test files.
Required because: SQL validation, read-only policy, attribution, cache identity and telemetry privacy are safety contracts.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.safety


# Source: tests/test_02_sql_validation.py
"""Test 02: SQL Validation — SQLValidator, safe preview, dialect handling."""


from validation.sql_validator import SQLValidator
from app.safe_preview import build_safe_preview_sql


class TestSQLValidator:
    def test_rejects_sensitive_column(self) -> None:
        schema = {"tables": {"customers": {"columns": {"customer_id": {}, "email": {}}}}}
        result = SQLValidator().validate("SELECT customers.email FROM customers LIMIT 10", schema=schema)
        assert not result["is_valid"]
        assert not result["checks"]["no_sensitive_columns"]

    def test_accepts_valid_select(self) -> None:
        schema = {"tables": {"orders": {"columns": {"order_id": {}, "amount": {}}}}}
        result = SQLValidator().validate("SELECT order_id, amount FROM orders LIMIT 10", schema=schema)
        assert result["is_valid"]

    def test_rejects_insert(self) -> None:
        result = SQLValidator().validate("INSERT INTO orders VALUES (1, 100)")
        assert not result["is_valid"]

    def test_rejects_drop(self) -> None:
        result = SQLValidator().validate("DROP TABLE orders")
        assert not result["is_valid"]

    def test_rejects_delete(self) -> None:
        result = SQLValidator().validate("DELETE FROM orders WHERE order_id = 1")
        assert not result["is_valid"]

    def test_sqlite_dialect(self) -> None:
        result = SQLValidator().validate("SELECT order_id FROM orders LIMIT 10", dialect="sqlite")
        assert result["is_valid"]

    def test_postgres_dialect(self) -> None:
        result = SQLValidator().validate("SELECT order_id FROM orders LIMIT 10", dialect="postgres")
        assert result["is_valid"]

    def test_rejects_dob_sensitive_column(self) -> None:
        schema = {"tables": {"users": {"columns": {"user_id": {}, "dob": {}}}}}
        result = SQLValidator().validate("SELECT users.dob FROM users LIMIT 10", schema=schema)
        assert not result["is_valid"]

    def test_rejects_credit_card_column(self) -> None:
        schema = {"tables": {"payments": {"columns": {"id": {}, "credit_card": {}}}}}
        result = SQLValidator().validate("SELECT payments.credit_card FROM payments LIMIT 10", schema=schema)
        assert not result["is_valid"]


class TestSafePreviewSQL:
    def test_builds_safe_preview(self) -> None:
        schema = {"tables": {"orders": {"columns": {"order_id": {}, "amount": {}}}}}
        sql = build_safe_preview_sql("orders", schema)
        assert sql is not None
        assert "order_id" in sql
        assert "LIMIT" in sql

    def test_excludes_sensitive_columns(self) -> None:
        schema = {"tables": {"users": {"columns": {"user_id": {}, "email": {}, "phone": {}}}}}
        sql = build_safe_preview_sql("users", schema)
        assert sql is not None
        assert "email" not in sql
        assert "phone" not in sql


# Source: tests/test_105_sql_validation_policy.py
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


# Source: tests/test_119_renderer_attribution.py
"""Tests for Stage 2 Renderer Attribution and Failure Attribution boundaries.

Validates the controlled control experiment and hierarchical stage failure checks.
"""


from unittest.mock import MagicMock, patch
import pytest

from evaluation.route_diagnostics import (
    attribute_failure_stage,
    run_renderer_control,
)


def test_native_query_ir_failure_attributed_to_route_generation() -> None:
    # 1. Native QueryIR doesn't match gold
    gold_ir = {"intent": "metric_summary", "base_table": "orders"}
    pred_res = MagicMock()
    # boundaries has wrong native query ir
    pred_res.debug = {
        "boundaries": {
            "native_query_ir": {"intent": "wrong_intent"},
            "resolved_query_ir": {"intent": "wrong_intent"},
            "validated_query_ir": {"intent": "wrong_intent"},
        }
    }
    
    stage = attribute_failure_stage("retrieval", gold_ir, pred_res, True, None)
    assert stage == "query_ir_semantic_failure"


def test_correct_native_ir_corrupted_by_slot_resolver() -> None:
    # 2. Native matched gold, but resolved did not (meaning slot resolution corrupted it)
    gold_ir = {"intent": "metric_summary", "base_table": "orders"}
    pred_res = MagicMock()
    pred_res.debug = {
        "boundaries": {
            "native_query_ir": {"intent": "metric_summary", "base_table": "orders"},
            "resolved_query_ir": {"intent": "wrong_intent"},
            "validated_query_ir": {"intent": "wrong_intent"},
        }
    }
    
    stage = attribute_failure_stage("retrieval", gold_ir, pred_res, True, None)
    assert stage == "slot_resolution_failure"


def test_correct_final_ir_invalid_sql_attributed_to_renderer() -> None:
    # 3. Final QueryIR correct, but rendered_sql is None or validation fails
    gold_ir = {"intent": "metric_summary", "base_table": "orders"}
    pred_res = MagicMock()
    pred_res.debug = {
        "boundaries": {
            "native_query_ir": {"intent": "metric_summary", "base_table": "orders"},
            "resolved_query_ir": {"intent": "metric_summary", "base_table": "orders"},
            "validated_query_ir": {"intent": "metric_summary", "base_table": "orders"},
            "rendered_sql": "SELECT * FROM orders",
            "sql_validation": {"is_valid": False, "issues": ["syntax error"]},
        }
    }
    
    stage = attribute_failure_stage("retrieval", gold_ir, pred_res, True, None)
    assert stage == "sql_validation_failure"


def test_valid_sql_execution_failure_attributed_to_database() -> None:
    # 4. Valid SQL validation, but database execution fails
    gold_ir = {"intent": "metric_summary", "base_table": "orders"}
    pred_res = MagicMock()
    pred_res.debug = {
        "boundaries": {
            "native_query_ir": {"intent": "metric_summary", "base_table": "orders"},
            "resolved_query_ir": {"intent": "metric_summary", "base_table": "orders"},
            "validated_query_ir": {"intent": "metric_summary", "base_table": "orders"},
            "rendered_sql": "SELECT amount FROM orders",
            "sql_validation": {"is_valid": True},
            "execution_result": {"failed": True, "error": "table doesn't exist"},
        }
    }
    
    stage = attribute_failure_stage("retrieval", gold_ir, pred_res, True, None)
    assert stage == "database_execution_failure"


def test_valid_executed_sql_wrong_results_attributed_to_semantic_execution() -> None:
    # 5. Executed successfully but result structure or values do not match gold
    gold_ir = {"intent": "metric_summary", "base_table": "orders"}
    pred_res = MagicMock()
    pred_res.debug = {
        "boundaries": {
            "native_query_ir": {"intent": "metric_summary", "base_table": "orders"},
            "resolved_query_ir": {"intent": "metric_summary", "base_table": "orders"},
            "validated_query_ir": {"intent": "metric_summary", "base_table": "orders"},
            "rendered_sql": "SELECT amount FROM orders",
            "sql_validation": {"is_valid": True},
            "execution_result": {"failed": False, "value_mismatch": True},
        }
    }
    
    stage = attribute_failure_stage("retrieval", gold_ir, pred_res, True, None)
    assert stage == "result_semantic_mismatch"


def test_renderer_control_experiment_dialect_integration_failure() -> None:
    # Gold IR fails and Predicted IR fails -> dialect_integration_failure
    model = MagicMock()
    model.orchestrator.sql_renderer.dialect = "sqlite"
    
    # Mocking gold & predicted rendering to raise exception
    model.orchestrator.sql_renderer.render.side_effect = Exception("syntax error in renderer")
    
    pred_res = MagicMock()
    pred_res.query_ir = {"intent": "metric_summary", "base_table": "orders"}
    gold_ir = {"intent": "metric_summary", "base_table": "orders"}
    
    ctrl = run_renderer_control("1", "question?", gold_ir, pred_res, model, {})
    assert ctrl.meaning == "dialect_integration_failure"
    assert ctrl.failure_stage == "schema_or_dialect_failure"


def test_renderer_control_experiment_renderer_edge_case() -> None:
    # Gold IR succeeds rendering but Predicted IR fails rendering -> renderer_edge_case
    model = MagicMock()
    model.orchestrator.sql_renderer.dialect = "sqlite"
    
    def render_side_effect(q_ir, dialect):
        if q_ir.get("intent") == "gold":
            return "SELECT * FROM orders"
        raise Exception("failed to render predicted IR")
        
    model.orchestrator.sql_renderer.render.side_effect = render_side_effect
    
    # We mock validator to say gold SQL is valid, but predicted SQL render raises exception
    model.orchestrator.sql_validator.validate_with_repair.return_value = {
        "final_validation": {"is_valid": True}
    }
    
    # QueryIR matches gold semantically (our diff returns no mismatch), but pred fails to render
    # To simulate matching QueryIR, we mock diff_query_ir to return all true
    with patch("evaluation.route_diagnostics.diff_query_ir", return_value={"intent_match": True, "base_table_match": True}):
        pred_res = MagicMock()
        pred_res.query_ir = {"intent": "pred"}
        gold_ir = {"intent": "gold"}
        
        ctrl = run_renderer_control("1", "question?", gold_ir, pred_res, model, {})
        assert ctrl.meaning == "renderer_edge_case"
        assert ctrl.failure_stage == "renderer_generation_failure"


# Source: tests/test_132_sqlite_prediction_cache.py
"""Unit and concurrency tests for the SQLite-backed prediction cache."""


import json
import sqlite3
import threading
import time
from pathlib import Path
import pytest

from inference.prediction_cache import PredictionCache


@pytest.fixture
def cache_db(tmp_path):
    db_path = tmp_path / "cache.db"
    return PredictionCache(cache_path=db_path, max_entries=3, ttl_days=1)


def test_cache_key_partitioning(cache_db):
    question = "show sales above 500"
    schema = {"schema_fingerprint": "sch_123", "tenant_id": "tenant_A"}
    
    # Base key
    key1 = cache_db.generate_hash_key(question, schema, None, None)
    
    # Different tenant_id
    schema_diff_tenant = {"schema_fingerprint": "sch_123", "tenant_id": "tenant_B"}
    key2 = cache_db.generate_hash_key(question, schema_diff_tenant, None, None)
    assert key1 != key2
    
    # Different schema fingerprint
    schema_diff_sch = {"schema_fingerprint": "sch_456", "tenant_id": "tenant_A"}
    key3 = cache_db.generate_hash_key(question, schema_diff_sch, None, None)
    assert key1 != key3


def test_cache_put_and_get(cache_db):
    question = "show transactions"
    schema = {"schema_fingerprint": "sch_123", "tenant_id": "tenant_A"}
    
    prediction = {
        "status": "completed",
        "query_ir": {
            "intent": "show_records",
            "filters": [{"column": "amount", "value": 500}]
        },
        "validation": {"is_valid": True},
        "sql": "SELECT * FROM transactions WHERE amount > 500",
        "confidence": 0.9,
    }
    
    cache_db.put(question, schema, None, prediction)
    
    cached = cache_db.get(question, schema, None)
    assert cached is not None
    assert cached["status"] == "completed"
    # Question text should not be present in retrieved cached prediction
    assert "question" not in cached
    # Filter values should be redacted in cache query_ir
    assert cached["query_ir"]["filters"][0]["value"] == "[REDACTED]"


def test_cache_evicts_lru_and_enforces_capacity(cache_db):
    schema = {"schema_fingerprint": "sch_123", "tenant_id": "tenant_A"}
    
    prediction = {
        "status": "completed",
        "query_ir": {"intent": "show"},
        "validation": {"is_valid": True},
    }
    
    # Put 4 items (capacity is 3)
    cache_db.put("q1", schema, None, prediction)
    time.sleep(0.01)
    cache_db.put("q2", schema, None, prediction)
    time.sleep(0.01)
    cache_db.put("q3", schema, None, prediction)
    time.sleep(0.01)
    cache_db.put("q4", schema, None, prediction)
    
    # q1 (the oldest) should be evicted
    assert cache_db.get("q1", schema, None) is None
    assert cache_db.get("q2", schema, None) is not None
    assert cache_db.get("q3", schema, None) is not None
    assert cache_db.get("q4", schema, None) is not None


def test_cache_concurrency(cache_db):
    schema = {"schema_fingerprint": "sch_123", "tenant_id": "tenant_A"}
    prediction = {
        "status": "completed",
        "query_ir": {"intent": "show"},
        "validation": {"is_valid": True},
    }
    
    errors = []
    
    def worker(num):
        try:
            for i in range(10):
                cache_db.put(f"worker_{num}_q_{i}", schema, None, prediction)
                cache_db.get(f"worker_{num}_q_{i}", schema, None)
        except Exception as exc:
            errors.append(exc)
            
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    assert len(errors) == 0, f"Concurrency errors: {errors}"


# Source: tests/test_133_telemetry_privacy.py
"""Unit and privacy compliance tests for the TelemetryLogger sanitization."""


import json
import os
from pathlib import Path
import pytest

from inference.telemetry_logger import TelemetryLogger


@pytest.fixture
def telemetry_logger(tmp_path):
    log_file = tmp_path / "telemetry.jsonl"
    return TelemetryLogger(log_path=log_file)


def test_telemetry_default_excludes_raw_content(telemetry_logger):
    question = "Please email target@example.com or call 555-019-9000 about credit card 4111111111111111"
    result = {
        "status": "completed",
        "source_model": "neural",
        "intent": "show_records",
        "sql": "SELECT * FROM users",
        "confidence": 0.9,
    }
    
    # Defaults do NOT write raw question or SQL text
    telemetry_logger.log_prediction(question, result)
    
    log_content = telemetry_logger.log_path.read_text(encoding="utf-8")
    log_entry = json.loads(log_content.strip())
    
    assert "question_hash" in log_entry
    assert "raw_question" not in log_entry
    assert "raw_sql" not in log_entry


def test_telemetry_pii_sanitization_rules(telemetry_logger):
    # Enable raw logs to inspect PII redactor outputs
    os.environ["NL2SQL_TELEMETRY_INCLUDE_RAW_QUESTION"] = "1"
    os.environ["NL2SQL_TELEMETRY_INCLUDE_RAW_SQL"] = "1"
    
    try:
        # Valid Luhn card (VISA test card)
        visa_card = "4111 1111 1111 1111"
        # Non-Luhn card (arbitrary number sequence)
        non_luhn = "1234-5678-9012-3456"
        
        question = f"My email is support@acme.com, phone 1-206-555-0100, card {visa_card}, order {non_luhn}."
        result = {
            "status": "completed",
            "source_model": "neural",
            "sql": "SELECT * FROM users WHERE apikey = 'sk-live-123456789abc'",
        }
        
        telemetry_logger.log_prediction(question, result)
        
        log_content = telemetry_logger.log_path.read_text(encoding="utf-8")
        log_entry = json.loads(log_content.strip())
        
        raw_q = log_entry["raw_question"]
        raw_sql = log_entry["raw_sql"]
        
        assert "support@acme.com" not in raw_q
        assert "[EMAIL]" in raw_q
        
        assert "1-206-555-0100" not in raw_q
        assert "[PHONE]" in raw_q
        
        # VISA card should be redacted because it passes Luhn check
        assert visa_card not in raw_q
        assert "[CARD]" in raw_q
        
        # Order number/Non-Luhn card should NOT be redacted as card
        assert non_luhn in raw_q
        
        # API keys/Secrets in SQL must be redacted
        assert "sk-live-123456789abc" not in raw_sql
        assert "[SECRET]" in raw_sql
        
    finally:
        del os.environ["NL2SQL_TELEMETRY_INCLUDE_RAW_QUESTION"]
        del os.environ["NL2SQL_TELEMETRY_INCLUDE_RAW_SQL"]


def test_recursive_payload_sanitizer(telemetry_logger):
    payload = {
        "user_profile": {
            "email": "user@gmail.com",
            "phones": ["+1-555-019-2831", "just a string"]
        },
        "query_history": [
            "show cards matching 4111-1111-1111-1111",
        ]
    }
    
    sanitized = telemetry_logger.sanitize_payload(payload)
    
    assert sanitized["user_profile"]["email"] == "[EMAIL]"
    assert sanitized["user_profile"]["phones"][0] == "[PHONE]"
    assert sanitized["user_profile"]["phones"][1] == "just a string"
    assert "[CARD]" in sanitized["query_history"][0]


def test_telemetry_logger_permissions_and_rotation(telemetry_logger):
    # Setup small rotation bytes
    telemetry_logger.max_bytes = 20
    
    # Write entries to trigger rotation
    telemetry_logger._write_entry({"data": "short"})
    telemetry_logger._write_entry({"data": "trigger_rotation"})
    telemetry_logger._write_entry({"data": "three"})
    
    # Check that rotation file exists
    assert telemetry_logger.log_path.exists()
    assert telemetry_logger.log_path.with_name(f"{telemetry_logger.log_path.name}.1").exists()
    
    # Permissions check (only on Unix/MacOS, skip on Windows)
    if os.name == "posix":
        mode = telemetry_logger.log_path.stat().st_mode
        assert oct(mode & 0o777) == "0o600"
