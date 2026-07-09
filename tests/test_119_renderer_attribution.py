"""Tests for Stage 2 Renderer Attribution and Failure Attribution boundaries.

Validates the controlled control experiment and hierarchical stage failure checks.
"""

from __future__ import annotations

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
