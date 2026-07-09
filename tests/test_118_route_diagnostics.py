"""Tests for Stage 2 offline route diagnostics and forced routing.

Validates safety constraints, leakage controls, oracle metrics, and router regret.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from inference import (
    RuntimeMode,
    PredictionRoute,
    DiagnosticContext,
    DiagnosticRoutingNotAllowedError,
)
from inference.prediction_models import PredictionResult
from inference.prediction_orchestrator import PredictionOrchestrator
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from evaluation.route_diagnostics import (
    compute_slot_matches,
    segment_metrics,
    run_diagnostics,
)


def test_forced_route_rejected_in_production() -> None:
    orchestrator = PredictionOrchestrator(use_neural_ir_fallback=False)
    ctx = DiagnosticContext(
        forced_route=PredictionRoute.RETRIEVAL,
        runtime_mode=RuntimeMode.PRODUCTION,
    )
    
    with pytest.raises(DiagnosticRoutingNotAllowedError) as exc_info:
        orchestrator.predict(
            question="What is the revenue?",
            schema={},
            retriever=None,
            diagnostic_context=ctx,
        )
    assert "Forced routing is forbidden in production runtime" in str(exc_info.value)


def test_forced_route_rejected_when_runtime_mode_unknown() -> None:
    orchestrator = PredictionOrchestrator(use_neural_ir_fallback=False)
    ctx = DiagnosticContext(
        forced_route=PredictionRoute.RETRIEVAL,
        runtime_mode="invalid_mode",  # type: ignore
    )
    
    with pytest.raises(DiagnosticRoutingNotAllowedError) as exc_info:
        orchestrator.predict(
            question="What is the revenue?",
            schema={},
            retriever=None,
            diagnostic_context=ctx,
        )
    assert "Unknown runtime mode" in str(exc_info.value) or "invalid_mode" in str(exc_info.value)


def test_forced_route_allowed_in_test_mode() -> None:
    orchestrator = PredictionOrchestrator(use_neural_ir_fallback=False)
    ctx = DiagnosticContext(
        forced_route=PredictionRoute.RETRIEVAL,
        runtime_mode=RuntimeMode.TEST,
    )
    
    from inference.prediction_models import SchemaMapping
    mock_mapping = SchemaMapping(
        base_table="orders",
        metric_table="orders",
        metric_column="amount",
        filter_table="orders",
        filter_column="status",
        warnings=[],
    )
    
    # Under forced retrieval, it should run normally (no error)
    # We mock retrieval candidate generator to prevent actual retrieval failure
    with patch.object(orchestrator.generator, "generate_candidates", return_value=[]), \
         patch.object(orchestrator.reranker, "rerank_candidates", return_value=[]), \
         patch.object(orchestrator.selector, "select_template", return_value={"template_id": "metric_summary", "intent": "metric_summary"}), \
         patch.object(orchestrator.slot_resolver, "resolve_slots", return_value={"slots": {}}), \
         patch.object(orchestrator.mapper, "map_slots_to_schema", return_value=mock_mapping):
        
        res = orchestrator.predict(
            question="What is the revenue?",
            schema={"tables": {"orders": {"columns": {"amount": "real", "status": "text"}}}},
            retriever=None,
            diagnostic_context=ctx,
        )
        assert res.source_model == "retrieval"
        assert res.router_decision["selected"] == "retrieval"


def test_forced_retrieval_bypasses_router_only() -> None:
    orchestrator = PredictionOrchestrator(use_neural_ir_fallback=True)
    # Even with neural fallback enabled, forcing retrieval should return retrieval
    ctx = DiagnosticContext(
        forced_route=PredictionRoute.RETRIEVAL,
        runtime_mode=RuntimeMode.TEST,
    )
    
    from inference.prediction_models import SchemaMapping
    mock_mapping = SchemaMapping(
        base_table="orders",
        metric_table="orders",
        metric_column="amount",
        filter_table="orders",
        filter_column="status",
        warnings=[],
    )
    
    with patch.object(orchestrator.generator, "generate_candidates", return_value=[]), \
         patch.object(orchestrator.reranker, "rerank_candidates", return_value=[]), \
         patch.object(orchestrator.selector, "select_template", return_value={"template_id": "metric_summary", "intent": "metric_summary"}), \
         patch.object(orchestrator.slot_resolver, "resolve_slots", return_value={"slots": {}}), \
         patch.object(orchestrator.mapper, "map_slots_to_schema", return_value=mock_mapping):
        
        res = orchestrator.predict(
            question="What is the revenue?",
            schema={"tables": {"orders": {"columns": {"amount": "real", "status": "text"}}}},
            retriever=None,
            diagnostic_context=ctx,
        )
        assert res.source_model == "retrieval"
        # Verify choose_route was not called by checking router decision reason
        assert res.router_decision["reason"] == "forced_retrieval_route"


def test_forced_neural_bypasses_router_only() -> None:
    orchestrator = PredictionOrchestrator(use_neural_ir_fallback=True)
    ctx = DiagnosticContext(
        forced_route=PredictionRoute.NEURAL,
        runtime_mode=RuntimeMode.TEST,
    )
    
    from inference.prediction_models import SchemaMapping
    mock_mapping = SchemaMapping(
        base_table="orders",
        metric_table="orders",
        metric_column="amount",
        filter_table="orders",
        filter_column="status",
        warnings=[],
    )
    
    with patch.object(orchestrator.generator, "generate_candidates", return_value=[]), \
         patch.object(orchestrator.reranker, "rerank_candidates", return_value=[]), \
         patch.object(orchestrator.selector, "select_template", return_value={"template_id": "metric_summary", "intent": "metric_summary"}), \
         patch.object(orchestrator.slot_resolver, "resolve_slots", return_value={"slots": {}}), \
         patch.object(orchestrator.mapper, "map_slots_to_schema", return_value=mock_mapping), \
         patch.object(orchestrator, "_available_neural_ir_model_dir", return_value=Path("dummy_path")), \
         patch("neural_ir.predictor.NeuralIRPredictor.__init__", return_value=None), \
         patch("neural_ir.predictor.NeuralIRPredictor.predict", return_value={"query_ir": {"intent": "metric_summary"}, "confidence": 0.95}):
        
        res = orchestrator.predict(
            question="What is the revenue?",
            schema={"tables": {"orders": {"columns": {"amount": "real", "status": "text"}}}},
            retriever=None,
            diagnostic_context=ctx,
        )
        assert res.source_model == "neural"
        assert res.router_decision["reason"] == "forced_neural_route"


def test_direct_planner_diagnostic_route() -> None:
    orchestrator = PredictionOrchestrator(use_neural_ir_fallback=True)
    ctx = DiagnosticContext(
        forced_route=PredictionRoute.DIRECT_PLANNER,
        runtime_mode=RuntimeMode.TEST,
    )
    
    # We mock TableIntentResolver so that it resolves (or handles) it
    from ir.query_ir_models import QueryIR
    mock_resolve = MagicMock()
    mock_resolve.handled = True
    mock_resolve.query_ir = QueryIR(
        query_ir_id="1",
        question="What is the revenue?",
        normalized_question="what is the revenue?",
        intent="metric_summary",
        base_table="orders",
        metrics=[],
        dimensions=[],
        filters=[]
    )
    mock_resolve.confidence = 0.9
    mock_resolve.warnings = []
    mock_resolve.reason = "mock direct route"
    mock_resolve.debug = {}
    
    with patch("generic_planner.TableIntentResolver.resolve", return_value=mock_resolve):
        res = orchestrator.predict(
            question="What is the revenue?",
            schema={},
            retriever=None,
            diagnostic_context=ctx,
        )
        assert res.source_model == "generic_direct_planner"
        assert res.router_decision["selected"] == "direct_planner"


def test_unavailable_route_is_reported_not_scored_as_failure() -> None:
    orchestrator = PredictionOrchestrator(use_neural_ir_fallback=True)
    ctx = DiagnosticContext(
        forced_route=PredictionRoute.NEURAL,
        runtime_mode=RuntimeMode.TEST,
    )
    
    from inference.prediction_models import SchemaMapping
    mock_mapping = SchemaMapping(
        base_table="orders",
        metric_table="orders",
        metric_column="amount",
        filter_table="orders",
        filter_column="status",
        warnings=[],
    )
    
    # Mocking neural dir as None to trigger unavailable path
    with patch.object(orchestrator.generator, "generate_candidates", return_value=[]), \
         patch.object(orchestrator.reranker, "rerank_candidates", return_value=[]), \
         patch.object(orchestrator.selector, "select_template", return_value={"template_id": "metric_summary", "intent": "metric_summary"}), \
         patch.object(orchestrator.slot_resolver, "resolve_slots", return_value={"slots": {}}), \
         patch.object(orchestrator.mapper, "map_slots_to_schema", return_value=mock_mapping), \
         patch.object(orchestrator, "_available_neural_ir_model_dir", return_value=None):
        
        res = orchestrator.predict(
            question="What is the revenue?",
            schema={"tables": {"orders": {"columns": {"amount": "real", "status": "text"}}}},
            retriever=None,
            diagnostic_context=ctx,
        )
        assert res.debug.get("forced_route_unavailable") is True
        assert res.debug.get("unavailable_reason") == "neural_model_missing"


def test_gold_query_ir_never_passed_to_orchestrator() -> None:
    """Leakage check: Verify that only question & schema enter the prediction path."""
    import inspect
    sig = inspect.signature(PredictionOrchestrator.predict)
    # Assert that no argument name starts with gold or labels
    for name in sig.parameters:
        assert not name.startswith("gold")
        assert not name.startswith("label")


def test_router_regret_metrics() -> None:
    # Test case segmenting & regret calculation logic
    cases = [
        # Selected route passed (no regret)
        {
            "example_id": "1",
            "selected_route_passed": True,
            "oracle_route_available": True,
            "router_regret": False,
            "route_results": {
                "retrieval": {"semantic_pass": True},
                "neural": {"semantic_pass": False},
            }
        },
        # Selected route failed, but oracle was available (regret!)
        {
            "example_id": "2",
            "selected_route_passed": False,
            "oracle_route_available": True,
            "router_regret": True,
            "route_results": {
                "retrieval": {"semantic_pass": False},
                "neural": {"semantic_pass": True},
            }
        },
    ]
    
    metrics = segment_metrics(cases, "dataset")
    assert "unknown" in metrics
    summary = metrics["unknown"]
    assert summary["total_cases"] == 2
    assert summary["router_regret_count"] == 1
    assert summary["router_regret_rate"] == 0.5
    assert summary["selected_semantic_pass_rate"] == 0.5
    assert summary["oracle_semantic_pass_rate"] == 1.0


def test_diagnostics_disable_cache_writes(tmp_path: Path) -> None:
    # Mock retrieval model and verify cache put is skipped when cache_write_enabled=False
    model = RetrievalNL2SQLModel(
        retriever=MagicMock(),
        orchestrator=MagicMock(),
    )
    # Set cache to mock
    model._cache = MagicMock()
    
    ctx = DiagnosticContext(
        forced_route=PredictionRoute.RETRIEVAL,
        cache_write_enabled=False,
    )
    
    model.predict("test question", MagicMock(), diagnostic_context=ctx)
    model._cache.put.assert_not_called()
