"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import json

from inference.prediction_models import PredictionResult
from inference.prediction_orchestrator import PredictionOrchestrator
from neural_ir.calibration import HybridRouterCalibrator, choose_route


def test_hybrid_router_calibrator_selects_expected_routes() -> None:
    assert choose_route({"confidence": 0.9, "validation": {"is_valid": True}}, {"confidence": 0.2, "sql_validation": {"is_valid": True}})["selected"] == "option_c"
    assert choose_route({"confidence": 0.1, "validation": {"is_valid": False}}, {"confidence": 0.2, "sql_validation": {"is_valid": True}})["selected"] == "option_a"
    report = HybridRouterCalibrator().calibrate(
        [{"id": "c1", "confidence": 0.9, "validation": {"is_valid": True}}],
        [{"id": "c1", "confidence": 0.2, "sql_validation": {"is_valid": True}}],
    )
    assert "router_accuracy" in report


def test_hybrid_router_calibration_file_loads(tmp_path) -> None:
    (tmp_path / "hybrid_calibration.json").write_text(json.dumps({"option_c_high_confidence_threshold": 0.7}), encoding="utf-8")
    orchestrator = PredictionOrchestrator(option_a_model_dir=tmp_path)
    assert orchestrator.option_a_threshold == 0.7


def test_hybrid_router_missing_option_a_selects_option_c(tmp_path) -> None:
    result = PredictionOrchestrator(option_a_model_dir=tmp_path)._maybe_option_a_fallback(
        option_c_result=_option_c_result(confidence=0.2, valid=True),
        question="How many orders?",
        schema={},
        enabled=True,
    )
    assert result.router_decision["selected"] == "option_c"
    assert result.router_decision["reason"] == "option_a_missing"


def _option_c_result(confidence: float, valid: bool) -> PredictionResult:
    return PredictionResult(
        question="q",
        normalized_question="q",
        source_model="option_c",
        intent="show_records",
        template_id="show_records",
        sql="SELECT orders.order_id FROM orders LIMIT 100" if valid else None,
        validation={"is_valid": valid, "ok": valid, "issues": [] if valid else ["bad"]},
        confidence=confidence,
        confidence_tier="high" if confidence >= 0.8 else "low",
        debug={},
    )

