"""Test 06: Adaptive Router — choose_route, calibration, confidence caps."""

from __future__ import annotations

from neural_ir.calibration import (
    AdaptiveRouterCalibrator,
    HybridRouterCalibrator,
    choose_route,
    DEFAULT_CALIBRATION,
)
from inference.prediction_models import PredictionResult
from inference.prediction_orchestrator import PredictionOrchestrator


class TestChooseRoute:
    def test_high_confidence_retrieval_ir_wins(self) -> None:
        decision = choose_route(
            {"confidence": 0.9, "validation": {"is_valid": True}},
            {"confidence": 0.5, "sql_validation": {"is_valid": True}},
        )
        assert decision["selected"] == "retrieval_ir"
        assert decision["reason"] == "retrieval_ir_high_confidence"

    def test_invalid_retrieval_ir_uses_neural_ir(self) -> None:
        decision = choose_route(
            {"confidence": 0.3, "validation": {"is_valid": False}},
            {"confidence": 0.6, "sql_validation": {"is_valid": True}},
        )
        assert decision["selected"] == "neural_ir"
        assert decision["reason"] == "retrieval_ir_invalid_sql"

    def test_higher_neural_ir_selected(self) -> None:
        decision = choose_route(
            {"confidence": 0.3, "validation": {"is_valid": True}},
            {"confidence": 0.7, "sql_validation": {"is_valid": True}},
        )
        assert decision["selected"] == "neural_ir"

    def test_invalid_neural_ir_keeps_retrieval(self) -> None:
        decision = choose_route(
            {"confidence": 0.3, "validation": {"is_valid": True}},
            {"confidence": 0.7, "sql_validation": {"is_valid": False}},
        )
        assert decision["selected"] == "retrieval_ir"

    def test_output_uses_new_field_names(self) -> None:
        decision = choose_route(
            {"confidence": 0.5, "validation": {"is_valid": True}},
            {"confidence": 0.5, "sql_validation": {"is_valid": True}},
        )
        assert "retrieval_ir_confidence" in decision
        assert "neural_ir_confidence" in decision
        assert "retrieval_ir_valid" in decision
        assert "neural_ir_valid" in decision
        # Old names should NOT be in the output
        assert "option_c_confidence" not in decision
        assert "option_a_confidence" not in decision


class TestAdaptiveRouterCalibrator:
    def test_backward_alias(self) -> None:
        assert HybridRouterCalibrator is AdaptiveRouterCalibrator

    def test_calibrate_empty_results(self) -> None:
        calibrator = AdaptiveRouterCalibrator()
        result = calibrator.calibrate([], [])
        assert "router_accuracy" in result

    def test_calibrate_with_results(self) -> None:
        retrieval_results = [
            {"confidence": 0.9, "validation": {"is_valid": True}},
            {"confidence": 0.3, "validation": {"is_valid": True}},
        ]
        neural_results = [
            {"confidence": 0.5, "sql_validation": {"is_valid": True}},
            {"confidence": 0.7, "sql_validation": {"is_valid": True}},
        ]
        calibrator = AdaptiveRouterCalibrator()
        result = calibrator.calibrate(retrieval_results, neural_results)
        assert 0.0 <= result["router_accuracy"] <= 1.0
        assert len(result["cases"]) == 2


class TestRouterInOrchestrator:
    def test_missing_neural_ir_returns_retrieval_ir(self, tmp_path) -> None:
        result = PredictionOrchestrator(neural_ir_model_dir=tmp_path)._maybe_neural_ir_fallback(
            retrieval_ir_result=_make_result(confidence=0.2, valid=True),
            question="How many orders?",
            schema={},
            enabled=True,
        )
        assert result.source_model == "retrieval_ir"
        assert result.router_decision["reason"] == "neural_ir_missing"

    def test_high_confidence_retrieval_ir_skips_neural(self, tmp_path) -> None:
        result = PredictionOrchestrator(neural_ir_model_dir=tmp_path)._maybe_neural_ir_fallback(
            retrieval_ir_result=_make_result(confidence=0.9, valid=True),
            question="Top customers",
            schema={},
            enabled=True,
        )
        assert result.source_model == "retrieval_ir"
        assert result.router_decision["reason"] == "retrieval_ir_high_confidence"

    def test_disabled_neural_ir_returns_retrieval_ir(self, tmp_path) -> None:
        result = PredictionOrchestrator(neural_ir_model_dir=tmp_path)._maybe_neural_ir_fallback(
            retrieval_ir_result=_make_result(confidence=0.2, valid=True),
            question="How many orders?",
            schema={},
            enabled=False,
        )
        assert result.source_model == "retrieval_ir"
        assert result.router_decision["reason"] == "neural_ir_disabled"


class TestConfidenceCaps:
    def test_ir_invalid_caps_confidence(self) -> None:
        from inference.prediction_confidence import PredictionConfidenceCalculator
        calc = PredictionConfidenceCalculator()
        result = calc.calculate({
            "candidates": [], "selected_template": {}, "slots": {},
            "schema_mapping": {}, "join_plan": {},
            "ir_validation": {"is_valid": False},
            "validation": {"is_valid": True},
            "warnings": [],
        })
        assert result["confidence"] <= 0.59

    def test_sql_invalid_caps_confidence(self) -> None:
        from inference.prediction_confidence import PredictionConfidenceCalculator
        calc = PredictionConfidenceCalculator()
        result = calc.calculate({
            "candidates": [], "selected_template": {}, "slots": {},
            "schema_mapping": {}, "join_plan": {},
            "ir_validation": {"is_valid": True},
            "validation": {"is_valid": False},
            "warnings": [],
        })
        assert result["confidence"] <= 0.59


def _make_result(confidence: float, valid: bool) -> PredictionResult:
    return PredictionResult(
        question="q", normalized_question="q", source_model="retrieval_ir",
        intent="show_records", template_id="show_records",
        sql="SELECT order_id FROM orders LIMIT 100" if valid else None,
        validation={"is_valid": valid, "ok": valid, "issues": [] if valid else ["bad"]},
        confidence=confidence, confidence_tier="high" if confidence >= 0.8 else "low",
        debug={},
    )
