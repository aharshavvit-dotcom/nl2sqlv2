from __future__ import annotations

from inference.prediction_models import PredictionResult
from inference.prediction_orchestrator import PredictionOrchestrator


def test_hybrid_router_keeps_high_confidence_option_c(tmp_path) -> None:
    result = PredictionOrchestrator(option_a_model_dir=tmp_path)._maybe_option_a_fallback(
        option_c_result=_option_c_result(confidence=0.9, valid=True),
        question="Top customers",
        schema={},
        enabled=True,
    )

    assert result.source_model == "option_c"
    assert result.debug["router_decision"] == "option_c_high_confidence"


def test_hybrid_router_uses_valid_option_a_when_option_c_low(tmp_path, monkeypatch) -> None:
    (tmp_path / "model.pt").write_text("present", encoding="utf-8")

    class DummyPredictor:
        def __init__(self, model_dir: str):
            self.model_dir = model_dir

        def predict(self, question: str, schema: dict) -> dict:
            return {
                "query_ir": {"intent": "count_records", "template_id": "count_records"},
                "ir_validation": {"is_valid": True, "warnings": [], "errors": []},
                "sql": "SELECT COUNT(*) AS record_count FROM orders LIMIT 100",
                "sql_validation": {"is_valid": True, "ok": True, "issues": []},
                "confidence": 0.7,
                "debug": {},
            }

    monkeypatch.setattr("neural_ir.predictor.OptionAIRPredictor", DummyPredictor)
    result = PredictionOrchestrator(option_a_model_dir=tmp_path)._maybe_option_a_fallback(
        option_c_result=_option_c_result(confidence=0.2, valid=True),
        question="How many orders?",
        schema={},
        enabled=True,
    )

    assert result.source_model == "option_a"
    assert result.debug["router_decision"] == "option_a_fallback_used"


def test_hybrid_router_missing_option_a_keeps_option_c(tmp_path) -> None:
    result = PredictionOrchestrator(option_a_model_dir=tmp_path)._maybe_option_a_fallback(
        option_c_result=_option_c_result(confidence=0.2, valid=True),
        question="How many orders?",
        schema={},
        enabled=True,
    )

    assert result.source_model == "option_c"
    assert result.debug["router_decision"] == "option_a_missing"


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
