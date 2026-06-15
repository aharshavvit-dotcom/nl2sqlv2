from __future__ import annotations

from inference.prediction_confidence import PredictionConfidenceCalculator
from inference.prediction_models import RetrievedCandidate


def _candidate() -> RetrievedCandidate:
    return RetrievedCandidate(
        rank=1,
        example_id="ex1",
        question="Top 5 customers by sales",
        template_id="top_n_metric_by_dimension",
        slots={"metric": "sales", "dimension": "customer"},
        similarity_score=0.9,
        rerank_score=0.9,
    )


def _base_parts() -> dict[str, object]:
    return {
        "candidates": [_candidate()],
        "selected_template": {"template_id": "top_n_metric_by_dimension", "confidence": 0.9},
        "slots": {
            "metric": {"value": "sales", "confidence": 0.9},
            "dimension": {"value": "customer", "confidence": 0.9},
            "entity": {"value": "orders", "confidence": 0.9},
        },
        "schema_mapping": {
            "metric_table": "orders",
            "dimension_table": "customers",
            "match_scores": {"metric": 0.95, "dimension": 0.95, "entity": 0.9, "date": 0.9},
        },
        "join_plan": {"confidence": 1.0, "warnings": []},
        "ir_validation": {"is_valid": True},
        "validation": {"is_valid": True, "ok": True},
        "warnings": [],
    }


def test_confidence_breakdown_includes_ir_and_sql_validation() -> None:
    confidence = PredictionConfidenceCalculator().calculate(_base_parts())

    assert confidence["confidence"] >= 0.8
    assert confidence["confidence_tier"] == "high"
    assert confidence["confidence_breakdown"]["ir_validation"] == 1.0
    assert confidence["confidence_breakdown"]["sql_validation"] == 1.0
    assert confidence["confidence_breakdown"]["join_planning"] == 1.0


def test_sql_validation_failure_caps_confidence() -> None:
    parts = _base_parts()
    parts["validation"] = {"is_valid": False, "ok": False}

    confidence = PredictionConfidenceCalculator().calculate(parts)

    assert confidence["confidence"] <= 0.59
    assert confidence["confidence_tier"] == "low"


def test_missing_join_caps_confidence() -> None:
    parts = _base_parts()
    parts["join_plan"] = {"confidence": 0.35, "warnings": ["no join path"]}

    confidence = PredictionConfidenceCalculator().calculate(parts)

    assert confidence["confidence"] <= 0.49
    assert confidence["confidence_tier"] == "low"


def test_product_semantic_warning_caps_confidence_medium() -> None:
    parts = _base_parts()
    parts["warnings"] = ["semantic grain risk: product-level revenue needs item-level quantity/price columns"]

    confidence = PredictionConfidenceCalculator().calculate(parts)

    assert confidence["confidence"] <= 0.69
    assert confidence["confidence_tier"] == "medium"
