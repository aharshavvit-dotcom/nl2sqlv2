"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

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


def test_valid_prediction_has_no_confidence_caps() -> None:
    confidence = PredictionConfidenceCalculator().calculate(_base_parts())

    assert confidence["confidence"] >= 0.8
    assert confidence["confidence_breakdown"]["caps_applied"] == []


def test_invalid_ir_and_sql_caps_are_reported() -> None:
    parts = _base_parts()
    parts["ir_validation"] = {"is_valid": False}
    parts["validation"] = {"is_valid": False, "ok": False}

    confidence = PredictionConfidenceCalculator().calculate(parts)

    assert confidence["confidence"] <= 0.59
    assert "ir_validation_failed" in confidence["confidence_breakdown"]["caps_applied"]
    assert "sql_validation_failed" in confidence["confidence_breakdown"]["caps_applied"]


def test_missing_required_dimension_cap_is_reported() -> None:
    parts = _base_parts()
    parts["schema_mapping"] = {
        "metric_table": "orders",
        "dimension_table": None,
        "match_scores": {"metric": 0.95},
    }

    confidence = PredictionConfidenceCalculator().calculate(parts)

    assert confidence["confidence"] <= 0.45
    assert "required_dimension_missing" in confidence["confidence_breakdown"]["caps_applied"]


def test_join_semantic_date_and_filter_caps_are_reported() -> None:
    parts = _base_parts()
    parts["join_plan"] = {"confidence": 0.3, "warnings": ["no join path from orders to products"]}
    parts["warnings"] = [
        "semantic grain risk: product-level revenue needs item-level quantity/price columns",
        "date filter requested but no date column was mapped",
        "filter requested but no filter column was mapped",
    ]

    confidence = PredictionConfidenceCalculator().calculate(parts)
    caps = confidence["confidence_breakdown"]["caps_applied"]

    assert confidence["confidence"] <= 0.49
    assert "join_path_missing" in caps
    assert "semantic_grain_risk" in caps
    assert "date_filter_missing" in caps
    assert "filter_mapping_missing" in caps

