"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from neural_ir.confidence_calibrator import OptionAConfidenceCalibrator


def test_confidence_calibrator_caps_invalid_ir_and_sql() -> None:
    calibrator = OptionAConfidenceCalibrator()

    assert calibrator.calibrate(0.95, {"ir_validation": {"is_valid": False}, "sql_validation": {"is_valid": True}}, {}) <= 0.49
    assert calibrator.calibrate(0.95, {"ir_validation": {"is_valid": True}, "sql_validation": {"is_valid": False}}, {}) <= 0.49


def test_confidence_calibrator_caps_successful_repair_and_allows_valid_high() -> None:
    calibrator = OptionAConfidenceCalibrator()

    repaired = calibrator.calibrate(
        0.95,
        {"ir_validation": {"is_valid": True}, "sql_validation": {"is_valid": True}, "repairs": {"repairs_applied": ["added_default_limit_100"]}},
        {"repairs": {"repairs_applied": ["added_default_limit_100"]}},
    )
    valid = calibrator.calibrate(
        0.9,
        {"ir_validation": {"is_valid": True}, "sql_validation": {"is_valid": True}},
        {"schema_linking": {"top_columns": [{"score": 0.9}]}, "confidence_breakdown": {"intent_confidence": 0.9, "pointer_confidence": 0.9}},
    )

    assert repaired <= 0.79
    assert valid >= 0.85
