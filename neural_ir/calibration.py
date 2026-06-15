from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CALIBRATION = {
    "retrieval_ir_high_confidence_threshold": 0.8,
    "neural_ir_min_confidence_threshold": 0.6,
    "neural_ir_confidence_margin": 0.05,
    "fallback_threshold": 0.8,
    "validation_bonus": 0.05,
    "sql_validity_bonus": 0.10,
    # Backward-compatible aliases (read-only)
    "option_c_high_confidence_threshold": 0.8,
    "option_a_min_confidence_threshold": 0.6,
    "option_a_confidence_margin": 0.05,
}


class AdaptiveRouterCalibrator:
    """Calibrator for the Adaptive QueryIR Router.

    Formerly named ``HybridRouterCalibrator``.
    """
    def calibrate(self, retrieval_ir_results: list[dict], neural_ir_results: list[dict]) -> dict[str, Any]:
        cases = []
        correct = 0
        total = max(len(retrieval_ir_results), len(neural_ir_results))
        for idx in range(total):
            retrieval_ir = retrieval_ir_results[idx] if idx < len(retrieval_ir_results) else {}
            neural_ir = neural_ir_results[idx] if idx < len(neural_ir_results) else {}
            decision = choose_route(retrieval_ir, neural_ir, DEFAULT_CALIBRATION)
            expected = retrieval_ir.get("expected_source") or neural_ir.get("expected_source")
            if expected is None:
                expected = _expected_from_validation(retrieval_ir, neural_ir)
            if decision["selected"] == expected:
                correct += 1
            cases.append({"id": retrieval_ir.get("id") or neural_ir.get("id") or idx, **decision})
        return {
            **DEFAULT_CALIBRATION,
            "router_accuracy": correct / max(total, 1),
            "cases": cases,
        }


# Backward-compatible alias
HybridRouterCalibrator = AdaptiveRouterCalibrator
"""Deprecated alias. Use ``AdaptiveRouterCalibrator``."""


def choose_route(retrieval_ir: dict[str, Any], neural_ir: dict[str, Any], calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    """Choose between retrieval IR and neural IR model results."""
    cfg = {**DEFAULT_CALIBRATION, **(calibration or {})}
    c_conf = float(retrieval_ir.get("confidence") or 0.0)
    a_conf = float(neural_ir.get("confidence") or 0.0)
    c_valid = _valid(retrieval_ir)
    a_valid = _valid(neural_ir)
    repairs_applied = neural_ir.get("repairs_applied") or (neural_ir.get("debug") or {}).get("repairs", {}).get("repairs_applied", [])
    margin = float(cfg.get("neural_ir_confidence_margin", cfg.get("option_a_confidence_margin", 0.05)))
    min_threshold = float(cfg.get("neural_ir_min_confidence_threshold", cfg.get("option_a_min_confidence_threshold", 0.6)))
    high_threshold = float(cfg.get("retrieval_ir_high_confidence_threshold", cfg.get("option_c_high_confidence_threshold", 0.8)))
    better_than_retrieval = a_conf >= max(c_conf + margin, min_threshold)
    if c_valid and c_conf >= high_threshold:
        selected, reason = "retrieval_ir", "retrieval_ir_high_confidence"
    elif not c_valid and a_valid:
        selected, reason = "neural_ir", "retrieval_ir_invalid_sql"
    elif a_valid and repairs_applied and better_than_retrieval:
        selected, reason = "neural_ir", "neural_ir_repaired_valid"
    elif a_valid and better_than_retrieval:
        selected, reason = "neural_ir", "neural_ir_higher_confidence"
    elif not a_valid:
        selected, reason = "retrieval_ir", "neural_ir_invalid"
    elif c_conf < high_threshold:
        selected, reason = "retrieval_ir", "retrieval_ir_low_confidence"
    else:
        selected, reason = "retrieval_ir", "fallback_to_retrieval_ir"
    return {
        "retrieval_ir_confidence": c_conf,
        "neural_ir_confidence": a_conf,
        "retrieval_ir_valid": c_valid,
        "neural_ir_valid": a_valid,
        "selected": selected,
        "reason": reason,
    }


def load_hybrid_calibration(path: str | Path) -> dict[str, Any]:
    calibration_path = Path(path)
    if not calibration_path.exists():
        return dict(DEFAULT_CALIBRATION)
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    return {**DEFAULT_CALIBRATION, **payload}


def _valid(result: dict[str, Any]) -> bool:
    validation = result.get("validation") or result.get("sql_validation") or {}
    return bool(validation.get("is_valid", validation.get("ok", False)))


def _expected_from_validation(retrieval_ir: dict[str, Any], neural_ir: dict[str, Any]) -> str:
    c_valid = _valid(retrieval_ir)
    a_valid = _valid(neural_ir)
    if c_valid and not a_valid:
        return "retrieval_ir"
    if a_valid and not c_valid:
        return "neural_ir"
    return "neural_ir" if float(neural_ir.get("confidence") or 0.0) > float(retrieval_ir.get("confidence") or 0.0) else "retrieval_ir"
