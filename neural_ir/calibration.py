from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CALIBRATION = {
    "option_c_high_confidence_threshold": 0.8,
    "option_a_min_confidence_threshold": 0.6,
    "option_a_confidence_margin": 0.05,
    "fallback_threshold": 0.8,
    "validation_bonus": 0.05,
    "sql_validity_bonus": 0.10,
}


class HybridRouterCalibrator:
    def calibrate(self, option_c_results: list[dict], option_a_results: list[dict]) -> dict[str, Any]:
        cases = []
        correct = 0
        total = max(len(option_c_results), len(option_a_results))
        for idx in range(total):
            option_c = option_c_results[idx] if idx < len(option_c_results) else {}
            option_a = option_a_results[idx] if idx < len(option_a_results) else {}
            decision = choose_route(option_c, option_a, DEFAULT_CALIBRATION)
            expected = option_c.get("expected_source") or option_a.get("expected_source")
            if expected is None:
                expected = _expected_from_validation(option_c, option_a)
            if decision["selected"] == expected:
                correct += 1
            cases.append({"id": option_c.get("id") or option_a.get("id") or idx, **decision})
        return {
            **DEFAULT_CALIBRATION,
            "router_accuracy": correct / max(total, 1),
            "cases": cases,
        }


def choose_route(option_c: dict[str, Any], option_a: dict[str, Any], calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {**DEFAULT_CALIBRATION, **(calibration or {})}
    c_conf = float(option_c.get("confidence") or 0.0)
    a_conf = float(option_a.get("confidence") or 0.0)
    c_valid = _valid(option_c)
    a_valid = _valid(option_a)
    repairs_applied = option_a.get("repairs_applied") or (option_a.get("debug") or {}).get("repairs", {}).get("repairs_applied", [])
    better_than_c = a_conf >= max(c_conf + float(cfg.get("option_a_confidence_margin", 0.05)), float(cfg["option_a_min_confidence_threshold"]))
    if c_valid and c_conf >= float(cfg["option_c_high_confidence_threshold"]):
        selected, reason = "option_c", "option_c_high_confidence"
    elif not c_valid and a_valid:
        selected, reason = "option_a", "option_c_invalid_sql"
    elif a_valid and repairs_applied and better_than_c:
        selected, reason = "option_a", "option_a_repaired_valid"
    elif a_valid and better_than_c:
        selected, reason = "option_a", "option_a_higher_confidence"
    elif not a_valid:
        selected, reason = "option_c", "option_a_invalid"
    elif c_conf < float(cfg["option_c_high_confidence_threshold"]):
        selected, reason = "option_c", "option_c_low_confidence"
    else:
        selected, reason = "option_c", "fallback_to_option_c"
    return {
        "option_c_confidence": c_conf,
        "option_a_confidence": a_conf,
        "option_c_valid": c_valid,
        "option_a_valid": a_valid,
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


def _expected_from_validation(option_c: dict[str, Any], option_a: dict[str, Any]) -> str:
    c_valid = _valid(option_c)
    a_valid = _valid(option_a)
    if c_valid and not a_valid:
        return "option_c"
    if a_valid and not c_valid:
        return "option_a"
    return "option_a" if float(option_a.get("confidence") or 0.0) > float(option_c.get("confidence") or 0.0) else "option_c"
