from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


class OptionAConfidenceCalibrator:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {
            "version": 1,
            "bias": 0.0,
            "slot_confidence_floor": 0.0,
            "fitted_rows": 0,
        }

    def fit(self, prediction_rows: list[dict]) -> dict:
        if not prediction_rows:
            self.payload.update({"fitted_rows": 0, "bias": 0.0})
            return dict(self.payload)
        residuals = []
        slot_confidences = []
        for row in prediction_rows:
            raw = float(row.get("raw_confidence", row.get("confidence", 0.0)) or 0.0)
            target = 1.0 if row.get("correct", row.get("passed", False)) else 0.0
            residuals.append(target - raw)
            debug = row.get("prediction_debug") or row.get("debug") or {}
            slot_confidences.extend(_slot_confidences(debug))
        self.payload.update(
            {
                "fitted_rows": len(prediction_rows),
                "bias": max(-0.25, min(0.25, mean(residuals))),
                "slot_confidence_floor": mean(slot_confidences) if slot_confidences else 0.0,
            }
        )
        return dict(self.payload)

    def calibrate(self, raw_confidence: float, validation_summary: dict, prediction_debug: dict) -> float:
        value = max(0.0, min(1.0, float(raw_confidence or 0.0) + float(self.payload.get("bias", 0.0))))
        ir_valid = _valid(validation_summary.get("ir_validation") or validation_summary.get("ir") or validation_summary)
        sql_valid = _valid(validation_summary.get("sql_validation") or validation_summary.get("sql") or validation_summary)
        repairs = prediction_debug.get("repairs") or validation_summary.get("repairs") or {}
        repairs_applied = repairs.get("repairs_applied") or prediction_debug.get("repairs_applied") or []
        missing_required = _missing_required_slot(validation_summary, prediction_debug)
        pointer_confidence = _mean(_slot_confidences(prediction_debug))
        schema_link_score = _schema_link_score(prediction_debug)

        if ir_valid and sql_valid and pointer_confidence >= 0.75 and schema_link_score >= 0.35:
            value = max(value, min(0.95, (value + pointer_confidence + schema_link_score) / 3.0 + 0.15))
        if repairs_applied:
            value = min(value, 0.79)
        if any("product_revenue" in str(item) for item in repairs_applied):
            value = min(value, 0.79)
        if not ir_valid:
            value = min(value, 0.49)
        if not sql_valid:
            value = min(value, 0.49)
        if missing_required:
            value = min(value, 0.39)
        return round(max(0.0, min(1.0, value)), 4)

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str):
        target = Path(path)
        if not target.exists():
            return cls()
        return cls(json.loads(target.read_text(encoding="utf-8")))


def _valid(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if "is_valid" in payload:
        return bool(payload.get("is_valid"))
    if "ok" in payload:
        return bool(payload.get("ok"))
    return not payload.get("errors")


def _slot_confidences(debug: dict[str, Any]) -> list[float]:
    values = []
    for container_key in ["confidence_breakdown", "candidate_scores", "decoded_prediction"]:
        container = debug.get(container_key) or {}
        if isinstance(container, dict):
            for key, value in container.items():
                if key.endswith("_confidence") or key in {"intent", "pointer"}:
                    try:
                        values.append(float(value))
                    except Exception:
                        pass
    return values


def _schema_link_score(debug: dict[str, Any]) -> float:
    linking = debug.get("schema_linking") or {}
    top = linking.get("top_columns") or []
    scores = [float(item.get("score") or 0.0) for item in top[:5] if isinstance(item, dict)]
    return _mean(scores)


def _missing_required_slot(validation_summary: dict[str, Any], prediction_debug: dict[str, Any]) -> bool:
    issues = []
    for key in ["ir_validation", "sql_validation"]:
        payload = validation_summary.get(key) or {}
        issues.extend(payload.get("errors") or [])
        issues.extend(payload.get("issues") or [])
    text = " ".join(str(item).lower() for item in issues)
    return any(marker in text for marker in ["missing metric", "missing dimension", "missing required", "requires at least"])


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)
