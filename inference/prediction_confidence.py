from __future__ import annotations

from typing import Any


class PredictionConfidenceCalculator:
    def calculate(self, result_parts: dict[str, Any]) -> dict[str, Any]:
        candidates = result_parts.get("candidates") or []
        selected_template = result_parts.get("selected_template") or {}
        slots = result_parts.get("slots") or {}
        mapping = result_parts.get("schema_mapping")
        join_plan = result_parts.get("join_plan")
        validation = result_parts.get("validation") or {}
        ir_validation = result_parts.get("ir_validation") or {}
        warnings = [str(item).lower() for item in result_parts.get("warnings", [])]

        retrieval_conf = float(candidates[0].rerank_score or candidates[0].similarity_score) if candidates else 0.0
        template_conf = float(selected_template.get("confidence") or 0.0)
        slot_conf = self._slot_confidence(slots)
        mapping_conf = self._mapping_confidence(mapping)
        join_conf = float(join_plan.get("confidence", 1.0)) if isinstance(join_plan, dict) else 1.0
        ir_conf = 1.0 if ir_validation.get("is_valid", False) else 0.0
        validation_conf = 1.0 if validation.get("is_valid", validation.get("ok", False)) else 0.0

        confidence = (
            0.20 * retrieval_conf
            + 0.15 * template_conf
            + 0.15 * slot_conf
            + 0.20 * mapping_conf
            + 0.10 * join_conf
            + 0.10 * ir_conf
            + 0.10 * validation_conf
        )
        caps_applied: list[str] = []

        def apply_cap(name: str, ceiling: float) -> None:
            nonlocal confidence
            confidence = min(confidence, ceiling)
            if name not in caps_applied:
                caps_applied.append(name)

        if not ir_validation.get("is_valid", False):
            apply_cap("ir_validation_failed", 0.59)
        if not validation.get("is_valid", validation.get("ok", False)):
            apply_cap("sql_validation_failed", 0.59)
        metric_table = mapping.get("metric_table") if isinstance(mapping, dict) else getattr(mapping, "metric_table", None)
        dimension_table = mapping.get("dimension_table") if isinstance(mapping, dict) else getattr(mapping, "dimension_table", None)
        if mapping and not metric_table and self._needs_metric(selected_template):
            apply_cap("required_metric_missing", 0.45)
        if mapping and self._needs_dimension(selected_template) and not dimension_table:
            apply_cap("required_dimension_missing", 0.45)
        if join_plan and join_plan.get("warnings"):
            apply_cap("join_path_missing", 0.49)
        if self._needs_metric(selected_template) and not self._slot_ok(slots, "metric"):
            apply_cap("required_metric_missing", 0.49)
        if self._needs_dimension(selected_template) and not self._slot_ok(slots, "dimension"):
            apply_cap("required_dimension_missing", 0.49)
        if any("semantic grain" in warning or "product-level revenue could not" in warning for warning in warnings):
            apply_cap("semantic_grain_risk", 0.69)
        if any("date filter requested" in warning or "date column" in warning for warning in warnings):
            apply_cap("date_filter_missing", 0.69)
        if any("filter requested" in warning or "filter column" in warning for warning in warnings):
            apply_cap("filter_mapping_missing", 0.69)

        tier = "high" if confidence >= 0.80 else "medium" if confidence >= 0.60 else "low"
        final = round(max(0.0, min(1.0, confidence)), 4)
        breakdown = {
            "retrieval": round(retrieval_conf, 4),
            "template": round(template_conf, 4),
            "slots": round(slot_conf, 4),
            "schema_mapping": round(mapping_conf, 4),
            "join_planning": round(join_conf, 4),
            "ir_validation": ir_conf,
            "sql_validation": validation_conf,
            "caps_applied": caps_applied,
            "final": final,
        }
        return {
            "confidence": final,
            "confidence_tier": tier,
            "components": breakdown,
            "confidence_breakdown": breakdown,
        }

    @staticmethod
    def _slot_confidence(slots: dict[str, Any]) -> float:
        values = [float(value.get("confidence", 0.0)) for value in slots.values() if isinstance(value, dict)]
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _mapping_confidence(mapping: Any) -> float:
        if not mapping:
            return 0.0
        scores = list((mapping.get("match_scores") if isinstance(mapping, dict) else mapping.match_scores).values())
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _needs_dimension(selected_template: dict[str, Any]) -> bool:
        return selected_template.get("template_id") in {
            "count_by_dimension",
            "metric_by_dimension",
            "top_n_metric_by_dimension",
            "bottom_n_metric_by_dimension",
        }

    @staticmethod
    def _needs_metric(selected_template: dict[str, Any]) -> bool:
        return selected_template.get("template_id") in {
            "metric_summary",
            "metric_by_dimension",
            "top_n_metric_by_dimension",
            "bottom_n_metric_by_dimension",
            "trend_by_date",
        }

    @staticmethod
    def _slot_ok(slots: dict[str, Any], slot_name: str) -> bool:
        slot = slots.get(slot_name)
        if not isinstance(slot, dict):
            return False
        has_value = "value" not in slot or slot.get("value") is not None
        return bool(has_value and float(slot.get("confidence", 0.0)) >= 0.55)
