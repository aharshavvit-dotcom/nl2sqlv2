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

        retrieval_conf = float(candidates[0].rerank_score or candidates[0].similarity_score) if candidates else 0.0
        template_conf = float(selected_template.get("confidence") or 0.0)
        slot_conf = self._slot_confidence(slots)
        mapping_conf = self._mapping_confidence(mapping)
        validation_conf = 1.0 if validation.get("ok") else 0.0

        confidence = (
            0.30 * retrieval_conf
            + 0.20 * template_conf
            + 0.20 * slot_conf
            + 0.20 * mapping_conf
            + 0.10 * validation_conf
        )
        if not validation.get("ok"):
            confidence = min(confidence, 0.59)
        if mapping and (not mapping.metric_table or (self._needs_dimension(selected_template) and not mapping.dimension_table)):
            confidence = min(confidence, 0.45)
        if join_plan and join_plan.get("warnings"):
            confidence = min(confidence, 0.55)

        tier = "high" if confidence >= 0.80 else "medium" if confidence >= 0.60 else "low"
        return {
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "confidence_tier": tier,
            "components": {
                "retrieval": round(retrieval_conf, 4),
                "template": round(template_conf, 4),
                "slots": round(slot_conf, 4),
                "schema_mapping": round(mapping_conf, 4),
                "validation": validation_conf,
            },
        }

    @staticmethod
    def _slot_confidence(slots: dict[str, Any]) -> float:
        values = [float(value.get("confidence", 0.0)) for value in slots.values() if isinstance(value, dict)]
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _mapping_confidence(mapping: Any) -> float:
        if not mapping:
            return 0.0
        scores = list(mapping.match_scores.values())
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _needs_dimension(selected_template: dict[str, Any]) -> bool:
        return selected_template.get("template_id") in {
            "count_by_dimension",
            "metric_by_dimension",
            "top_n_metric_by_dimension",
            "bottom_n_metric_by_dimension",
            "trend_by_date",
        }
