from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .prediction_models import RetrievedCandidate, RuntimeSlot
from .runtime_schema_context import RuntimeSchemaContext


METRIC_SYNONYMS = {
    "revenue": ["sales", "revenue", "amount", "value", "order value", "total sales"],
    "quantity": ["qty", "quantity", "units", "units sold"],
    "order_count": ["orders", "transactions", "order count", "number of orders", "count"],
    "profit": ["profit", "margin"],
    "discount": ["discount", "markdown"],
    "average_order_value": ["average order value", "avg order value", "aov", "average revenue"],
}

DIMENSION_SYNONYMS = {
    "customer": ["customer", "customers", "client", "clients", "buyer", "buyers"],
    "product": ["product", "products", "item", "items", "sku"],
    "region": ["region", "regions", "area", "zone", "territory"],
    "status": ["status", "state", "condition"],
    "month": ["month", "monthly"],
    "year": ["year", "yearly"],
    "category": ["category", "categories"],
    "store": ["store", "stores"],
    "brand": ["brand", "brands"],
    "city": ["city", "cities"],
    "state": ["state", "states"],
    "customer_segment": ["segment", "customer segment", "customer tier", "tier"],
    "sales_rep": ["sales rep", "sales reps", "representative", "representatives", "salesperson", "salespeople"],
}


class SlotResolver:
    def resolve_slots(
        self,
        question: str,
        selected_template: dict[str, Any],
        candidates: list[RetrievedCandidate],
        schema_context: RuntimeSchemaContext,
        synonym_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        q = question.lower()
        slots: dict[str, RuntimeSlot] = {}
        slots["metric"] = self._metric_slot(q, candidates, selected_template)
        slots["dimension"] = self._dimension_slot(q, candidates, selected_template)
        slots["entity"] = self._entity_slot(q, schema_context)
        slots["limit"] = self._limit_slot(q)
        slots["sort_direction"] = RuntimeSlot(
            slot_name="sort_direction",
            value="ASC" if any(word in q for word in ["bottom", "lowest", "least", "worst"]) else "DESC",
            source="question",
            confidence=0.9,
        )
        slots["date_grain"] = self._date_grain_slot(q)
        slots["date_filter"] = self._date_filter_slot(q)
        slots["filter_column"] = RuntimeSlot(slot_name="filter_column", value=None, source="default", confidence=0.0)
        slots["filter_value"] = RuntimeSlot(slot_name="filter_value", value=None, source="default", confidence=0.0)

        template_id = selected_template.get("template_id")
        if template_id in {"count_records"} and not slots["metric"].value:
            slots["metric"] = RuntimeSlot(slot_name="metric", value="order_count", source="default", confidence=0.72)
        if template_id in {"metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension"} and not slots["dimension"].value:
            voted = self._candidate_vote(candidates, "dimension")
            if voted:
                slots["dimension"] = RuntimeSlot(slot_name="dimension", value=voted, source="retrieved_example", confidence=0.5)

        clarification_questions = []
        if template_id not in {"show_records", "count_records"} and not slots["metric"].value:
            clarification_questions.append("Which metric should I aggregate?")
        if template_id in {"metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension", "count_by_dimension"} and not slots["dimension"].value:
            clarification_questions.append("Which dimension should I group by?")

        return {
            "slots": {key: value.model_dump() for key, value in slots.items()},
            "clarification_questions": clarification_questions,
        }

    def _metric_slot(self, q: str, candidates: list[RetrievedCandidate], selected_template: dict[str, Any]) -> RuntimeSlot:
        for metric, aliases in METRIC_SYNONYMS.items():
            if any(alias in q for alias in aliases):
                return RuntimeSlot(slot_name="metric", value=metric, source="question", confidence=0.92)
        voted = self._candidate_vote(candidates, "metric")
        if voted and voted not in {"*", "None"}:
            return RuntimeSlot(slot_name="metric", value=str(voted), source="retrieved_example", confidence=0.45)
        if selected_template.get("template_id") in {"count_records", "count_by_dimension"}:
            return RuntimeSlot(slot_name="metric", value="order_count", source="default", confidence=0.8)
        return RuntimeSlot(slot_name="metric", value="revenue", source="default", confidence=0.55)

    def _dimension_slot(self, q: str, candidates: list[RetrievedCandidate], selected_template: dict[str, Any]) -> RuntimeSlot:
        if "by month" in q or "monthly" in q:
            return RuntimeSlot(slot_name="dimension", value="month", source="question", confidence=0.95)
        if "by year" in q or "yearly" in q:
            return RuntimeSlot(slot_name="dimension", value="year", source="question", confidence=0.95)
        for dimension, aliases in DIMENSION_SYNONYMS.items():
            if any(re.search(rf"\b{re.escape(alias)}\b", q) for alias in aliases):
                return RuntimeSlot(slot_name="dimension", value=dimension, source="question", confidence=0.9)
        by_match = re.search(r"\bby\s+([a-z_ ]+)", q)
        if by_match:
            return RuntimeSlot(slot_name="dimension", value=by_match.group(1).strip(), source="question", confidence=0.65)
        voted = self._candidate_vote(candidates, "dimension")
        return RuntimeSlot(slot_name="dimension", value=voted, source="retrieved_example" if voted else "default", confidence=0.45 if voted else 0.0)

    @staticmethod
    def _entity_slot(q: str, schema_context: RuntimeSchemaContext) -> RuntimeSlot:
        for table in schema_context.get_tables():
            if table.lower() in q or table.lower().rstrip("s") in q:
                return RuntimeSlot(slot_name="entity", value=table, source="question", confidence=0.85)
        for preferred in ["orders", "sales", "transactions", "invoices", "order_items"]:
            if schema_context.has_table(preferred):
                return RuntimeSlot(slot_name="entity", value=preferred, source="schema_match", confidence=0.72)
        tables = schema_context.get_tables()
        return RuntimeSlot(slot_name="entity", value=tables[0] if tables else None, source="default", confidence=0.35)

    @staticmethod
    def _limit_slot(q: str) -> RuntimeSlot:
        match = re.search(r"\b(?:top|first|show|limit)?\s*(\d{1,4})\b", q)
        value = min(int(match.group(1)), 1000) if match else 100
        return RuntimeSlot(slot_name="limit", value=value, source="question" if match else "default", confidence=0.9 if match else 0.65)

    @staticmethod
    def _date_grain_slot(q: str) -> RuntimeSlot:
        if "month" in q or "monthly" in q:
            return RuntimeSlot(slot_name="date_grain", value="month", source="question", confidence=0.9)
        if "year" in q or "yearly" in q:
            return RuntimeSlot(slot_name="date_grain", value="year", source="question", confidence=0.9)
        return RuntimeSlot(slot_name="date_grain", value=None, source="default", confidence=0.0)

    @staticmethod
    def _date_filter_slot(q: str) -> RuntimeSlot:
        for phrase in ["last month", "this month", "last year", "this year", "last 30 days"]:
            if phrase in q:
                return RuntimeSlot(slot_name="date_filter", value=phrase, source="question", confidence=0.75)
        return RuntimeSlot(slot_name="date_filter", value=None, source="default", confidence=0.0)

    @staticmethod
    def _candidate_vote(candidates: list[RetrievedCandidate], slot_name: str) -> str | None:
        values = [str(item.slots.get(slot_name)) for item in candidates if item.slots.get(slot_name)]
        if not values:
            return None
        return Counter(values).most_common(1)[0][0]
