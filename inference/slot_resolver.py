from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .prediction_models import RetrievedCandidate, RuntimeSlot
from .runtime_schema_context import RuntimeSchemaContext
from .synonym_loader import load_synonym_config, normalize_section


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
        synonyms = self._synonym_config(synonym_config, schema_context)
        slots: dict[str, RuntimeSlot] = {}
        slots["metric"] = self._metric_slot(q, candidates, selected_template, synonyms["metrics"])
        slots["dimension"] = self._dimension_slot(q, candidates, selected_template, synonyms["dimensions"])
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
        filter_column, filter_value, filter_operator = self._filter_slots(q, synonyms["dimensions"])
        slots["filter_column"] = filter_column
        slots["filter_value"] = filter_value
        slots["filter_operator"] = filter_operator

        template_id = selected_template.get("template_id")
        if template_id in {"count_records"} and not slots["metric"].value:
            slots["metric"] = RuntimeSlot(slot_name="metric", value="record_count", source="default", confidence=0.72)
        if template_id in {"metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension"} and not slots["dimension"].value:
            voted = self._candidate_vote(candidates, "dimension")
            if voted:
                slots["dimension"] = RuntimeSlot(slot_name="dimension", value=voted, source="retrieved_example", confidence=0.5)

        return {
            "slots": {key: value.model_dump() for key, value in slots.items()},
            "clarification_questions": [],
        }

    def _metric_slot(
        self,
        q: str,
        candidates: list[RetrievedCandidate],
        selected_template: dict[str, Any],
        metric_synonyms: dict[str, list[str]],
    ) -> RuntimeSlot:
        metric_aliases = [
            (metric, alias)
            for metric, aliases in metric_synonyms.items()
            for alias in self._aliases(metric, aliases)
        ]
        for metric, alias in sorted(metric_aliases, key=lambda item: len(item[1]), reverse=True):
            if self._contains_alias(q, alias):
                return RuntimeSlot(slot_name="metric", value=metric, source="question", confidence=0.92)
        voted = self._candidate_vote(candidates, "metric")
        if voted and voted not in {"*", "None"}:
            return RuntimeSlot(slot_name="metric", value=str(voted), source="retrieved_example", confidence=0.45)
        if selected_template.get("template_id") in {"count_records", "count_by_dimension"}:
            return RuntimeSlot(slot_name="metric", value="record_count", source="default", confidence=0.8)
        return RuntimeSlot(slot_name="metric", value=None, source="default", confidence=0.0)

    def _dimension_slot(
        self,
        q: str,
        candidates: list[RetrievedCandidate],
        selected_template: dict[str, Any],
        dimension_synonyms: dict[str, list[str]],
    ) -> RuntimeSlot:
        if "by month" in q or "monthly" in q:
            return RuntimeSlot(slot_name="dimension", value="month", source="question", confidence=0.95)
        if "by year" in q or "yearly" in q:
            return RuntimeSlot(slot_name="dimension", value="year", source="question", confidence=0.95)
        for dimension, aliases in dimension_synonyms.items():
            if any(self._contains_alias(q, alias) for alias in self._aliases(dimension, aliases)):
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

    def _filter_slots(
        self,
        q: str,
        dimension_synonyms: dict[str, list[str]],
    ) -> tuple[RuntimeSlot, RuntimeSlot, RuntimeSlot]:
        stop_words = {
            "limit",
            "order",
            "group",
            "by",
            "from",
            "with",
            "where",
            "show",
            "list",
            "display",
            "records",
            "rows",
            "orders",
            "customers",
            "products",
            "stores",
        }
        candidates: list[tuple[int, str, str, str]] = []
        if "excluding " in q:
            excluded = q.split("excluding ", 1)[1].strip().split()[0]
            return (
                RuntimeSlot(slot_name="filter_column", value=None, source="default", confidence=0.0),
                RuntimeSlot(slot_name="filter_value", value=excluded, source="question", confidence=0.4),
                RuntimeSlot(slot_name="filter_operator", value="not_equals", source="question", confidence=0.4),
            )
        for dimension, aliases in dimension_synonyms.items():
            if dimension in {"month", "year"}:
                continue
            for alias in sorted(self._aliases(dimension, aliases), key=len, reverse=True):
                pattern = rf"(?:where|with|for|in|from)?\s*\b{re.escape(alias.lower())}\b\s*(is\s+not|!=|<>|equals|equal\s+to|is|=|as|in)?\s+([a-z0-9][a-z0-9 -]{{0,40}})"
                match = re.search(pattern, q)
                if not match:
                    continue
                raw_operator = (match.group(1) or "").strip()
                raw_value = match.group(2).strip()
                words = []
                for word in raw_value.split():
                    if word in stop_words:
                        break
                    words.append(word)
                value = " ".join(words).strip()
                if value:
                    operator = "not_equals" if raw_operator in {"is not", "!=", "<>"} else "equals"
                    candidates.append((match.start(), dimension, value, operator))
        if candidates:
            _, dimension, value, operator = max(candidates, key=lambda item: item[0])
            return (
                RuntimeSlot(slot_name="filter_column", value=dimension, source="question", confidence=0.82),
                RuntimeSlot(slot_name="filter_value", value=value, source="question", confidence=0.82),
                RuntimeSlot(slot_name="filter_operator", value=operator, source="question", confidence=0.82),
            )
        return (
            RuntimeSlot(slot_name="filter_column", value=None, source="default", confidence=0.0),
            RuntimeSlot(slot_name="filter_value", value=None, source="default", confidence=0.0),
            RuntimeSlot(slot_name="filter_operator", value="equals", source="default", confidence=0.0),
        )

    @staticmethod
    def _candidate_vote(candidates: list[RetrievedCandidate], slot_name: str) -> str | None:
        values = [str(item.slots.get(slot_name)) for item in candidates if item.slots.get(slot_name)]
        if not values:
            return None
        return Counter(values).most_common(1)[0][0]

    @staticmethod
    def _contains_alias(text: str, alias: str) -> bool:
        return re.search(rf"\b{re.escape(alias.lower())}\b", text) is not None

    @staticmethod
    def _aliases(key: str, aliases: list[str]) -> list[str]:
        return [key.replace("_", " "), key, *aliases]

    @staticmethod
    def _synonym_config(
        synonym_config: dict[str, Any] | None,
        schema_context: RuntimeSchemaContext,
    ) -> dict[str, dict[str, list[str]]]:
        from generic_planner.generic_slot_resolver import is_sample_retail_schema

        if synonym_config and (synonym_config.get("metrics") or synonym_config.get("dimensions")):
            configured = {
                "metrics": normalize_section(synonym_config.get("metrics") or {}),
                "dimensions": normalize_section(synonym_config.get("dimensions") or {}),
            }
        elif is_sample_retail_schema(schema_context.get_tables()):
            raw = load_synonym_config()
            configured = {
                "metrics": normalize_section(raw.get("metrics") or {}),
                "dimensions": normalize_section(raw.get("dimensions") or {}),
            }
        else:
            configured = {"metrics": {}, "dimensions": {}}

        # Connected schemas always contribute their own neutral vocabulary.  This
        # is the generic fallback; bundled retail terms are never needed for it.
        for qualified in schema_context.get_columns():
            table, column = qualified.split(".", 1)
            info = schema_context.column_info(table, column)
            if info.get("is_sensitive"):
                continue
            aliases = [column, column.replace("_", " "), f"{table} {column.replace('_', ' ')}"]
            configured["dimensions"].setdefault(column, aliases)
            if info.get("is_numeric") and not info.get("is_id"):
                configured["metrics"].setdefault(column, aliases)
        return configured
