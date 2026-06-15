from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta
from typing import Any
from uuid import uuid4

from .query_ir_models import IRDateFilter, IRDimension, IRFilter, IRJoin, IRMetric, IROrderBy, QueryIR
from .semantic_metric_resolver import SemanticMetricResolver


METRIC_INTENTS = {
    "metric_summary",
    "metric_by_dimension",
    "top_n_metric_by_dimension",
    "bottom_n_metric_by_dimension",
    "trend_by_date",
}
COUNT_INTENTS = {"count_records", "count_by_dimension"}
DIMENSION_INTENTS = {
    "metric_by_dimension",
    "top_n_metric_by_dimension",
    "bottom_n_metric_by_dimension",
    "count_by_dimension",
}


class RetrievalIRConverter:
    """Converts retrieval runtime state into QueryIR.

    Formerly named ``OptionCToIRConverter``.
    """
    def convert(
        self,
        question: str,
        normalized_question: str,
        intent: str,
        template_id: str | None,
        slots: dict[str, Any],
        schema_mapping: dict[str, Any] | object,
        join_plan: dict[str, Any] | object | None,
        validation_context: dict[str, Any] | None = None,
        dialect: str = "sqlite",
    ) -> QueryIR:
        mapping = self._dump(schema_mapping)
        mapping = self._apply_semantic_metric_resolution(mapping, slots, validation_context)
        plan = self._dump(join_plan) if join_plan else {}
        template = template_id or intent
        warnings = list(mapping.get("warnings") or [])
        base_table = plan.get("base_table") or mapping.get("base_table") or mapping.get("metric_table") or mapping.get("entity_table")
        limit = min(max(int(self._slot_value(slots, "limit", 100) or 100), 1), 1000)

        metrics = self._metrics(template, slots, mapping, base_table, warnings)
        dimensions = self._dimensions(template, slots, mapping, warnings)
        date_filters = self._date_filters(template, slots, mapping, warnings)
        filters = self._filters(slots, mapping, warnings)
        joins = self._joins(plan)
        group_by = self._group_by(template, dimensions, date_filters)
        order_by = self._order_by(template, metrics, dimensions, date_filters, slots)
        required_tables = self._required_tables(base_table, mapping, joins, template)
        select_mode = self._select_mode(template)

        return QueryIR(
            query_ir_id=f"qir_{uuid4().hex}",
            question=question,
            normalized_question=normalized_question,
            intent=intent or template or "unknown",
            template_id=template,
            dialect=dialect or "sqlite",
            base_table=base_table,
            required_tables=required_tables,
            metrics=metrics,
            dimensions=dimensions,
            filters=filters,
            date_filters=date_filters,
            joins=joins,
            group_by=group_by,
            order_by=order_by,
            limit=limit,
            select_mode=select_mode,
            warnings=warnings,
            metadata={
                "slots": slots,
                "schema_mapping": mapping,
                "join_plan": plan,
                "validation_context": validation_context or {},
            },
        )

    @staticmethod
    def _metrics(
        template_id: str | None,
        slots: dict[str, Any],
        mapping: dict[str, Any],
        base_table: str | None,
        warnings: list[str],
    ) -> list[IRMetric]:
        if template_id in COUNT_INTENTS:
            return [
                IRMetric(
                    name="record_count",
                    aggregation="COUNT",
                    table=base_table,
                    column="*",
                    expression="*",
                    alias="record_count",
                    source_slot="metric",
                    confidence=RetrievalIRConverter._slot_confidence(slots, "metric", 0.9),
                )
            ]
        if template_id not in METRIC_INTENTS:
            return []
        table = mapping.get("metric_table")
        column = mapping.get("metric_column")
        expression = mapping.get("metric_expression")
        aggregation = str(mapping.get("metric_aggregation") or "SUM").upper()
        name = str(mapping.get("metric_name") or RetrievalIRConverter._slot_value(slots, "metric", "metric"))
        if mapping.get("semantic_grain_risk"):
            warnings.append("semantic grain risk: product-level revenue needs item-level quantity/price columns")
        if not expression and table and column:
            expression = f"{table}.{column}"
        if not table or not expression:
            warnings.append("missing metric mapping")
            return []
        alias = str(mapping.get("metric_alias") or ("revenue" if name in {"sales", "revenue"} else ("record_count" if aggregation == "COUNT" else name)))
        return [
            IRMetric(
                name=name,
                aggregation=aggregation,
                table=table,
                column=column,
                expression=expression,
                alias=alias,
                source_slot="metric",
                confidence=RetrievalIRConverter._slot_confidence(slots, "metric"),
            )
        ]

    @staticmethod
    def _dimensions(
        template_id: str | None,
        slots: dict[str, Any],
        mapping: dict[str, Any],
        warnings: list[str],
    ) -> list[IRDimension]:
        if template_id not in DIMENSION_INTENTS:
            return []
        name = str(mapping.get("dimension_name") or RetrievalIRConverter._slot_value(slots, "dimension", "dimension"))
        table = mapping.get("dimension_table")
        column = mapping.get("dimension_column")
        if not table or not column:
            warnings.append("missing dimension mapping")
            return []
        return [
            IRDimension(
                name=name,
                table=table,
                column=column,
                expression=f"{table}.{column}",
                alias=name,
                source_slot="dimension",
                confidence=RetrievalIRConverter._slot_confidence(slots, "dimension"),
            )
        ]

    @staticmethod
    def _date_filters(
        template_id: str | None,
        slots: dict[str, Any],
        mapping: dict[str, Any],
        warnings: list[str],
    ) -> list[IRDateFilter]:
        date_table = mapping.get("date_table")
        date_column = mapping.get("date_column")
        date_filters: list[IRDateFilter] = []
        if template_id == "trend_by_date":
            grain = str(RetrievalIRConverter._slot_value(slots, "date_grain", None) or RetrievalIRConverter._slot_value(slots, "dimension", None) or "month")
            if date_table and date_column:
                date_filters.append(
                    IRDateFilter(
                        date_table=date_table,
                        date_column=date_column,
                        date_expression=f"{date_table}.{date_column}",
                        filter_type="grain",
                        start_date=None,
                        end_date=None,
                        date_grain="year" if grain == "year" else "month",
                        raw_text=f"by {grain}",
                        confidence=RetrievalIRConverter._slot_confidence(slots, "date_grain", 0.9),
                    )
                )
            else:
                warnings.append("date grain requested but no date column was mapped")

        raw_filter = RetrievalIRConverter._slot_value(slots, "date_filter", None)
        if raw_filter:
            if not date_table or not date_column:
                warnings.append("date filter requested but no date column was mapped")
                return date_filters
            start, end = RetrievalIRConverter._date_range(str(raw_filter))
            if start or end:
                date_filters.append(
                    IRDateFilter(
                        date_table=date_table,
                        date_column=date_column,
                        date_expression=f"{date_table}.{date_column}",
                        filter_type="relative_range",
                        start_date=start.isoformat() if start else None,
                        end_date=end.isoformat() if end else None,
                        date_grain=None,
                        raw_text=str(raw_filter),
                        confidence=RetrievalIRConverter._slot_confidence(slots, "date_filter", 0.8),
                    )
                )
        return date_filters

    @staticmethod
    def _filters(slots: dict[str, Any], mapping: dict[str, Any], warnings: list[str]) -> list[IRFilter]:
        filter_name = RetrievalIRConverter._slot_value(slots, "filter_column", None)
        filter_value = RetrievalIRConverter._slot_value(slots, "filter_value", None)
        if filter_name is None or filter_value is None:
            return []
        table = mapping.get("filter_table")
        column = mapping.get("filter_column")
        if not table or not column:
            warnings.append("filter requested but no filter column was mapped")
            return []
        operator = str(RetrievalIRConverter._slot_value(slots, "filter_operator", "equals") or "equals")
        if operator not in {"equals", "not_equals", "contains", "in", "not_in", "greater_than", "greater_equal", "less_than", "less_equal"}:
            operator = "equals"
        return [
            IRFilter(
                name=str(filter_name),
                table=table,
                column=column,
                expression=f"{table}.{column}",
                operator=operator,
                value=filter_value,
                value_type="number" if isinstance(filter_value, (int, float)) else "string",
                raw_text=str(filter_value),
                confidence=RetrievalIRConverter._slot_confidence(slots, "filter_value", 0.8),
            )
        ]

    @staticmethod
    def _joins(plan: dict[str, Any]) -> list[IRJoin]:
        joins: list[IRJoin] = []
        for index, step in enumerate(plan.get("join_steps") or [], start=1):
            current = step.get("current")
            from_table = step.get("from_table")
            to_table = step.get("to_table")
            from_column = step.get("from_column")
            to_column = step.get("to_column")
            if not all([from_table, to_table, from_column, to_column]):
                continue
            if current == to_table:
                left_table, left_column = to_table, to_column
                right_table, right_column = from_table, from_column
            else:
                left_table, left_column = from_table, from_column
                right_table, right_column = to_table, to_column
            joins.append(
                IRJoin(
                    left_table=left_table,
                    left_column=left_column,
                    right_table=right_table,
                    right_column=right_column,
                    join_type=str(step.get("join_type") or "INNER"),
                    condition=str(step.get("condition") or f"{from_table}.{from_column} = {to_table}.{to_column}"),
                    path_order=index,
                )
            )
        return joins

    @staticmethod
    def _group_by(template_id: str | None, dimensions: list[IRDimension], date_filters: list[IRDateFilter]) -> list[str]:
        if template_id == "trend_by_date":
            grain = next((item for item in date_filters if item.filter_type == "grain"), None)
            if grain:
                return [f"DATE_GRAIN({grain.date_expression}, {grain.date_grain or 'month'})"]
            return []
        if template_id in DIMENSION_INTENTS and dimensions:
            return [dimensions[0].expression]
        return []

    @staticmethod
    def _order_by(
        template_id: str | None,
        metrics: list[IRMetric],
        dimensions: list[IRDimension],
        date_filters: list[IRDateFilter],
        slots: dict[str, Any],
    ) -> list[IROrderBy]:
        if template_id == "trend_by_date":
            return [IROrderBy(expression="period", alias="period", direction="ASC", source="date")]
        if template_id == "bottom_n_metric_by_dimension" and metrics:
            return [IROrderBy(expression=metrics[0].alias, alias=metrics[0].alias, direction="ASC", source="metric")]
        if template_id in {"top_n_metric_by_dimension", "metric_by_dimension"} and metrics:
            return [IROrderBy(expression=metrics[0].alias, alias=metrics[0].alias, direction="DESC", source="metric")]
        if template_id == "count_by_dimension":
            direction = str(RetrievalIRConverter._slot_value(slots, "sort_direction", "DESC") or "DESC").upper()
            return [IROrderBy(expression="record_count", alias="record_count", direction="ASC" if direction == "ASC" else "DESC", source="count")]
        if dimensions:
            return [IROrderBy(expression=dimensions[0].alias, alias=dimensions[0].alias, direction="ASC", source="dimension")]
        return []

    @staticmethod
    def _required_tables(base_table: str | None, mapping: dict[str, Any], joins: list[IRJoin], template_id: str | None) -> list[str]:
        include_metric = template_id in METRIC_INTENTS or template_id in COUNT_INTENTS
        tables = [
            base_table,
            mapping.get("metric_table") if include_metric else None,
            mapping.get("dimension_table"),
            mapping.get("entity_table"),
            mapping.get("date_table"),
            mapping.get("filter_table"),
            *(mapping.get("semantic_required_tables") or []),
            *[join.left_table for join in joins],
            *[join.right_table for join in joins],
        ]
        return [table for table in dict.fromkeys(tables) if table]

    @staticmethod
    def _select_mode(template_id: str | None) -> str:
        if template_id in COUNT_INTENTS:
            return "count"
        if template_id == "trend_by_date":
            return "trend"
        if template_id in METRIC_INTENTS:
            return "aggregate"
        return "records"

    @staticmethod
    def _date_range(raw_text: str) -> tuple[date | None, date | None]:
        phrase = raw_text.lower().replace("_", " ").strip()
        today = date.today()
        first_this_month = today.replace(day=1)
        if phrase == "last month":
            previous_month = first_this_month - timedelta(days=1)
            start = previous_month.replace(day=1)
            return start, first_this_month
        if phrase == "this month":
            days_in_month = monthrange(today.year, today.month)[1]
            return first_this_month, today.replace(day=days_in_month) + timedelta(days=1)
        if phrase == "last year":
            return date(today.year - 1, 1, 1), date(today.year, 1, 1)
        if phrase == "this year":
            return date(today.year, 1, 1), date(today.year + 1, 1, 1)
        if phrase == "last 30 days":
            return today - timedelta(days=30), today + timedelta(days=1)
        return None, None

    @staticmethod
    def _slot_value(slots: dict[str, Any], key: str, default: Any = None) -> Any:
        slot = slots.get(key)
        if isinstance(slot, dict):
            return slot.get("value", default)
        return default if slot is None else slot

    @staticmethod
    def _slot_confidence(slots: dict[str, Any], key: str, default: float = 1.0) -> float:
        slot = slots.get(key)
        if isinstance(slot, dict):
            return float(slot.get("confidence", default) or 0.0)
        return default

    @staticmethod
    def _apply_semantic_metric_resolution(
        mapping: dict[str, Any],
        slots: dict[str, Any],
        validation_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        schema_context = (validation_context or {}).get("schema_context")
        if not schema_context:
            return mapping

        updated = dict(mapping)
        resolution = SemanticMetricResolver().resolve_metric_expression(
            metric_name=str(updated.get("metric_name") or RetrievalIRConverter._slot_value(slots, "metric", "") or ""),
            dimension_name=updated.get("dimension_name") or RetrievalIRConverter._slot_value(slots, "dimension", None),
            schema_context=schema_context,
            current_metric_table=updated.get("metric_table"),
            current_metric_column=updated.get("metric_column"),
        )
        if resolution.get("base_table"):
            updated["base_table"] = resolution["base_table"]
        if resolution.get("metric_expression"):
            updated["metric_table"] = resolution.get("metric_table")
            updated["metric_column"] = resolution.get("metric_column")
            updated["metric_expression"] = resolution.get("metric_expression")
            updated["metric_aggregation"] = resolution.get("metric_aggregation") or updated.get("metric_aggregation")
            updated["metric_alias"] = resolution.get("metric_alias") or updated.get("metric_alias")
        if resolution.get("required_tables"):
            updated["semantic_required_tables"] = list(resolution["required_tables"])
        if resolution.get("semantic_grain_risk"):
            updated["semantic_grain_risk"] = True
        warnings = list(updated.get("warnings") or [])
        for warning in resolution.get("warnings") or []:
            if warning not in warnings:
                warnings.append(warning)
        updated["warnings"] = warnings
        return updated

    @staticmethod
    def _dump(value: dict[str, Any] | object) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        return dict(value)  # type: ignore[arg-type]


# Backward-compatible alias
OptionCToIRConverter = RetrievalIRConverter
"""Deprecated alias. Use ``RetrievalIRConverter``."""
