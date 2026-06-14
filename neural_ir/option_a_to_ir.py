from __future__ import annotations

from typing import Any

from inference.prediction_models import SchemaMapping
from inference.runtime_join_planner import RuntimeJoinPlanner
from inference.runtime_schema_context import RuntimeSchemaContext
from ir.option_c_to_ir import OptionCToIRConverter
from ir.query_ir_models import QueryIR

from .schema_linearizer import _schema_dict_from_serialized


COUNT_INTENTS = {"count_records", "count_by_dimension"}
METRIC_INTENTS = {
    "metric_summary",
    "metric_by_dimension",
    "top_n_metric_by_dimension",
    "bottom_n_metric_by_dimension",
    "trend_by_date",
}
DIMENSION_INTENTS = {
    "count_by_dimension",
    "metric_by_dimension",
    "top_n_metric_by_dimension",
    "bottom_n_metric_by_dimension",
}


class OptionAToIRConverter:
    def __init__(self) -> None:
        self.join_planner = RuntimeJoinPlanner()
        self.option_c_converter = OptionCToIRConverter()

    def convert(
        self,
        question: str,
        schema: dict | str | Any,
        decoded_prediction: dict,
    ) -> QueryIR:
        runtime_schema = _normalize_schema(schema)
        schema_context = RuntimeSchemaContext(runtime_schema)
        intent = decoded_prediction.get("template_id") or decoded_prediction.get("intent") or "show_records"
        base_table = decoded_prediction.get("base_table") or _first_table(schema_context)
        metric_col = decoded_prediction.get("metric_column") or {}
        dimension_col = decoded_prediction.get("dimension_column") or {}
        date_col = decoded_prediction.get("date_column") or {}
        filter_col = decoded_prediction.get("filter_column") or {}
        warnings: list[str] = []

        mapping = SchemaMapping(
            base_table=base_table,
            metric_name=_metric_name(decoded_prediction, metric_col),
            metric_table=metric_col.get("table"),
            metric_column=metric_col.get("column"),
            metric_expression=None,
            metric_aggregation=_metric_aggregation(intent, decoded_prediction),
            metric_alias=_metric_alias(decoded_prediction, metric_col),
            dimension_name=_column_alias(dimension_col),
            dimension_table=dimension_col.get("table"),
            dimension_column=dimension_col.get("column"),
            entity_table=base_table,
            date_table=date_col.get("table"),
            date_column=date_col.get("column"),
            filter_table=filter_col.get("table"),
            filter_column=filter_col.get("column"),
            warnings=warnings,
        )
        self._apply_metric_expression(decoded_prediction, schema_context, mapping, warnings)
        if intent in METRIC_INTENTS and not mapping.metric_expression and mapping.metric_table and mapping.metric_column:
            mapping.metric_expression = f"{mapping.metric_table}.{mapping.metric_column}"

        slots = self._slots(question, intent, decoded_prediction, mapping, warnings)
        required_tables = self._required_tables(intent, mapping)
        if not base_table:
            base_table = RuntimeJoinPlanner.choose_base_table(mapping.metric_table, mapping.entity_table, required_tables or schema_context.get_tables())
            mapping.base_table = base_table
        join_plan = self.join_planner.plan_joins(schema_context, base_table, required_tables)

        query_ir = self.option_c_converter.convert(
            question=question,
            normalized_question=" ".join(str(question).lower().split()),
            intent=intent,
            template_id=intent,
            slots=slots,
            schema_mapping=mapping,
            join_plan=join_plan,
            validation_context={"schema_context": schema_context.serialize_for_debug()},
            dialect=schema_context.dialect,
        )
        query_ir.metadata["source_model"] = "option_a"
        return query_ir

    def _apply_metric_expression(
        self,
        decoded_prediction: dict[str, Any],
        schema_context: RuntimeSchemaContext,
        mapping: SchemaMapping,
        warnings: list[str],
    ) -> None:
        expression_type = decoded_prediction.get("metric_expression_type") or "none"
        if expression_type == "count_star":
            mapping.metric_name = "record_count"
            mapping.metric_column = "*"
            mapping.metric_expression = "*"
            mapping.metric_aggregation = "COUNT"
            mapping.metric_alias = "record_count"
            return
        if expression_type != "product_revenue_expression":
            return
        if schema_context.has_column("order_items", "quantity") and schema_context.has_column("order_items", "price"):
            mapping.base_table = mapping.base_table or "order_items"
            mapping.metric_table = "order_items"
            mapping.metric_column = None
            mapping.metric_expression = "order_items.quantity * order_items.price"
            mapping.metric_aggregation = "SUM"
            mapping.metric_name = "revenue"
            mapping.metric_alias = "revenue"
            if "order_items" not in mapping.semantic_required_tables:
                mapping.semantic_required_tables.append("order_items")
        else:
            warnings.append("product revenue expression requested but order_items.quantity and order_items.price were not available")

    def _slots(
        self,
        question: str,
        intent: str,
        decoded_prediction: dict[str, Any],
        mapping: SchemaMapping,
        warnings: list[str],
    ) -> dict[str, Any]:
        limit = int(decoded_prediction.get("limit") or 100)
        slots: dict[str, Any] = {
            "limit": {"value": limit, "source": "neural_ir", "confidence": 0.7},
            "sort_direction": {
                "value": _order_direction(intent, decoded_prediction.get("order_direction")),
                "source": "neural_ir",
                "confidence": 0.7,
            },
        }
        if mapping.metric_name:
            slots["metric"] = {"value": mapping.metric_name, "source": "neural_ir", "confidence": 0.7}
        if mapping.dimension_name:
            slots["dimension"] = {"value": mapping.dimension_name, "source": "neural_ir", "confidence": 0.7}
        if intent == "trend_by_date":
            grain = decoded_prediction.get("date_grain")
            slots["date_grain"] = {"value": "year" if grain == "year" else "month", "source": "neural_ir", "confidence": 0.7}
        date_filter_type = decoded_prediction.get("date_filter_type")
        if date_filter_type and date_filter_type != "none":
            slots["date_filter"] = {"value": date_filter_type.replace("_", " "), "source": "neural_ir", "confidence": 0.6}
        filter_operator = decoded_prediction.get("filter_operator")
        if mapping.filter_column and filter_operator and filter_operator != "none":
            value = _extract_filter_value(question, mapping.filter_column)
            if value is not None:
                slots["filter_column"] = {"value": mapping.filter_column, "source": "neural_ir", "confidence": 0.6}
                slots["filter_operator"] = {"value": filter_operator, "source": "neural_ir", "confidence": 0.6}
                slots["filter_value"] = {"value": value, "source": "question", "confidence": 0.55}
            elif intent == "simple_filter":
                warnings.append("filter column was predicted but filter value could not be extracted from the question")
        return slots

    @staticmethod
    def _required_tables(intent: str, mapping: SchemaMapping) -> list[str]:
        tables = [mapping.base_table, mapping.entity_table]
        if intent in METRIC_INTENTS or intent in COUNT_INTENTS:
            tables.append(mapping.metric_table)
        if intent in DIMENSION_INTENTS:
            tables.append(mapping.dimension_table)
        if mapping.date_table:
            tables.append(mapping.date_table)
        if mapping.filter_table:
            tables.append(mapping.filter_table)
        tables.extend(mapping.semantic_required_tables)
        return [table for table in dict.fromkeys(tables) if table]


def _normalize_schema(schema: dict | str | Any) -> dict[str, Any] | Any:
    if isinstance(schema, str):
        return _schema_dict_from_serialized(schema)
    if isinstance(schema, dict) and "serialized_schema" in schema and not schema.get("tables"):
        return _schema_dict_from_serialized(str(schema.get("serialized_schema") or ""))
    return schema


def _first_table(schema_context: RuntimeSchemaContext) -> str | None:
    tables = schema_context.get_tables()
    return tables[0] if tables else None


def _metric_name(decoded: dict[str, Any], metric_col: dict[str, str]) -> str | None:
    if decoded.get("metric_expression_type") == "count_star":
        return "record_count"
    column = metric_col.get("column")
    if not column:
        return None
    name = column.replace("_", " ")
    if column in {"amount", "sales"}:
        return "sales"
    return name


def _metric_alias(decoded: dict[str, Any], metric_col: dict[str, str]) -> str | None:
    if decoded.get("metric_expression_type") == "count_star":
        return "record_count"
    column = metric_col.get("column")
    if not column:
        return None
    if column == "amount":
        return "revenue"
    return column


def _metric_aggregation(intent: str, decoded: dict[str, Any]) -> str:
    if intent in COUNT_INTENTS or decoded.get("metric_expression_type") == "count_star":
        return "COUNT"
    aggregation = str(decoded.get("metric_aggregation") or "SUM").upper()
    return "SUM" if aggregation == "NONE" else aggregation


def _column_alias(column: dict[str, str] | None) -> str | None:
    if not column:
        return None
    name = str(column.get("column") or "")
    return name.replace("_name", "").replace("_", " ") if name else None


def _order_direction(intent: str, direction: str | None) -> str:
    if direction in {"ASC", "DESC"}:
        return direction
    if intent == "bottom_n_metric_by_dimension":
        return "ASC"
    if intent in {"top_n_metric_by_dimension", "metric_by_dimension", "count_by_dimension"}:
        return "DESC"
    return "ASC"


def _extract_filter_value(question: str, column: str) -> str | int | float | None:
    import re

    text = str(question)
    escaped = re.escape(column.replace("_", " "))
    patterns = [
        rf"\b{escaped}\b\s*(?:is|=|equals|equal to)?\s*([A-Za-z0-9_.-]+)",
        rf"\b(?:in|with|where|for)\s+{escaped}\s+([A-Za-z0-9_.-]+)",
    ]
    lowered = text.lower().replace("_", " ")
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            return _coerce_value(match.group(1))
    if column == "status":
        for status in ["pending", "completed", "shipped", "cancelled", "canceled"]:
            if re.search(rf"\b{status}\b", lowered):
                return "cancelled" if status == "canceled" else status
    return None


def _coerce_value(value: str) -> str | int | float:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
