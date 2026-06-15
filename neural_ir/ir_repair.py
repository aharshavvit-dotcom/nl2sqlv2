from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from inference.runtime_join_planner import RuntimeJoinPlanner
from inference.runtime_schema_context import RuntimeSchemaContext
from ir.query_ir_models import IRDateFilter, IRDimension, IRFilter, IRJoin, IRMetric, QueryIR


METRIC_INTENTS = {"metric_summary", "metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension", "trend_by_date"}
DIMENSION_INTENTS = {"metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension", "count_by_dimension"}
SENSITIVE_MARKERS = ("email", "phone", "password", "token", "secret", "ssn", "address")


class OptionAIRRepairer:
    def __init__(self) -> None:
        self.join_planner = RuntimeJoinPlanner()

    def repair(self, query_ir, schema: dict, question: str, validation_result=None) -> dict:
        repairs: list[str] = []
        warnings: list[str] = []
        try:
            ir = _to_query_ir(query_ir)
        except Exception as exc:
            return {
                "query_ir": query_ir.model_dump() if hasattr(query_ir, "model_dump") else query_ir,
                "repairs_applied": [],
                "repair_warnings": [f"could not parse QueryIR for repair: {exc}"],
                "repair_success": False,
            }
        schema_context = RuntimeSchemaContext(schema)
        self._repair_limit(ir, question, repairs)
        self._repair_count_metric(ir, repairs)
        self._repair_missing_metric(ir, schema_context, question, repairs, warnings)
        self._repair_product_revenue(ir, schema_context, question, repairs)
        self._repair_date_grain(ir, schema_context, question, repairs)
        self._repair_filter_value(ir, question, repairs, warnings)
        self._repair_missing_dimension(ir, schema_context, question, repairs, warnings)
        self._remove_sensitive_references(ir, repairs, warnings)
        self._repair_joins(ir, schema_context, repairs, warnings)
        ir.metadata.setdefault("debug", {})
        ir.metadata["option_a_repairs_applied"] = repairs
        ir.metadata["option_a_repair_warnings"] = warnings
        return {
            "query_ir": ir.model_dump(),
            "repairs_applied": repairs,
            "repair_warnings": warnings,
            "repair_success": not any("could not" in warning.lower() for warning in warnings),
        }

    @staticmethod
    def _repair_limit(ir: QueryIR, question: str, repairs: list[str]) -> None:
        text = question.lower()
        if ir.limit and ir.limit > 0:
            return
        match = re.search(r"\b(?:top|bottom|first|last)\s+(\d{1,3})\b", text)
        if match:
            ir.limit = int(match.group(1))
            repairs.append(f"inferred_limit_{ir.limit}")
        elif "top" in text or "bottom" in text:
            ir.limit = 5
            repairs.append("inferred_limit_5")
        else:
            ir.limit = 100
            repairs.append("added_default_limit_100")

    @staticmethod
    def _repair_count_metric(ir: QueryIR, repairs: list[str]) -> None:
        if ir.intent not in {"count_records", "count_by_dimension"} or ir.metrics:
            return
        ir.metrics.append(
            IRMetric(
                name="record_count",
                aggregation="COUNT",
                table=ir.base_table,
                column="*",
                expression="*",
                alias="record_count",
                confidence=0.65,
            )
        )
        ir.select_mode = "count"
        repairs.append("added_count_star_metric")

    @staticmethod
    def _repair_missing_metric(
        ir: QueryIR,
        schema_context: RuntimeSchemaContext,
        question: str,
        repairs: list[str],
        warnings: list[str],
    ) -> None:
        if ir.intent not in METRIC_INTENTS or ir.metrics:
            return
        text = question.lower()
        if not ({"revenue", "sales", "amount", "total"} & set(re.findall(r"[a-z0-9]+", text))):
            warnings.append("metric intent has no metric and no revenue/sales cue")
            return
        candidate = _best_numeric_metric(schema_context, preferred=("revenue", "sales", "amount", "total", "price"))
        if not candidate:
            warnings.append("could not repair missing metric because no numeric metric candidate was found")
            return
        table, column = candidate
        ir.metrics.append(
            IRMetric(
                name="revenue" if column in {"amount", "revenue", "sales", "total"} else column,
                aggregation="SUM",
                table=table,
                column=column,
                expression=f"{table}.{column}",
                alias="revenue" if column == "amount" else column,
                confidence=0.55,
            )
        )
        ir.required_tables = list(dict.fromkeys([*(ir.required_tables or []), table]))
        ir.base_table = ir.base_table or table
        repairs.append("added_revenue_metric_candidate")

    @staticmethod
    def _repair_product_revenue(ir: QueryIR, schema_context: RuntimeSchemaContext, question: str, repairs: list[str]) -> None:
        text = question.lower()
        has_product_dimension = any("product" in (dimension.name or "").lower() or dimension.table == "products" for dimension in ir.dimensions)
        has_revenue_metric = any(metric.name.lower() in {"revenue", "sales", "total_sales"} or "amount" in metric.expression.lower() for metric in ir.metrics)
        if "product" not in text and not has_product_dimension:
            return
        if not ({"revenue", "sales"} & set(re.findall(r"[a-z0-9]+", text))) and not has_revenue_metric:
            return
        if not (schema_context.has_column("order_items", "quantity") and schema_context.has_column("order_items", "price")):
            return
        if any("order_items.quantity" in metric.expression and "order_items.price" in metric.expression for metric in ir.metrics):
            return
        if not ir.metrics:
            ir.metrics.append(
                IRMetric(
                    name="revenue",
                    aggregation="SUM",
                    table="order_items",
                    column=None,
                    expression="order_items.quantity * order_items.price",
                    alias="revenue",
                    confidence=0.6,
                )
            )
        else:
            metric = ir.metrics[0]
            metric.name = "revenue"
            metric.aggregation = "SUM"
            metric.table = "order_items"
            metric.column = None
            metric.expression = "order_items.quantity * order_items.price"
            metric.alias = "revenue"
            metric.confidence = min(metric.confidence, 0.79)
        ir.base_table = "order_items"
        ir.required_tables = list(dict.fromkeys([*(ir.required_tables or []), "order_items", "products"]))
        ir.joins = []
        repairs.append("corrected_product_revenue_to_item_level_expression")

    @staticmethod
    def _repair_date_grain(ir: QueryIR, schema_context: RuntimeSchemaContext, question: str, repairs: list[str]) -> None:
        if ir.intent != "trend_by_date":
            return
        text = question.lower()
        grain = "year" if "year" in text or "yearly" in text else "month" if "month" in text or "monthly" in text else None
        if not grain:
            return
        existing = next((item for item in ir.date_filters if item.filter_type == "grain"), None)
        if existing:
            if not existing.date_grain:
                existing.date_grain = grain
                repairs.append(f"inferred_date_grain_{grain}")
            return
        candidate = _best_date_column(schema_context)
        if not candidate:
            return
        table, column = candidate
        ir.date_filters.append(
            IRDateFilter(
                date_table=table,
                date_column=column,
                date_expression=f"{table}.{column}",
                filter_type="grain",
                date_grain=grain,
                raw_text=grain,
                confidence=0.55,
            )
        )
        ir.group_by = list(dict.fromkeys([*(ir.group_by or []), "DATE_GRAIN(date)"]))
        ir.required_tables = list(dict.fromkeys([*(ir.required_tables or []), table]))
        repairs.append(f"inferred_date_grain_{grain}")

    @staticmethod
    def _repair_filter_value(ir: QueryIR, question: str, repairs: list[str], warnings: list[str]) -> None:
        if ir.intent != "simple_filter":
            return
        for item in ir.filters:
            if item.value is not None and item.value != "":
                continue
            value = _extract_filter_value(question, item.column)
            if value is None:
                warnings.append(f"could not extract filter value for {item.table}.{item.column}")
                continue
            item.value = value
            item.raw_text = str(value)
            repairs.append("extracted_missing_filter_value")

    @staticmethod
    def _repair_missing_dimension(
        ir: QueryIR,
        schema_context: RuntimeSchemaContext,
        question: str,
        repairs: list[str],
        warnings: list[str],
    ) -> None:
        if ir.intent not in DIMENSION_INTENTS or ir.dimensions:
            return
        candidate = _best_dimension_column(schema_context, question)
        if not candidate:
            warnings.append("could not repair missing dimension because no dimension candidate was found")
            return
        table, column = candidate
        ir.dimensions.append(
            IRDimension(
                name=column.replace("_name", "").replace("_", " "),
                table=table,
                column=column,
                expression=f"{table}.{column}",
                alias=column,
                confidence=0.55,
            )
        )
        ir.required_tables = list(dict.fromkeys([*(ir.required_tables or []), table]))
        ir.group_by = list(dict.fromkeys([*(ir.group_by or []), f"{table}.{column}"]))
        repairs.append("added_schema_linked_dimension")

    @staticmethod
    def _remove_sensitive_references(ir: QueryIR, repairs: list[str], warnings: list[str]) -> None:
        before = len(ir.dimensions), len(ir.filters), len(ir.date_filters)
        ir.dimensions = [item for item in ir.dimensions if not _is_sensitive(item.column)]
        ir.filters = [item for item in ir.filters if not _is_sensitive(item.column)]
        ir.date_filters = [item for item in ir.date_filters if not _is_sensitive(item.date_column)]
        after = len(ir.dimensions), len(ir.filters), len(ir.date_filters)
        if before != after:
            repairs.append("removed_sensitive_column_references")
            warnings.append("removed sensitive column references before SQL rendering")

    def _repair_joins(
        self,
        ir: QueryIR,
        schema_context: RuntimeSchemaContext,
        repairs: list[str],
        warnings: list[str],
    ) -> None:
        required = _required_tables(ir)
        if not ir.base_table and required:
            ir.base_table = RuntimeJoinPlanner.choose_base_table(
                ir.metrics[0].table if ir.metrics else None,
                ir.dimensions[0].table if ir.dimensions else None,
                required,
            )
            repairs.append("inferred_base_table_for_join_repair")
        if not ir.base_table:
            return
        required = list(dict.fromkeys([ir.base_table, *required]))
        if len(required) <= 1:
            ir.required_tables = required
            return
        if ir.joins:
            ir.required_tables = required
            return
        plan = self.join_planner.plan_joins(schema_context, ir.base_table, required)
        if plan.warnings:
            warnings.extend(plan.warnings)
        ir.required_tables = plan.required_tables
        ir.joins = [
            IRJoin(
                left_table=step["from_table"],
                left_column=step["from_column"],
                right_table=step["to_table"],
                right_column=step["to_column"],
                join_type=step.get("join_type", "INNER"),
                condition=step.get("condition") or f"{step['from_table']}.{step['from_column']} = {step['to_table']}.{step['to_column']}",
                path_order=idx,
                confidence=plan.confidence,
            )
            for idx, step in enumerate(plan.join_steps)
        ]
        if plan.join_steps:
            repairs.append("added_missing_join_plan")


def _to_query_ir(query_ir: Any) -> QueryIR:
    if isinstance(query_ir, QueryIR):
        return query_ir.model_copy(deep=True)
    payload = deepcopy(query_ir)
    if hasattr(QueryIR, "model_validate"):
        return QueryIR.model_validate(payload)
    return QueryIR.parse_obj(payload)


def _best_numeric_metric(schema_context: RuntimeSchemaContext, preferred: tuple[str, ...]) -> tuple[str, str] | None:
    columns = []
    for qualified in schema_context.get_numeric_columns():
        table, column = qualified.split(".", 1)
        if _is_sensitive(column):
            continue
        score = 0
        for idx, token in enumerate(preferred):
            if token in column.lower():
                score = max(score, len(preferred) - idx)
        columns.append((score, table, column))
    if not columns:
        return None
    columns.sort(key=lambda item: (-item[0], item[1], item[2]))
    _, table, column = columns[0]
    return table, column


def _best_date_column(schema_context: RuntimeSchemaContext) -> tuple[str, str] | None:
    columns = schema_context.get_date_columns()
    if not columns:
        return None
    table, column = columns[0].split(".", 1)
    return table, column


def _best_dimension_column(schema_context: RuntimeSchemaContext, question: str) -> tuple[str, str] | None:
    terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    candidates = []
    for qualified in schema_context.get_text_columns():
        table, column = qualified.split(".", 1)
        if _is_sensitive(column):
            continue
        tokens = set(re.findall(r"[a-z0-9]+", f"{table} {column}".replace("_", " ")))
        score = len(tokens & terms)
        if column.endswith("_name"):
            score += 1
        candidates.append((score, table, column))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    _, table, column = candidates[0]
    return table, column


def _required_tables(ir: QueryIR) -> list[str]:
    tables = list(ir.required_tables or [])
    tables.extend(metric.table for metric in ir.metrics if metric.table)
    tables.extend(dimension.table for dimension in ir.dimensions if dimension.table)
    tables.extend(item.table for item in ir.filters if item.table)
    tables.extend(item.date_table for item in ir.date_filters if item.date_table)
    return [table for table in dict.fromkeys(tables) if table]


def _extract_filter_value(question: str, column: str) -> str | int | float | None:
    lowered = question.lower().replace("_", " ")
    column_text = column.replace("_", " ")
    patterns = [
        rf"\b{re.escape(column_text)}\b\s*(?:is|=|equals|equal to)?\s*([A-Za-z0-9_.-]+)",
        rf"\b(?:in|with|where|for)\s+{re.escape(column_text)}\s+([A-Za-z0-9_.-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return _coerce(match.group(1))
    known_values = ["completed", "pending", "shipped", "cancelled", "canceled", "west", "east", "electronics"]
    for value in known_values:
        if re.search(rf"\b{value}\b", lowered):
            return "cancelled" if value == "canceled" else value
    return None


def _coerce(value: str) -> str | int | float:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _is_sensitive(column: str | None) -> bool:
    name = str(column or "").lower()
    return any(marker in name for marker in SENSITIVE_MARKERS)
