from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field


class IRExpression(BaseModel):
    table: str | None = None
    column: str | None = None
    expression: str | None = None
    alias: str | None = None


class IRMetric(BaseModel):
    name: str
    aggregation: str
    table: str | None = None
    column: str | None = None
    expression: str
    alias: str
    source_slot: str | None = None
    confidence: float = 1.0


class IRDimension(BaseModel):
    name: str
    table: str
    column: str
    expression: str
    alias: str
    source_slot: str | None = None
    confidence: float = 1.0


class IRFilter(BaseModel):
    name: str | None = None
    table: str
    column: str
    expression: str
    operator: Literal[
        "equals",
        "not_equals",
        "contains",
        "in",
        "not_in",
        "greater_than",
        "greater_equal",
        "less_than",
        "less_equal",
    ]
    value: str | int | float | list[Any]
    value_type: str = "string"
    raw_text: str | None = None
    confidence: float = 1.0


class IRDateFilter(BaseModel):
    date_table: str
    date_column: str
    date_expression: str
    filter_type: Literal["relative_range", "absolute_range", "grain"]
    start_date: str | None = None
    end_date: str | None = None
    date_grain: str | None = None
    raw_text: str | None = None
    confidence: float = 1.0


class IRJoin(BaseModel):
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    join_type: str = "INNER"
    condition: str
    path_order: int
    confidence: float = 1.0


class IROrderBy(BaseModel):
    expression: str
    alias: str | None = None
    direction: Literal["ASC", "DESC"]
    source: Literal["metric", "dimension", "date", "count", "explicit", "default"]


class QueryIR(BaseModel):
    query_ir_id: str
    question: str
    normalized_question: str
    intent: str
    template_id: str | None = None
    dialect: str = "sqlite"
    base_table: str | None = None
    required_tables: list[str] = Field(default_factory=list)
    metrics: list[IRMetric] = Field(default_factory=list)
    dimensions: list[IRDimension] = Field(default_factory=list)
    filters: list[IRFilter] = Field(default_factory=list)
    date_filters: list[IRDateFilter] = Field(default_factory=list)
    joins: list[IRJoin] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    order_by: list[IROrderBy] = Field(default_factory=list)
    limit: int = 100
    select_mode: Literal["records", "aggregate", "trend", "count"] = "records"
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IRValidationIssue(BaseModel):
    severity: Literal["error", "warning", "info"]
    issue_type: str
    message: str
    suggested_action: str | None = None


class IRValidationResult(BaseModel):
    is_valid: bool
    issues: list[IRValidationIssue] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def diff_query_ir(predicted_ir: QueryIR | dict[str, Any] | None, gold_ir: QueryIR | dict[str, Any] | None) -> dict[str, Any]:
    """Return a deterministic, slot-level semantic diff between two QueryIR values."""
    predicted = _ir_payload(predicted_ir)
    gold = _ir_payload(gold_ir)

    predicted_metrics = _items(predicted, "metrics")
    gold_metrics = _items(gold, "metrics")
    predicted_dimensions = _items(predicted, "dimensions")
    gold_dimensions = _items(gold, "dimensions")
    predicted_filters = _items(predicted, "filters")
    gold_filters = _items(gold, "filters")

    checks = {
        "intent_match": _scalar(predicted.get("intent")) == _scalar(gold.get("intent")),
        "base_table_match": _scalar(predicted.get("base_table")) == _scalar(gold.get("base_table")),
        "projection_match": _signatures(predicted_dimensions, ("expression", "table", "column"))
        == _signatures(gold_dimensions, ("expression", "table", "column")),
        "metric_match": _signatures(predicted_metrics, ("expression", "column"))
        == _signatures(gold_metrics, ("expression", "column")),
        "aggregation_match": _signatures(predicted_metrics, ("aggregation",))
        == _signatures(gold_metrics, ("aggregation",)),
        "filter_column_match": _signatures(predicted_filters, ("table", "column"))
        == _signatures(gold_filters, ("table", "column")),
        "filter_value_match": _signatures(predicted_filters, ("value",), normalize_values=True)
        == _signatures(gold_filters, ("value",), normalize_values=True),
        "filter_operator_match": _signatures(predicted_filters, ("operator",))
        == _signatures(gold_filters, ("operator",)),
        "date_filter_match": _signatures(
            _items(predicted, "date_filters"),
            ("date_table", "date_column", "filter_type", "start_date", "end_date", "date_grain"),
            normalize_values=True,
        )
        == _signatures(
            _items(gold, "date_filters"),
            ("date_table", "date_column", "filter_type", "start_date", "end_date", "date_grain"),
            normalize_values=True,
        ),
        "join_match": _signatures(
            _items(predicted, "joins"),
            ("left_table", "left_column", "right_table", "right_column", "join_type"),
        )
        == _signatures(
            _items(gold, "joins"),
            ("left_table", "left_column", "right_table", "right_column", "join_type"),
        ),
        "group_by_match": _sequence(predicted.get("group_by"), ordered=False)
        == _sequence(gold.get("group_by"), ordered=False),
        "order_by_match": _signatures(
            _items(predicted, "order_by"), ("expression", "alias", "direction"), ordered=True
        )
        == _signatures(_items(gold, "order_by"), ("expression", "alias", "direction"), ordered=True),
        "limit_match": int(predicted.get("limit") or 100) == int(gold.get("limit") or 100),
    }
    checks["dimension_match"] = checks["projection_match"]
    checks["selected_columns_match"] = checks["projection_match"]
    checks["filters_match"] = bool(
        checks["filter_column_match"]
        and checks["filter_value_match"]
        and checks["filter_operator_match"]
    )

    failure_order = (
        "filter_column",
        "filter_value",
        "filter_operator",
        "projection",
        "aggregation",
        "metric",
        "base_table",
        "join",
        "group_by",
        "order_by",
        "limit",
        "date_filter",
        "intent",
    )
    checks["primary_failure_slot"] = next(
        (slot for slot in failure_order if checks.get(f"{slot}_match") is False),
        None,
    )
    checks["all_slots_match"] = checks["primary_failure_slot"] is None
    return checks


def _ir_payload(value: QueryIR | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, QueryIR):
        return value.model_dump()
    return dict(value or {})


def _items(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [dict(item) for item in payload.get(key) or [] if isinstance(item, dict)]


def _scalar(value: Any) -> str:
    return _normalize(value)


def _normalize(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ",".join(_normalize(item) for item in value) + "]"
    text = str(value if value is not None else "").strip().lower().replace('"', "").replace("`", "")
    text = re.sub(r"\b[a-z_][a-z0-9_]*\.", "", text)
    return " ".join(text.split())


def _signatures(
    items: list[dict[str, Any]],
    keys: tuple[str, ...],
    *,
    ordered: bool = False,
    normalize_values: bool = False,
) -> list[tuple[str, ...]]:
    values = [
        tuple(_normalize(item.get(key)) if normalize_values or key != "value" else str(item.get(key)) for key in keys)
        for item in items
    ]
    return values if ordered else sorted(values)


def _sequence(value: Any, *, ordered: bool) -> list[str]:
    values = [_normalize(item) for item in (value or [])]
    return values if ordered else sorted(values)
