from __future__ import annotations

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
