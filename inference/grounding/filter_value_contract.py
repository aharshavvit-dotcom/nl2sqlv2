from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


class QueryTimeContext(BaseModel):
    current_datetime: datetime
    timezone: str = "UTC"
    fiscal_year_start_month: int | None = None


class ExtractedLiteral(BaseModel):
    literal_id: str
    raw_text: str
    normalized_value: Any
    value_type: Literal[
        "string",
        "integer",
        "decimal",
        "date",
        "datetime",
        "boolean",
        "null",
        "percentage",
        "currency",
        "list",
        "range",
        "year",
        "quarter",
        "month",
        "relative date",
        "numeric_value",
    ]
    span_start: int
    span_end: int
    extraction_method: str
    extraction_confidence: float


class GroundedFilterCandidate(BaseModel):
    literal_id: str
    table_name: str
    column_name: str
    operator: str
    normalized_value: Any
    grounding_score: float
    grounding_signals: dict[str, float] = Field(default_factory=dict)
    ambiguity_score: float


class GroundedFilter(BaseModel):
    literal_id: str
    selected_candidate: GroundedFilterCandidate | None = None
    candidate_columns: list[GroundedFilterCandidate] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_question: str | None = None
