from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


FeedbackRating = Literal["correct", "incorrect", "partially_correct", "unsafe", "not_sure"]

ALLOWED_RATINGS = {"correct", "incorrect", "partially_correct", "unsafe", "not_sure"}
ALLOWED_FEEDBACK_TAGS = {
    "wrong_table",
    "wrong_join",
    "unnecessary_join",
    "wrong_metric",
    "wrong_dimension",
    "missing_filter",
    "wrong_filter",
    "missing_date_filter",
    "wrong_date_filter",
    "invalid_sql",
    "unsafe_sql",
    "good_answer",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class QueryFeedback(BaseModel):
    feedback_id: str = Field(default_factory=lambda: f"fb_{uuid4().hex}")
    created_at: str = Field(default_factory=utc_now_iso)
    db_type: str = "unknown"
    schema_fingerprint: str = "unknown"
    question: str
    generated_query_ir: dict[str, Any] | None = None
    generated_sql: str | None = None
    source_model: str | None = None
    validation_status: dict[str, Any] | None = None
    execution_status: dict[str, Any] | None = None
    user_rating: FeedbackRating
    user_comment: str | None = None
    corrected_sql: str | None = None
    corrected_query_ir: dict[str, Any] | None = None
    feedback_tags: list[str] = Field(default_factory=list)

    @field_validator("user_rating")
    @classmethod
    def validate_rating(cls, value: str) -> str:
        if value not in ALLOWED_RATINGS:
            raise ValueError(f"Unsupported feedback rating: {value}")
        return value

    @field_validator("feedback_tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        invalid = sorted(set(value) - ALLOWED_FEEDBACK_TAGS)
        if invalid:
            raise ValueError("Unsupported feedback tag(s): " + ", ".join(invalid))
        return list(dict.fromkeys(value))
