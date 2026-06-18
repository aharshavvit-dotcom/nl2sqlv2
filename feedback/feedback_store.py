from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .feedback_models import QueryFeedback


SENSITIVE_KEY_MARKERS = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "connection_string",
    "sqlalchemy_url",
    "dsn",
)


class FeedbackStore:
    def __init__(self, path: str | Path = "data/feedback/query_feedback.jsonl"):
        self.path = Path(path)

    def append(self, feedback: QueryFeedback | dict[str, Any]) -> str:
        row = feedback if isinstance(feedback, QueryFeedback) else QueryFeedback.model_validate(feedback)
        payload = self._redact(row.model_dump())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return row.feedback_id

    def load_all(self) -> list[QueryFeedback]:
        if not self.path.exists():
            return []
        rows: list[QueryFeedback] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(QueryFeedback.model_validate(json.loads(line)))
        return rows

    def filter(self, rating: str | None = None, tags: list[str] | None = None) -> list[QueryFeedback]:
        rows = self.load_all()
        if rating is not None:
            rows = [row for row in rows if row.user_rating == rating]
        if tags:
            requested = set(tags)
            rows = [row for row in rows if requested.issubset(set(row.feedback_tags))]
        return rows

    @classmethod
    def _redact(cls, value: Any) -> Any:
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key).lower()
                if any(marker in key_text for marker in SENSITIVE_KEY_MARKERS):
                    redacted[key] = "***"
                else:
                    redacted[key] = cls._redact(item)
            return redacted
        if isinstance(value, list):
            return [cls._redact(item) for item in value]
        return value


def append_feedback(path: str | Path, payload: dict[str, Any]) -> str:
    """Backward-compatible helper used by older UI code."""
    rating_map = {
        "thumbs_up": "correct",
        "thumbs_down": "incorrect",
        "skip": "not_sure",
    }
    normalized = {
        "question": payload.get("question") or "",
        "generated_sql": payload.get("generated_sql") or payload.get("sql"),
        "user_rating": rating_map.get(str(payload.get("rating")), payload.get("rating") or "not_sure"),
        "user_comment": payload.get("user_comment") or payload.get("notes"),
        "db_type": payload.get("db_type") or "unknown",
        "schema_fingerprint": payload.get("schema_fingerprint") or "unknown",
        "generated_query_ir": payload.get("generated_query_ir"),
        "source_model": payload.get("source_model"),
        "validation_status": payload.get("validation_status"),
        "execution_status": payload.get("execution_status"),
        "corrected_sql": payload.get("corrected_sql"),
        "corrected_query_ir": payload.get("corrected_query_ir"),
        "feedback_tags": payload.get("feedback_tags") or [],
    }
    return FeedbackStore(path).append(QueryFeedback.model_validate(normalized))


__all__ = ["FeedbackStore", "QueryFeedback", "append_feedback"]
