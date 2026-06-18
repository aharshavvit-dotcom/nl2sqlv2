from __future__ import annotations

from typing import Any

from .correction_parser import CorrectionParser
from .feedback_models import QueryFeedback


POSITIVE_RATINGS = {"correct"}
NEGATIVE_RATINGS = {"incorrect", "partially_correct"}


class FeedbackToIRExampleBuilder:
    def __init__(self, dialect: str = "sqlite"):
        self.dialect = dialect

    def build_examples(self, feedback_rows: list[QueryFeedback | dict[str, Any]]) -> dict[str, Any]:
        positives: list[dict[str, Any]] = []
        hard_negatives: list[dict[str, Any]] = []
        safety_regressions: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for raw in feedback_rows:
            row = raw if isinstance(raw, QueryFeedback) else QueryFeedback.model_validate(raw)
            if row.user_rating == "unsafe":
                safety_regressions.append(self._safety_regression(row))
                if row.generated_query_ir:
                    hard_negatives.append(self._hard_negative(row, reason="unsafe_feedback"))
                continue

            if row.user_rating in POSITIVE_RATINGS:
                if row.generated_query_ir and self._validation_passed(row.validation_status):
                    positives.append(self._positive(row, row.generated_query_ir, row.generated_sql, "user_marked_correct"))
                continue

            if row.user_rating in NEGATIVE_RATINGS:
                correction = self._corrected_query_ir(row)
                if correction.get("query_ir"):
                    positives.append(self._positive(row, correction["query_ir"], row.corrected_sql, "user_correction"))
                elif correction.get("error"):
                    errors.append({"feedback_id": row.feedback_id, "error": correction["error"]})
                if row.generated_query_ir:
                    hard_negatives.append(self._hard_negative(row, reason="incorrect_feedback"))

        return {
            "positive_examples": positives,
            "hard_negatives": hard_negatives,
            "safety_regressions": safety_regressions,
            "summary": {
                "feedback_rows": len(feedback_rows),
                "positive_examples": len(positives),
                "hard_negatives": len(hard_negatives),
                "safety_regressions": len(safety_regressions),
                "conversion_errors": len(errors),
            },
            "errors": errors,
        }

    def _corrected_query_ir(self, row: QueryFeedback) -> dict[str, Any]:
        if row.corrected_query_ir:
            return {"query_ir": row.corrected_query_ir}
        if not row.corrected_sql:
            return {}
        schema = self._schema_from_feedback(row)
        dialect = row.db_type if row.db_type in {"sqlite", "postgres", "postgresql"} else self.dialect
        parsed = CorrectionParser(dialect=dialect).corrected_sql_to_query_ir(row.question, row.corrected_sql, schema=schema)
        if parsed.get("success") and parsed.get("query_ir"):
            return {"query_ir": parsed["query_ir"]}
        return {"error": parsed.get("error_message") or parsed.get("unsupported_reason") or "corrected_sql_conversion_failed"}

    @staticmethod
    def _schema_from_feedback(row: QueryFeedback) -> dict[str, Any] | None:
        for query_ir in [row.corrected_query_ir, row.generated_query_ir]:
            if not isinstance(query_ir, dict):
                continue
            metadata = query_ir.get("metadata") or {}
            validation_context = metadata.get("validation_context") or {}
            schema_context = validation_context.get("schema_context") or {}
            if schema_context.get("tables"):
                return {"dialect": query_ir.get("dialect") or row.db_type, "tables": schema_context["tables"]}
        return None

    @staticmethod
    def _validation_passed(validation_status: dict[str, Any] | None) -> bool:
        if not validation_status:
            return True
        return bool(validation_status.get("is_valid", validation_status.get("ok", False)))

    @staticmethod
    def _positive(row: QueryFeedback, query_ir: dict[str, Any], sql: str | None, source: str) -> dict[str, Any]:
        return {
            "example_id": f"{row.feedback_id}_{source}",
            "source": "feedback",
            "feedback_id": row.feedback_id,
            "question": row.question,
            "schema_fingerprint": row.schema_fingerprint,
            "db_type": row.db_type,
            "query_ir": query_ir,
            "corrected_sql": sql,
            "intent": query_ir.get("intent"),
            "template_id": query_ir.get("template_id") or query_ir.get("intent"),
            "feedback_tags": row.feedback_tags,
            "label": "positive",
            "source_reason": source,
        }

    @staticmethod
    def _hard_negative(row: QueryFeedback, reason: str) -> dict[str, Any]:
        query_ir = row.generated_query_ir or {}
        return {
            "example_id": f"{row.feedback_id}_hard_negative",
            "source": "feedback",
            "feedback_id": row.feedback_id,
            "question": row.question,
            "schema_fingerprint": row.schema_fingerprint,
            "db_type": row.db_type,
            "query_ir": query_ir,
            "generated_sql": row.generated_sql,
            "intent": query_ir.get("intent"),
            "template_id": query_ir.get("template_id") or query_ir.get("intent"),
            "feedback_tags": row.feedback_tags,
            "negative_reason": reason,
            "label": "hard_negative",
        }

    @staticmethod
    def _safety_regression(row: QueryFeedback) -> dict[str, Any]:
        return {
            "case_id": f"{row.feedback_id}_safety",
            "feedback_id": row.feedback_id,
            "question": row.question,
            "generated_sql": row.generated_sql,
            "generated_query_ir": row.generated_query_ir,
            "schema_fingerprint": row.schema_fingerprint,
            "feedback_tags": row.feedback_tags,
            "expected_safe": False,
        }
