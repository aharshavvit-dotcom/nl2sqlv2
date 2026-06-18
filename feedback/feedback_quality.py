from __future__ import annotations

from typing import Any

from .feedback_models import QueryFeedback


class FeedbackQualityChecker:
    def assess(self, feedback: QueryFeedback) -> dict[str, Any]:
        issues: list[str] = []
        if not feedback.question.strip():
            issues.append("missing_question")
        if feedback.user_rating in {"incorrect", "partially_correct"} and not (
            feedback.corrected_sql or feedback.corrected_query_ir or feedback.generated_query_ir
        ):
            issues.append("incorrect_feedback_without_correction_or_generated_ir")
        if feedback.user_rating == "correct" and not feedback.generated_query_ir:
            issues.append("correct_feedback_without_generated_ir")
        return {"is_usable": not issues, "issues": issues}
