from __future__ import annotations

from feedback.feedback_models import QueryFeedback
from feedback.feedback_to_ir_examples import FeedbackToIRExampleBuilder


def _generated_ir() -> dict:
    return {
        "query_ir_id": "bad",
        "question": "list all users",
        "normalized_question": "list all users",
        "intent": "show_records",
        "template_id": "show_records",
        "dialect": "sqlite",
        "base_table": "assignments",
        "required_tables": ["assignments", "users"],
        "metrics": [],
        "dimensions": [],
        "filters": [],
        "date_filters": [],
        "joins": [{"left_table": "assignments", "left_column": "user_id", "right_table": "users", "right_column": "id", "condition": "assignments.user_id = users.id"}],
        "metadata": {},
    }


def test_corrected_sql_converted_to_query_ir_and_bad_ir_hard_negative() -> None:
    row = QueryFeedback(
        db_type="sqlite",
        schema_fingerprint="schema1",
        question="list all users",
        generated_query_ir=_generated_ir(),
        generated_sql="SELECT assignments.id FROM assignments JOIN users ON assignments.user_id = users.id LIMIT 100",
        user_rating="incorrect",
        corrected_sql="SELECT users.id, users.name FROM users LIMIT 100",
        feedback_tags=["unnecessary_join"],
    )

    result = FeedbackToIRExampleBuilder().build_examples([row])

    assert result["summary"]["positive_examples"] == 1
    assert result["summary"]["hard_negatives"] == 1
    assert result["positive_examples"][0]["query_ir"]["base_table"] == "users"
    assert result["hard_negatives"][0]["query_ir"]["base_table"] == "assignments"
