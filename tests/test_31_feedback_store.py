from __future__ import annotations

from pathlib import Path

from feedback.feedback_models import QueryFeedback
from feedback.feedback_store import FeedbackStore


def test_feedback_saved_without_passwords(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    feedback = QueryFeedback(
        db_type="postgres",
        schema_fingerprint="abc",
        question="list all users",
        generated_query_ir={"metadata": {"connection": {"password": "secret", "dsn": "postgres://u:secret@h/db"}}},
        generated_sql="SELECT users.id FROM users LIMIT 100",
        source_model="generic_direct_planner",
        validation_status={"is_valid": True},
        execution_status={"ok": True},
        user_rating="correct",
        feedback_tags=["good_answer"],
    )

    feedback_id = FeedbackStore(path).append(feedback)
    text = path.read_text(encoding="utf-8")

    assert feedback_id.startswith("fb_")
    assert "secret" not in text
    assert "***" in text
    assert FeedbackStore(path).load_all()[0].question == "list all users"
