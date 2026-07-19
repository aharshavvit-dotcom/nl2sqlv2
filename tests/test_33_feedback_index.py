"""
Purpose: Protects ir unit behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from pathlib import Path

from retrieval.feedback_index import FeedbackIndex


def test_feedback_index_retrieves_similar_correction(tmp_path: Path) -> None:
    examples = [
        {
            "example_id": "fb_users",
            "question": "list all users",
            "schema_fingerprint": "schema1",
            "feedback_tags": ["unnecessary_join"],
            "query_ir": {"intent": "show_records", "base_table": "users", "required_tables": ["users"], "joins": []},
        },
        {
            "example_id": "fb_orders",
            "question": "orders where status is completed",
            "schema_fingerprint": "schema2",
            "query_ir": {"intent": "simple_filter", "base_table": "orders", "required_tables": ["orders"], "joins": []},
        },
    ]
    index = FeedbackIndex()
    index.build(examples)
    path = tmp_path / "feedback_index.pkl"
    index.save(path)

    loaded = FeedbackIndex.load(path)
    results = loaded.search("list users", {"schema_fingerprint": "schema1", "tables": {"users": {"columns": {"id": {}}}}})

    assert results[0]["example_id"] == "fb_users"
    assert results[0]["source"] == "feedback_index"
