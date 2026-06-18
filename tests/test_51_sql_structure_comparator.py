from __future__ import annotations

from execution_eval.sql_structure_comparator import SQLStructureComparator


def test_identical_sql_scores_high() -> None:
    result = SQLStructureComparator().compare("SELECT users.id FROM users LIMIT 100", "SELECT users.id FROM users LIMIT 100", {})
    assert result["structure_score"] >= 0.99


def test_wrong_base_table_and_wrong_filter_detected() -> None:
    result = SQLStructureComparator().compare(
        "SELECT products.id FROM products WHERE products.status = 'x' LIMIT 100",
        "SELECT users.id FROM users WHERE users.role = 'admin' LIMIT 100",
        {},
    )
    assert "wrong_base_table" in result["errors"]
    assert "wrong_filter" in result["errors"]


def test_missing_and_unnecessary_join_detected() -> None:
    comparator = SQLStructureComparator()
    unnecessary = comparator.compare(
        "SELECT users.id FROM users JOIN assignments ON assignments.user_id = users.id LIMIT 100",
        "SELECT users.id FROM users LIMIT 100",
        {},
    )
    missing = comparator.compare(
        "SELECT users.id FROM users LIMIT 100",
        "SELECT users.id FROM users JOIN assignments ON assignments.user_id = users.id LIMIT 100",
        {},
    )
    assert "unnecessary_join" in unnecessary["errors"]
    assert "missing_join" in missing["errors"]
