from __future__ import annotations

from execution_eval.result_comparator import ResultComparator


def test_same_rows_match_order_insensitive_and_numeric_tolerance() -> None:
    pred = {"success": True, "columns": ["id", "amount"], "rows": [{"id": 2, "amount": 1.0000001}, {"id": 1, "amount": 2.0}]}
    gold = {"success": True, "columns": ["id", "amount"], "rows": [{"id": 1, "amount": 2.0}, {"id": 2, "amount": 1.0}]}

    assert ResultComparator().compare_results(pred, gold)["result_match"] is True


def test_different_row_count_and_columns_fail() -> None:
    comparator = ResultComparator()
    row_count = comparator.compare_results({"success": True, "columns": ["id"], "rows": [{"id": 1}]}, {"success": True, "columns": ["id"], "rows": [{"id": 1}, {"id": 2}]})
    columns = comparator.compare_results({"success": True, "columns": ["id"], "rows": [{"id": 1}]}, {"success": True, "columns": ["name"], "rows": [{"name": "A"}]})

    assert row_count["row_count_match"] is False
    assert columns["column_match"] is False
