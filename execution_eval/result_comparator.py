from __future__ import annotations

from math import isclose
from typing import Any


class ResultComparator:
    def __init__(self, numeric_tolerance: float = 1e-6):
        self.numeric_tolerance = numeric_tolerance

    def compare_results(self, predicted_result: dict[str, Any], gold_result: dict[str, Any]) -> dict[str, Any]:
        warnings: list[str] = []
        pred_success = bool(predicted_result.get("success", predicted_result.get("ok", False)))
        gold_success = bool(gold_result.get("success", gold_result.get("ok", False)))
        if not pred_success or not gold_success:
            return {
                "result_match": False,
                "result_score": 0.0,
                "column_match": False,
                "row_count_match": False,
                "value_match": False,
                "warnings": ["execution_failed"],
            }

        pred_columns = list(predicted_result.get("columns") or _columns_from_rows(predicted_result.get("rows") or []))
        gold_columns = list(gold_result.get("columns") or _columns_from_rows(gold_result.get("rows") or []))
        pred_rows = _normalize_rows(predicted_result.get("rows") or [], pred_columns)
        gold_rows = _normalize_rows(gold_result.get("rows") or [], gold_columns)

        column_match = [str(c).lower() for c in pred_columns] == [str(c).lower() for c in gold_columns]
        row_count_match = len(pred_rows) == len(gold_rows)
        order_sensitive = bool(predicted_result.get("order_sensitive") or gold_result.get("order_sensitive"))
        value_match = False
        if column_match and row_count_match:
            left = pred_rows
            right = gold_rows
            if not order_sensitive:
                left = sorted(left, key=lambda row: repr(row))
                right = sorted(right, key=lambda row: repr(row))
            value_match = all(self._row_equal(a, b) for a, b in zip(left, right))
        if not column_match:
            warnings.append("columns_differ")
        if not row_count_match:
            warnings.append("row_count_differs")
        if column_match and row_count_match and not value_match:
            warnings.append("values_differ")

        score = (0.30 if column_match else 0.0) + (0.25 if row_count_match else 0.0) + (0.45 if value_match else 0.0)
        return {
            "result_match": column_match and row_count_match and value_match,
            "result_score": round(score, 6),
            "column_match": column_match,
            "row_count_match": row_count_match,
            "value_match": value_match,
            "warnings": warnings,
        }

    def _row_equal(self, left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
        if len(left) != len(right):
            return False
        return all(self._value_equal(a, b) for a, b in zip(left, right))

    def _value_equal(self, left: Any, right: Any) -> bool:
        if left is None or right is None:
            return left is right
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return isclose(float(left), float(right), rel_tol=self.numeric_tolerance, abs_tol=self.numeric_tolerance)
        return str(left) == str(right)


def _columns_from_rows(rows: list[Any]) -> list[str]:
    if rows and isinstance(rows[0], dict):
        return list(rows[0].keys())
    return []


def _normalize_rows(rows: list[Any], columns: list[str]) -> list[tuple[Any, ...]]:
    normalized = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append(tuple(row.get(column) for column in columns))
        elif isinstance(row, (list, tuple)):
            normalized.append(tuple(row))
        else:
            normalized.append((row,))
    return normalized
