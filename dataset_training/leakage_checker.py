from __future__ import annotations

from itertools import combinations
from typing import Any

from .utils import normalize_text


class DatasetLeakageChecker:
    def check_database_leakage(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        dbs = {
            name: {str(row.get("db_id")) for row in rows if row.get("db_id")}
            for name, rows in splits.items()
            if name != "unsupported"
        }
        overlap: dict[str, list[str]] = {}
        for left, right in combinations(sorted(dbs), 2):
            shared = sorted(dbs[left] & dbs[right])
            if shared:
                overlap[f"{left}__{right}"] = shared
        train_unseen = sorted((dbs.get("train", set()) | dbs.get("validation", set())) & dbs.get("unseen_db_test", set()))
        return {
            "has_database_leakage": bool(train_unseen),
            "database_overlap": overlap,
            "train_unseen_overlap": train_unseen,
        }

    def check_question_leakage(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return self._text_overlap(splits, "question", "has_question_leakage", "question_overlap_count")

    def check_sql_leakage(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return self._text_overlap(splits, "source_sql", "has_sql_leakage", "sql_overlap_count")

    def run_all_checks(self, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        database = self.check_database_leakage(splits)
        question = self.check_question_leakage(splits)
        sql = self.check_sql_leakage(splits)
        result = {
            **database,
            **question,
            **sql,
        }
        result["strict_passed"] = not (
            result["has_database_leakage"]
            or result["has_question_leakage"]
            or result["has_sql_leakage"]
        )
        result["passed"] = not result["has_database_leakage"]
        return result

    @staticmethod
    def _text_overlap(
        splits: dict[str, list[dict[str, Any]]],
        key: str,
        flag_name: str,
        count_name: str,
    ) -> dict[str, Any]:
        train_values = {normalize_text(row.get(key)) for row in splits.get("train", []) if row.get(key)}
        other_values = {
            normalize_text(row.get(key))
            for split_name in ["validation", "test", "unseen_db_test"]
            for row in splits.get(split_name, [])
            if row.get(key)
        }
        shared = {value for value in train_values & other_values if value}
        return {flag_name: bool(shared), count_name: len(shared), f"{key}_overlap": sorted(shared)}
