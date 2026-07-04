from __future__ import annotations

from collections import defaultdict
from typing import Any


class ExecutionReporter:
    def summarize(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(rows)
        execution_available = sum(1 for row in rows if row.get("execution_available"))
        execution_matches = sum(1 for row in rows if row.get("execution_match"))
        structure_matches = sum(1 for row in rows if row.get("structure", {}).get("structure_score", 0.0) >= 0.99)
        unnecessary = sum(1 for row in rows if "unnecessary_join" in row.get("structure", {}).get("errors", []))
        wrong_table = sum(1 for row in rows if "wrong_base_table" in row.get("structure", {}).get("errors", []))
        return {
            "summary": {
                "total_examples": total,
                "execution_available": execution_available,
                "execution_required": False,
                "execution_unavailable": execution_available == 0,
                "execution_unavailable_reason": "no_database_connection" if execution_available == 0 else None,
                "execution_status": (
                    "execution_unavailable" if execution_available == 0
                    else "execution_available_and_passed" if execution_matches == execution_available
                    else "execution_available_but_failed"
                ),
                "execution_match_rate": execution_matches / execution_available if execution_available else None,
                "structure_match_rate": structure_matches / total if total else 0.0,
                "unnecessary_join_rate": unnecessary / total if total else 0.0,
                "wrong_table_rate": wrong_table / total if total else 0.0,
            },
            "by_dataset": self._bucket(rows, "dataset_name"),
            "by_intent": self._bucket(rows, "intent"),
            "failures": [
                {
                    "example_id": row.get("example_id"),
                    "question": row.get("question"),
                    "errors": row.get("structure", {}).get("errors", []),
                    "structure_score": row.get("structure", {}).get("structure_score"),
                    "execution_match": row.get("execution_match"),
                }
                for row in rows
                if row.get("structure", {}).get("errors") or row.get("execution_match") is False
            ][:100],
            "examples": rows[:100],
        }

    @staticmethod
    def _bucket(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(row.get(key) or "unknown")].append(row)
        result = {}
        for name, items in buckets.items():
            total = len(items)
            result[name] = {
                "total_examples": total,
                "structure_match_rate": sum(1 for row in items if row.get("structure", {}).get("structure_score", 0.0) >= 0.99) / total if total else 0.0,
                "unnecessary_join_rate": sum(1 for row in items if "unnecessary_join" in row.get("structure", {}).get("errors", [])) / total if total else 0.0,
            }
        return result
