from __future__ import annotations

from copy import deepcopy
from typing import Any


class HardNegativeCorpusBuilder:
    def build(
        self,
        examples: list[dict[str, Any]],
        max_negatives_per_example: int = 5,
    ) -> list[dict[str, Any]]:
        negatives: list[dict[str, Any]] = []
        table_pool = sorted(
            {
                table
                for row in examples
                for table in ((row.get("query_ir") or {}).get("required_tables") or [])
                if table
            }
        )
        for row in examples:
            gold_ir = row.get("query_ir") or {}
            row_negatives = []
            row_negatives.extend(self._wrong_table(row, gold_ir, table_pool))
            row_negatives.extend(self._unnecessary_join(row, gold_ir, table_pool))
            row_negatives.extend(self._wrong_intent(row, gold_ir))
            for index, negative in enumerate(row_negatives[:max_negatives_per_example], start=1):
                negative["negative_id"] = f"{row.get('example_id')}_neg_{index}"
                negatives.append(negative)
        return negatives

    @staticmethod
    def _wrong_table(row: dict[str, Any], gold_ir: dict[str, Any], table_pool: list[str]) -> list[dict[str, Any]]:
        base = gold_ir.get("base_table")
        replacement = next((table for table in table_pool if table != base), None)
        if not replacement:
            return []
        ir = deepcopy(gold_ir)
        ir["base_table"] = replacement
        ir["required_tables"] = [replacement]
        return [HardNegativeCorpusBuilder._negative(row, ir, "wrong_table")]

    @staticmethod
    def _unnecessary_join(row: dict[str, Any], gold_ir: dict[str, Any], table_pool: list[str]) -> list[dict[str, Any]]:
        if gold_ir.get("joins"):
            return []
        base = gold_ir.get("base_table")
        other = next((table for table in table_pool if table != base), None)
        if not base or not other:
            return []
        ir = deepcopy(gold_ir)
        ir["required_tables"] = [base, other]
        ir["joins"] = [
            {
                "left_table": base,
                "left_column": "id",
                "right_table": other,
                "right_column": f"{base.rstrip('s')}_id",
                "join_type": "INNER",
                "condition": f"{other}.{base.rstrip('s')}_id = {base}.id",
                "path_order": 1,
                "confidence": 0.1,
            }
        ]
        return [HardNegativeCorpusBuilder._negative(row, ir, "unnecessary_join")]

    @staticmethod
    def _wrong_intent(row: dict[str, Any], gold_ir: dict[str, Any]) -> list[dict[str, Any]]:
        ir = deepcopy(gold_ir)
        if ir.get("intent") == "show_records":
            ir["intent"] = "top_n_metric_by_dimension"
            ir["template_id"] = "top_n_metric_by_dimension"
        else:
            ir["intent"] = "show_records"
            ir["template_id"] = "show_records"
        return [HardNegativeCorpusBuilder._negative(row, ir, "wrong_intent")]

    @staticmethod
    def _negative(row: dict[str, Any], negative_ir: dict[str, Any], negative_type: str) -> dict[str, Any]:
        return {
            "example_id": row.get("example_id"),
            "question": row.get("question"),
            "dataset_name": row.get("dataset_name"),
            "db_id": row.get("db_id"),
            "gold_query_ir": row.get("query_ir"),
            "negative_query_ir": negative_ir,
            "negative_type": negative_type,
        }
