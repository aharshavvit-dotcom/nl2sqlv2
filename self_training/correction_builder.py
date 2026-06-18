from __future__ import annotations

from typing import Any


class CorrectionBuilder:
    def build(self, prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
        positives = []
        repairs = []
        for row in prediction_rows:
            gold_ir = row.get("gold_query_ir") or row.get("query_ir")
            if not gold_ir:
                continue
            correction = {
                "example_id": f"{row.get('example_id')}_gold_correction",
                "original_example_id": row.get("example_id"),
                "question": row.get("question"),
                "dataset_name": row.get("dataset_name"),
                "db_id": row.get("db_id"),
                "query_ir": gold_ir,
                "source_sql": row.get("gold_sql") or row.get("source_sql"),
                "source": "gold_correction",
            }
            positives.append(correction)
            if row.get("predicted_query_ir") and row.get("predicted_query_ir") != gold_ir:
                repairs.append({**correction, "wrong_query_ir": row.get("predicted_query_ir"), "repair_type": "gold_repair"})
        return {
            "correction_positive_examples": positives,
            "queryir_repair_examples": repairs,
            "summary": {"positive_examples": len(positives), "repair_examples": len(repairs)},
        }
