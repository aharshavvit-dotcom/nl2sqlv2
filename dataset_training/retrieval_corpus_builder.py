from __future__ import annotations

from typing import Any


class RetrievalCorpusBuilder:
    def build(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for row in examples:
            rows.append(
                {
                    "example_id": row.get("example_id"),
                    "question": row.get("question"),
                    "training_text": " ".join(
                        str(item or "")
                        for item in [
                            row.get("question"),
                            row.get("serialized_schema"),
                            row.get("intent"),
                            row.get("template_id"),
                        ]
                    ),
                    "intent": row.get("intent"),
                    "template_id": row.get("template_id"),
                    "query_ir": row.get("query_ir"),
                    "dataset_name": row.get("dataset_name"),
                    "db_id": row.get("db_id"),
                }
            )
        return rows
