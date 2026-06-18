from __future__ import annotations

from typing import Any


class NeuralCorpusBuilder:
    def build(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            row
            for row in examples
            if row.get("query_ir") and row.get("question") and row.get("serialized_schema") is not None
        ]
