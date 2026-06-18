from __future__ import annotations

from typing import Any

from dataset_training.utils import query_ir_tables, schema_tokens


class SchemaIndex:
    def __init__(self):
        self.examples: list[dict[str, Any]] = []
        self.example_tokens: list[set[str]] = []

    def build(self, examples: list[dict[str, Any]]) -> None:
        self.examples = list(examples)
        self.example_tokens = [self._tokens(row) for row in self.examples]

    def search_schema_matches(self, question: str, schema: dict, top_k: int = 20) -> list[dict[str, Any]]:
        current_tokens = schema_tokens(schema) | set(question.lower().replace("_", " ").split())
        scored = []
        for row, tokens in zip(self.examples, self.example_tokens):
            union = current_tokens | tokens
            score = len(current_tokens & tokens) / len(union) if union else 0.0
            scored.append({**row, "schema_overlap_score": score, "score": score, "source": "schema_index"})
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]

    @staticmethod
    def _tokens(row: dict[str, Any]) -> set[str]:
        schema = row.get("schema") or {}
        tokens = schema_tokens(schema)
        for table in query_ir_tables(row.get("query_ir")):
            tokens.update(table.lower().replace("_", " ").split())
        return tokens
