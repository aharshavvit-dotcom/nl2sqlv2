from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dataset_training.utils import query_ir_tables, schema_tokens


class FeedbackIndex:
    def __init__(self):
        self.examples: list[dict[str, Any]] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    def build(self, feedback_examples: list[dict[str, Any]]) -> None:
        self.examples = list(feedback_examples)
        texts = [self._text(row) for row in self.examples]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    def search(self, question: str, schema: dict[str, Any], top_k: int = 10) -> list[dict[str, Any]]:
        if not self.examples:
            return []
        base_scores = [0.0 for _ in self.examples]
        if self.vectorizer is not None and self.matrix is not None:
            query = self.vectorizer.transform([question])
            base_scores = list(cosine_similarity(query, self.matrix)[0])

        requested_fingerprint = self._schema_fingerprint(schema)
        requested_tokens = schema_tokens(schema)
        scored = []
        for index, row in enumerate(self.examples):
            query_ir = row.get("query_ir") or {}
            candidate_tables = query_ir_tables(query_ir)
            candidate_tokens: set[str] = set()
            for table in candidate_tables:
                candidate_tokens.update(table.lower().replace("_", " ").split())
            union = requested_tokens | candidate_tokens
            table_overlap = len(requested_tokens & candidate_tokens) / len(union) if union else 0.0
            fingerprint_boost = 0.25 if requested_fingerprint and row.get("schema_fingerprint") == requested_fingerprint else 0.0
            feedback_tag_boost = 0.10 if row.get("feedback_tags") else 0.0
            score = min(1.0, (0.65 * float(base_scores[index])) + (0.25 * table_overlap) + fingerprint_boost + feedback_tag_boost)
            scored.append(
                {
                    **row,
                    "score": score,
                    "question_similarity": float(base_scores[index]),
                    "schema_overlap_score": table_overlap,
                    "source": "feedback_index",
                }
            )
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"examples": self.examples, "vectorizer": self.vectorizer, "matrix": self.matrix}, target)

    @classmethod
    def load(cls, path: str | Path) -> "FeedbackIndex":
        payload = joblib.load(path)
        index = cls()
        index.examples = payload.get("examples", [])
        index.vectorizer = payload.get("vectorizer")
        index.matrix = payload.get("matrix")
        return index

    @staticmethod
    def _text(row: dict[str, Any]) -> str:
        query_ir = row.get("query_ir") or {}
        return " ".join(
            str(value or "")
            for value in [
                row.get("question"),
                row.get("intent"),
                row.get("template_id"),
                row.get("schema_fingerprint"),
                " ".join(row.get("feedback_tags") or []),
                query_ir.get("intent"),
                query_ir.get("template_id"),
                " ".join(query_ir.get("required_tables") or []),
            ]
        )

    @staticmethod
    def _schema_fingerprint(schema: dict[str, Any] | None) -> str | None:
        if not schema:
            return None
        return schema.get("schema_fingerprint") or schema.get("fingerprint") or schema.get("schema_hash")
