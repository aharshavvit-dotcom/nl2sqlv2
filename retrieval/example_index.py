from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ExampleIndex:
    def __init__(self):
        self.examples: list[dict[str, Any]] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    def build(self, examples: list[dict[str, Any]]) -> None:
        self.examples = list(examples)
        texts = [self._text(row) for row in self.examples]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    def search(self, question: str, top_k: int = 20) -> list[dict[str, Any]]:
        if not self.examples or self.vectorizer is None or self.matrix is None:
            return []
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix)[0]
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            {
                **self.examples[index],
                "question_similarity": float(score),
                "score": float(score),
                "rank": rank,
                "source": "example_index",
            }
            for rank, (index, score) in enumerate(ranked, start=1)
        ]

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"examples": self.examples, "vectorizer": self.vectorizer, "matrix": self.matrix}, target)

    @classmethod
    def load(cls, path: str):
        payload = joblib.load(path)
        index = cls()
        index.examples = payload["examples"]
        index.vectorizer = payload["vectorizer"]
        index.matrix = payload["matrix"]
        return index

    @staticmethod
    def _text(row: dict[str, Any]) -> str:
        query_ir = row.get("query_ir") or {}
        return " ".join(
            str(item or "")
            for item in [
                row.get("question"),
                row.get("intent"),
                row.get("template_id"),
                row.get("serialized_schema"),
                query_ir.get("intent"),
                query_ir.get("template_id"),
                " ".join(query_ir.get("required_tables") or []),
            ]
        )
