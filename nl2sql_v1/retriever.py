from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RetrievalResult:
    example_id: str
    question: str
    score: float
    template_id: str
    example: dict[str, Any]


def load_examples(path: str | Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


class TfidfRetriever:
    def __init__(self, examples: list[dict[str, Any]], vectorizer: TfidfVectorizer, matrix: Any):
        self.examples = examples
        self.vectorizer = vectorizer
        self.matrix = matrix

    @classmethod
    def train(cls, examples_path: str | Path) -> "TfidfRetriever":
        examples = load_examples(examples_path)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)
        questions = [row.get("training_text") or row["question"] for row in examples]
        matrix = vectorizer.fit_transform(questions)
        return cls(examples=examples, vectorizer=vectorizer, matrix=matrix)

    @classmethod
    def load(cls, model_path: str | Path) -> "TfidfRetriever":
        path = Path(model_path)
        if path.is_dir():
            examples = load_examples(path / "training_examples.jsonl")
            return cls(
                examples=examples,
                vectorizer=joblib.load(path / "tfidf_vectorizer.pkl"),
                matrix=joblib.load(path / "tfidf_matrix.pkl"),
            )
        payload = joblib.load(path)
        return cls(
            examples=payload["examples"],
            vectorizer=payload["vectorizer"],
            matrix=payload["matrix"],
        )

    @classmethod
    def load_or_train(cls, model_path: str | Path, examples_path: str | Path) -> "TfidfRetriever":
        path = Path(model_path)
        if path.is_dir() and (path / "training_examples.jsonl").exists():
            return cls.load(path)
        if path.exists() and path.is_file():
            return cls.load(path)
        retriever = cls.train(examples_path)
        retriever.save(path)
        return retriever

    def save(self, model_path: str | Path) -> None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "examples": self.examples,
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
            },
            path,
        )

    def query(self, text: str, top_k: int = 3) -> list[RetrievalResult]:
        query_vec = self.vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        return [
            RetrievalResult(
                example_id=self.examples[i].get("id") or self.examples[i]["example_id"],
                question=self.examples[i]["question"],
                score=float(scores[i]),
                template_id=self.examples[i]["template_id"],
                example=self.examples[i],
            )
            for i in ranked
        ]
