from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from retrieval.tfidf_retriever import RetrievalResult

from .example_index import ExampleIndex
from .feedback_index import FeedbackIndex
from .pattern_index import PatternIndex
from .retrieval_reranker import RetrievalReranker
from .schema_index import SchemaIndex
from .artifact_compatibility import validate_sklearn_metadata


class LocalRAGRetriever:
    def __init__(
        self,
        example_index: ExampleIndex,
        schema_index: SchemaIndex,
        pattern_index: PatternIndex,
        reranker: RetrievalReranker,
        feedback_index: FeedbackIndex | None = None,
    ):
        self.example_index = example_index
        self.schema_index = schema_index
        self.pattern_index = pattern_index
        self.reranker = reranker
        self.feedback_index = feedback_index

    def retrieve(self, question: str, schema: dict, top_k: int = 10) -> dict[str, Any]:
        schema = _schema_with_semantic_profile(schema)
        examples = self.example_index.search(question, top_k=max(top_k * 2, 20))
        schema_matches = self.schema_index.search_schema_matches(question, schema, top_k=max(top_k * 2, 20))
        patterns = self.pattern_index.search_patterns(question, top_k=10)
        feedback_matches = self.feedback_index.search(question, schema, top_k=max(top_k * 2, 20)) if self.feedback_index else []
        merged = self._merge(examples, schema_matches, feedback_matches)
        reranked = self.reranker.rerank(question, schema, merged, pattern_matches=patterns, top_k=top_k)
        return {
            "examples": examples[:top_k],
            "schema_matches": schema_matches[:top_k],
            "feedback_matches": feedback_matches[:top_k],
            "patterns": patterns,
            "reranked": reranked,
            "debug": {
                "example_scores": [{"example_id": row.get("example_id"), "score": row.get("score")} for row in examples[:top_k]],
                "schema_scores": [{"example_id": row.get("example_id"), "score": row.get("score")} for row in schema_matches[:top_k]],
                "feedback_scores": [{"example_id": row.get("example_id"), "score": row.get("score")} for row in feedback_matches[:top_k]],
                "pattern_scores": patterns,
            },
        }

    @staticmethod
    def _merge(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for group in groups:
            for row in group:
                key = str(row.get("example_id") or id(row))
                current = merged.get(key, {})
                combined = {**current, **row}
                combined["question_similarity"] = max(float(current.get("question_similarity", 0.0) or 0.0), float(row.get("question_similarity", 0.0) or 0.0))
                combined["schema_overlap_score"] = max(float(current.get("schema_overlap_score", 0.0) or 0.0), float(row.get("schema_overlap_score", 0.0) or 0.0))
                merged[key] = combined
        return list(merged.values())


class RAGRetrieverAdapter:
    """Compatibility wrapper exposing the legacy retriever query API."""

    def __init__(self, retriever: LocalRAGRetriever):
        self.retriever = retriever

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "RAGRetrieverAdapter":
        path = Path(artifact_dir)
        validate_sklearn_metadata(path, mode="runtime")
        example_index = ExampleIndex.load(str(path / "example_index.pkl"))
        schema_index = joblib.load(path / "schema_index.pkl")
        pattern_index = joblib.load(path / "pattern_index.pkl")
        feedback_path = path / "feedback_index.pkl"
        feedback_index = FeedbackIndex.load(feedback_path) if feedback_path.exists() else None
        return cls(LocalRAGRetriever(example_index, schema_index, pattern_index, RetrievalReranker(), feedback_index=feedback_index))

    def query(self, text: str, top_k: int = 3) -> list[RetrievalResult]:
        return self.query_with_schema(text, schema={}, top_k=top_k)

    def query_with_schema(self, text: str, schema: dict[str, Any] | None, top_k: int = 3) -> list[RetrievalResult]:
        payload = self.retriever.retrieve(text, schema or {}, top_k=top_k)
        rows = payload.get("reranked") or payload.get("examples") or []
        return [self._to_result(row, rank) for rank, row in enumerate(rows[:top_k], start=1)]

    @staticmethod
    def _to_result(row: dict[str, Any], rank: int) -> RetrievalResult:
        query_ir = row.get("query_ir") or {}
        example_id = str(row.get("example_id") or row.get("id") or f"rag_{rank}")
        template_id = str(row.get("template_id") or query_ir.get("template_id") or row.get("intent") or "unknown")
        return RetrievalResult(
            example_id=example_id,
            question=str(row.get("question") or ""),
            score=float(row.get("final_score", row.get("score", row.get("question_similarity", 0.0))) or 0.0),
            template_id=template_id,
            example={**row, "example_id": example_id, "template_id": template_id},
        )


def _schema_with_semantic_profile(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict) or schema.get("semantic_profile"):
        return schema
    try:
        from semantic_layer import build_semantic_profile

        return {**schema, "semantic_profile": build_semantic_profile(schema)}
    except Exception:
        return schema
