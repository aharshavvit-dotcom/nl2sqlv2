from __future__ import annotations

from typing import Any

from .prediction_models import RetrievedCandidate


class CandidateGenerator:
    def generate_candidates(
        self,
        question: str,
        retriever: Any,
        top_k: int = 10,
        schema: dict[str, Any] | None = None,
    ) -> list[RetrievedCandidate]:
        if schema is not None and hasattr(retriever, "query_with_schema"):
            results = retriever.query_with_schema(question, schema=schema, top_k=top_k)
        else:
            results = retriever.query(question, top_k=top_k)
        candidates: list[RetrievedCandidate] = []
        for rank, result in enumerate(results, start=1):
            example = result.example
            slots = {
                "metric": example.get("metric"),
                "dimension": example.get("dimension"),
                "entity": example.get("entity"),
                "limit": example.get("limit"),
                "order": example.get("order"),
                **(example.get("extracted_slots") or {}),
            }
            candidates.append(
                RetrievedCandidate(
                    rank=rank,
                    example_id=result.example_id,
                    question=result.question,
                    dataset_name=example.get("dataset_name"),
                    db_id=example.get("db_id"),
                    intent=example.get("intent"),
                    template_id=example.get("template_id"),
                    slots={key: value for key, value in slots.items() if value is not None},
                    sql_features=example.get("sql_features") or {},
                    similarity_score=float(result.score),
                )
            )
        return candidates
