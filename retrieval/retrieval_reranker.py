from __future__ import annotations

from typing import Any

from dataset_training.utils import query_ir_tables, schema_tokens
from reward.reward_scorer import RewardScorer
from .pattern_index import infer_pattern


class RetrievalReranker:
    def __init__(self, reward_scorer: RewardScorer | None = None):
        self.reward_scorer = reward_scorer or RewardScorer()

    def rerank(
        self,
        question: str,
        schema: dict[str, Any],
        candidates: list[dict[str, Any]],
        pattern_matches: list[dict[str, Any]] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        current_pattern = infer_pattern(question)
        current_schema_tokens = schema_tokens(schema)
        pattern_scores = {item["pattern"]: float(item.get("score", 0.0)) for item in pattern_matches or []}
        reranked = []
        for candidate in candidates:
            query_ir = candidate.get("query_ir") or {}
            candidate_pattern = candidate.get("intent") or candidate.get("template_id") or query_ir.get("intent")
            question_similarity = float(candidate.get("question_similarity", candidate.get("score", 0.0)) or 0.0)
            intent_pattern_score = pattern_scores.get(candidate_pattern, 1.0 if candidate_pattern == current_pattern else 0.1)
            candidate_schema_tokens = set()
            for table in query_ir_tables(query_ir):
                candidate_schema_tokens.update(table.lower().replace("_", " ").split())
            schema_union = current_schema_tokens | candidate_schema_tokens
            schema_overlap_score = len(current_schema_tokens & candidate_schema_tokens) / len(schema_union) if schema_union else 0.0
            structure_score = self._structure_score(current_pattern, query_ir)
            dataset_quality_score = float(candidate.get("dataset_quality_score", 1.0))
            reward = self.reward_scorer.score(candidate, question, schema)
            reward_score = float(reward.get("reward_score", 0.0))
            final_score = (
                0.30 * question_similarity
                + 0.22 * intent_pattern_score
                + 0.22 * schema_overlap_score
                + 0.08 * structure_score
                + 0.04 * dataset_quality_score
                + 0.14 * reward_score
            )
            final_score *= self._penalty(question, current_pattern, candidate_pattern, query_ir, schema_overlap_score)
            reranked.append(
                {
                    **candidate,
                    "final_score": final_score,
                    "schema_overlap_score": schema_overlap_score,
                    "intent_pattern_score": intent_pattern_score,
                    "reward_score": reward_score,
                    "reward": reward,
                }
            )
        return sorted(reranked, key=lambda item: item["final_score"], reverse=True)[:top_k]

    @staticmethod
    def _structure_score(current_pattern: str, query_ir: dict[str, Any]) -> float:
        if current_pattern == "show_records":
            return 1.0 if not query_ir.get("metrics") and not query_ir.get("joins") else 0.1
        if current_pattern == "count_records":
            return 1.0 if query_ir.get("metrics") else 0.3
        if current_pattern in {"metric_by_dimension", "top_n_metric_by_dimension"}:
            return 1.0 if query_ir.get("metrics") and query_ir.get("dimensions") else 0.3
        return 0.7

    @staticmethod
    def _penalty(
        question: str,
        current_pattern: str,
        candidate_pattern: str | None,
        query_ir: dict[str, Any],
        schema_overlap_score: float,
    ) -> float:
        penalty = 1.0
        if current_pattern == "show_records" and candidate_pattern in {"metric_summary", "metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension"}:
            penalty *= 0.15
        if query_ir.get("joins") and not any(word in question.lower() for word in ["join", "with", "by", "along", "including"]):
            penalty *= 0.25
        if schema_overlap_score == 0.0:
            penalty *= 0.45
        return penalty
