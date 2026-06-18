from __future__ import annotations

from typing import Any

from .reward_scorer import RewardScorer


class CandidateReranker:
    def __init__(self, scorer: RewardScorer | None = None):
        self.scorer = scorer or RewardScorer()

    def rerank(
        self,
        candidates: list[dict[str, Any]],
        question: str,
        schema: dict[str, Any],
        context: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        scored = []
        for candidate in candidates:
            reward = self.scorer.score(candidate, question, schema, context=context)
            base = float(candidate.get("final_score", candidate.get("score", 0.0)) or 0.0)
            combined = (0.70 * base) + (0.30 * float(reward["reward_score"]))
            scored.append({**candidate, "reward": reward, "reward_score": reward["reward_score"], "combined_score": combined})
        ranked = sorted(scored, key=lambda item: item["combined_score"], reverse=True)
        return ranked[:top_k] if top_k is not None else ranked
