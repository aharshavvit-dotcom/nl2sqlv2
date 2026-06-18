from __future__ import annotations

from typing import Any

from reward.reward_scorer import RewardScorer


class DatasetRewardScorer:
    def __init__(self):
        self.reward_scorer = RewardScorer()

    def features(self, candidate: dict[str, Any], gold: dict[str, Any], schema: dict[str, Any] | None = None) -> dict[str, float]:
        query_ir = candidate.get("query_ir") or candidate.get("predicted_query_ir") or {}
        gold_ir = gold.get("query_ir") or gold.get("gold_query_ir") or {}
        reward = self.reward_scorer.score({"query_ir": query_ir, "sql": candidate.get("predicted_sql") or candidate.get("sql")}, candidate.get("question") or gold.get("question") or "", schema or {})
        return {
            "reward_score": float(reward["reward_score"]),
            "intent_match": 1.0 if query_ir.get("intent") == gold_ir.get("intent") else 0.0,
            "base_table_match": 1.0 if query_ir.get("base_table") == gold_ir.get("base_table") else 0.0,
            "join_count_delta": abs(len(query_ir.get("joins") or []) - len(gold_ir.get("joins") or [])),
            "metric_count_delta": abs(len(query_ir.get("metrics") or []) - len(gold_ir.get("metrics") or [])),
            "filter_count_delta": abs(len(query_ir.get("filters") or []) - len(gold_ir.get("filters") or [])),
        }
