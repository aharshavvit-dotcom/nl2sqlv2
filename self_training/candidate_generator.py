from __future__ import annotations

from copy import deepcopy
from typing import Any


class GoldCandidateGenerator:
    def generate(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        gold = row.get("gold_query_ir") or row.get("query_ir") or {}
        predicted = row.get("predicted_query_ir") or gold
        candidates = [
            {"candidate_id": f"{row.get('example_id')}_pred", "question": row.get("question"), "query_ir": predicted, "label": 1 if predicted == gold else 0},
            {"candidate_id": f"{row.get('example_id')}_gold", "question": row.get("question"), "query_ir": gold, "label": 1},
        ]
        if gold:
            bad_join = deepcopy(gold)
            if not bad_join.get("joins"):
                bad_join["joins"] = [{"condition": "unnecessary.join = users.id", "left_table": "assignments", "right_table": bad_join.get("base_table")}]
                candidates.append({"candidate_id": f"{row.get('example_id')}_unnecessary_join", "question": row.get("question"), "query_ir": bad_join, "label": 0})
        return candidates
