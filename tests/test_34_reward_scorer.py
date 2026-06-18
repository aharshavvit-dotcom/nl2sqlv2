from __future__ import annotations

from reward.reward_scorer import RewardScorer


SCHEMA = {
    "dialect": "sqlite",
    "tables": {
        "users": {"columns": {"id": {}, "name": {}}},
        "assignments": {"columns": {"id": {}, "user_id": {}}},
    },
}


def test_reward_scorer_penalizes_unnecessary_joins() -> None:
    good = {
        "query_ir": {"intent": "show_records", "base_table": "users", "required_tables": ["users"], "joins": [], "metrics": [], "dimensions": [{"table": "users", "column": "id"}]},
        "sql": "SELECT users.id FROM users LIMIT 100",
    }
    bad = {
        "query_ir": {
            "intent": "show_records",
            "base_table": "users",
            "required_tables": ["users", "assignments"],
            "joins": [{"condition": "assignments.user_id = users.id"}],
            "metrics": [],
        },
        "sql": "SELECT users.id FROM users JOIN assignments ON assignments.user_id = users.id LIMIT 100",
    }

    scorer = RewardScorer()
    good_score = scorer.score(good, "list all users", SCHEMA)
    bad_score = scorer.score(bad, "list all users", SCHEMA)

    assert bad_score["reward_score"] < good_score["reward_score"]
    assert "unnecessary_join_for_direct_query" in bad_score["penalties"]
