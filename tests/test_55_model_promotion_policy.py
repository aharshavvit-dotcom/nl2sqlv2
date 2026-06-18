from __future__ import annotations

from model_selection.promotion_policy import PromotionPolicy


THRESHOLDS = {"minimums": {"sql_validation_rate": 0.9, "no_select_star_rate": 1.0, "unsafe_sql_count_max": 0, "unnecessary_join_rate_max": 0.05, "wrong_table_rate_max": 0.15, "model_promotion_min_improvement": 0.01}}


def _metrics(**overrides):
    values = {"sql_validation_rate": 1.0, "no_select_star_rate": 1.0, "unsafe_sql_count": 0, "unnecessary_join_rate": 0.0, "wrong_table_rate": 0.0, "simple_query_pass_rate": 1.0, "gold_comparison_score": 0.9, "unseen_db_sql_validation_rate": 0.9}
    values.update(overrides)
    return values


def test_challenger_promoted_when_better() -> None:
    decision = PromotionPolicy().can_promote(_metrics(gold_comparison_score=0.91), _metrics(gold_comparison_score=0.9), THRESHOLDS)
    assert decision["can_promote"] is True


def test_challenger_blocked_when_unsafe_or_simple_regresses() -> None:
    unsafe = PromotionPolicy().can_promote(_metrics(unsafe_sql_count=1), _metrics(), THRESHOLDS)
    simple_regression = PromotionPolicy().can_promote(_metrics(simple_query_pass_rate=0.5), _metrics(simple_query_pass_rate=1.0), THRESHOLDS)

    assert unsafe["can_promote"] is False
    assert simple_regression["can_promote"] is False
