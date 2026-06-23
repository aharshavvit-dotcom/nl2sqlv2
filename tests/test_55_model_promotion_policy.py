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


def test_bootstrap_report_uses_paired_examples() -> None:
    challenger = _metrics(per_example=[{"example_id": str(i), "intent_correct": True} for i in range(10)])
    champion = _metrics(per_example=[{"example_id": str(i), "intent_correct": False} for i in range(10)])

    decision = PromotionPolicy().can_promote(challenger, champion, THRESHOLDS, bootstrap_iterations=100)

    report = decision["statistical_report"]
    assert report["paired_examples"] == 10
    assert report["metrics"]["intent_macro_f1"]["delta_p05"] > 0
    assert decision["can_promote"] is True


def test_bootstrap_coverage_does_not_skip_uncovered_point_regression() -> None:
    challenger = _metrics(
        unseen_db_sql_validation_rate=0.5,
        per_example=[{"example_id": str(i), "intent_correct": True} for i in range(10)],
    )
    champion = _metrics(
        unseen_db_sql_validation_rate=0.9,
        per_example=[{"example_id": str(i), "intent_correct": False} for i in range(10)],
    )

    decision = PromotionPolicy().can_promote(challenger, champion, THRESHOLDS, bootstrap_iterations=100)

    assert decision["statistical_checks"]["intent_macro_f1"]["statistical_check_available"] is True
    assert decision["point_estimate_fallback_checks"]["unseen_db_sql_validation_rate"]["regression_detected"] is True
    assert decision["can_promote"] is False
