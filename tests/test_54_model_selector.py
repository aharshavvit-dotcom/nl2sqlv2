from __future__ import annotations

from model_selection.model_candidate import ModelCandidate
from model_selection.model_selector import ModelSelector


THRESHOLDS = {"minimums": {"sql_validation_rate": 0.9, "no_select_star_rate": 1.0, "unsafe_sql_count_max": 0, "unnecessary_join_rate_max": 0.05, "wrong_table_rate_max": 0.15}}


def _candidate(name: str, **metrics) -> ModelCandidate:
    defaults = {"sql_validation_rate": 1.0, "no_select_star_rate": 1.0, "unsafe_sql_count": 0, "unnecessary_join_rate": 0.0, "wrong_table_rate": 0.0, "gold_comparison_score": 0.8}
    defaults.update(metrics)
    return ModelCandidate(name, f"artifacts/{name}", "neural_ir", defaults, "now", {})


def test_best_model_selected_and_unsafe_rejected() -> None:
    report = ModelSelector().select_best([_candidate("safe", gold_comparison_score=0.9), _candidate("unsafe", unsafe_sql_count=1)], THRESHOLDS)

    assert report["selected_model"]["name"] == "safe"
    assert report["rejected_models"]


def test_model_with_high_unnecessary_joins_rejected() -> None:
    report = ModelSelector().select_best([_candidate("joiny", unnecessary_join_rate=0.5)], THRESHOLDS)

    assert report["selected_model"] is None
    assert "unnecessary_join_rate" in report["rejected_models"][0]["blocking_issues"]
