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


def test_model_selector_surfaces_predicted_sql_metrics() -> None:
    report = ModelSelector().select_best([
        _candidate(
            "safe",
            controlled_predicted_sql_execution_match_rate=0.7,
            controlled_predicted_sql_safe_sql_rate=1.0,
            controlled_predicted_sql_unsafe_sql_count=0,
        )
    ], THRESHOLDS)

    predicted = report["predicted_sql_execution"]
    assert predicted["available"] is True
    assert predicted["execution_match_rate"] == 0.7
    assert predicted["safe_sql_rate"] == 1.0


def test_gold_baseline_is_not_promotion_eligible() -> None:
    candidate = _candidate("gold")
    candidate.model_artifact_source = "gold_baseline"
    candidate.evaluation_mode = "gold_replay"
    candidate.eligible_for_promotion = False
    report = ModelSelector().select_best([candidate], THRESHOLDS)
    assert report["selected_model"] is None
    issues = report["rejected_models"][0]["blocking_issues"]
    assert "evaluation_mode_not_real_model_predictions" in issues
    assert "candidate_not_eligible_for_promotion" in issues


def test_legacy_cached_report_is_not_promotion_eligible() -> None:
    candidate = _candidate("legacy")
    candidate.model_artifact_source = "legacy_cache"
    report = ModelSelector().select_best([candidate], THRESHOLDS)
    assert report["selected_model"] is None
    assert report["selection_blocked"] is True
    assert report["ineligible_candidates"][0]["reason"] == "stale_report"


def test_real_candidate_with_failed_quality_gate_is_rejected() -> None:
    candidate = _candidate("real")
    candidate.metadata["quality_gate_passed"] = False
    report = ModelSelector().select_best([candidate], THRESHOLDS)
    assert report["selected_model"] is None
    assert "quality_gate_not_passed" in report["rejected_models"][0]["blocking_issues"]


def test_stale_or_bundle_mismatched_candidate_is_rejected() -> None:
    candidate = _candidate("stale")
    candidate.candidate_bundle_id = "bundle-old"
    candidate.manifest_bundle_id = "bundle-current"
    candidate.generated_at = "2026-06-17T00:00:00+00:00"
    candidate.metadata.update({
        "enforce_freshness": True,
        "candidate_bundle_generated_at": "2026-07-04T00:00:00+00:00",
    })

    report = ModelSelector().select_best([candidate], THRESHOLDS)

    assert report["selected_model"] is None
    reasons = report["ineligible_candidates"][0]["reasons"]
    assert "bundle_id_mismatch" in reasons
    assert "stale_report" in reasons
    assert report["selection_blocked_reason"] == "no_eligible_candidate"
