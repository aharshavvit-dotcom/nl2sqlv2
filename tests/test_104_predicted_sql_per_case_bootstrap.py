"""
Purpose: Protects data unit behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_selection.promotion_policy import _compare_predicted_sql_per_case
from validation.sql_validator import SQLValidator


ROOT = Path(__file__).resolve().parents[1]


def test_controlled_predicted_sql_fixture_set_supports_bootstrap():
    path = ROOT / "evaluation" / "fixtures" / "controlled_evaluation_cases.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(cases) >= 10
    case_ids = [case.get("case_id") for case in cases]
    assert all(case_ids)
    assert len(case_ids) == len(set(case_ids))

    validator = SQLValidator()
    invalid = {
        case["case_id"]: validator.validate(case["gold_sql"])["issues"]
        for case in cases
        if not validator.validate(case["gold_sql"])["is_valid"]
    }
    assert invalid == {}


def test_predicted_sql_per_case_bootstrap():
    challenger = [
        {"case_id": f"c_{i}", "final_execution_match": True, "question": f"q_{i}"}
        for i in range(20)
    ]
    champion = [
        {"case_id": f"c_{i}", "final_execution_match": (i % 2 == 0), "question": f"q_{i}"}
        for i in range(20)
    ]
    
    result = _compare_predicted_sql_per_case(challenger, champion)
    
    assert result["available"] is True
    assert result["common_cases"] == 20
    assert result["improvement_count"] == 10
    assert result["regression_count"] == 0
    assert result["execution_match_delta"] == 0.5
    assert result["statistical_check_available"] is True
    assert result["reason"] == ""
    
    # Verify percentiles are computed and valid
    assert 0.0 <= result["delta_p05"] <= 1.0
    assert 0.0 <= result["delta_p50"] <= 1.0
    assert 0.0 <= result["delta_p95"] <= 1.0
    assert result["regression_detected"] is False


def test_insufficient_cases_has_reason():
    challenger = [
        {"case_id": f"c_{i}", "final_execution_match": True, "question": f"q_{i}"}
        for i in range(5)
    ]
    champion = [
        {"case_id": f"c_{i}", "final_execution_match": (i % 2 == 0), "question": f"q_{i}"}
        for i in range(5)
    ]
    
    result = _compare_predicted_sql_per_case(challenger, champion)
    
    assert result["available"] is True
    assert result["common_cases"] == 5
    assert result["statistical_check_available"] is False
    assert result["reason"] == "insufficient_common_cases"
    assert result["minimum_cases_required"] == 10
    
    # Delta falls back to point estimate
    assert result["delta_p05"] == result["execution_match_delta"]
    assert result["delta_p95"] == result["execution_match_delta"]


def test_missing_cases_do_not_crash():
    challenger = [
        {"case_id": "c_1", "final_execution_match": True, "question": "q1"},
        {"case_id": "c_2", "final_execution_match": True, "question": "q2"},
    ]
    # champion has extra/different cases
    champion = [
        {"case_id": "c_2", "final_execution_match": False, "question": "q2"},
        {"case_id": "c_3", "final_execution_match": True, "question": "q3"},
    ]
    
    result = _compare_predicted_sql_per_case(challenger, champion)
    
    assert result["available"] is True
    assert result["common_cases"] == 1  # Only "c_2" is common
    assert result["statistical_check_available"] is False
    assert result["reason"] == "insufficient_common_cases"


def test_empty_cases():
    # Empty lists
    result1 = _compare_predicted_sql_per_case([], [])
    assert result1["available"] is False
    assert result1["reason"] == "no_challenger_cases"
    
    result2 = _compare_predicted_sql_per_case([{"case_id": "c1"}], None)
    assert result2["available"] is False
    assert result2["reason"] == "no_champion_cases"
