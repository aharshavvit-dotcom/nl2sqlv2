import pytest
from nl2sqlv2.model_selection.promotion_policy import _compare_predicted_sql_per_case

def test_predicted_sql_per_case_bootstrap():
    challenger = [
        {"case_id": f"c_{i}", "final_execution_match": True} for i in range(20)
    ]
    champion = [
        {"case_id": f"c_{i}", "final_execution_match": (i % 2 == 0)} for i in range(20)
    ]
    
    result = _compare_predicted_sql_per_case(challenger, champion)
    
    assert result["available"] is True
    assert result["common_cases"] == 20
    assert result["improvement_count"] == 10
    assert result["regression_count"] == 0
    assert result["execution_match_delta"] == 0.5
    assert result["statistical_check_available"] is True
    
    # Check that p05, p50, p95 are populated and reasonable
    assert 0.0 <= result["delta_p05"] <= 1.0
    assert 0.0 <= result["delta_p50"] <= 1.0
    assert 0.0 <= result["delta_p95"] <= 1.0
    assert result["regression_detected"] is False

def test_insufficient_cases_fallback():
    challenger = [
        {"case_id": f"c_{i}", "final_execution_match": True} for i in range(5)
    ]
    champion = [
        {"case_id": f"c_{i}", "final_execution_match": (i % 2 == 0)} for i in range(5)
    ]
    
    result = _compare_predicted_sql_per_case(challenger, champion)
    
    assert result["available"] is True
    assert result["common_cases"] == 5
    assert result["statistical_check_available"] is False
    assert result["delta_p05"] == result["execution_match_delta"]
    assert result["delta_p95"] == result["execution_match_delta"]
