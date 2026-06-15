from __future__ import annotations

from neural_ir.benchmark import HybridBenchmark


def test_hybrid_benchmark_reports_all_model_sections() -> None:
    cases = [
        {
            "id": "count_orders",
            "question": "Count orders",
            "expected_sql_contains": ["COUNT"],
            "expected_sql_not_contains": ["SELECT *"],
            "should_execute": False,
        }
    ]

    def predictor(question: str) -> dict:
        return {"sql": "SELECT COUNT(*) AS record_count FROM orders LIMIT 100", "validation": {"is_valid": True}, "confidence": 0.8}

    report = HybridBenchmark(option_c_predictor=predictor, option_a_predictor=predictor, hybrid_predictor=predictor).run(cases)

    assert "option_c" in report
    assert "option_a" in report
    assert "hybrid" in report
    assert "comparison" in report
    assert report["hybrid"]["case_pass_rate"] == 1.0
