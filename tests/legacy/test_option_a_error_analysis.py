"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from neural_ir.error_analysis import OptionAErrorAnalyzer


def test_option_a_error_analysis_report_shape() -> None:
    report = OptionAErrorAnalyzer().analyze(
        [
            {
                "id": "x1",
                "question": "Top customers by sales",
                "dataset_name": "wikisql",
                "gold": {"intent": "top_n_metric_by_dimension", "metric_column": {"table": "orders", "column": "amount"}},
                "prediction": {"intent": "show_records", "metric_column": None},
                "sql_validation": {"is_valid": False, "issues": ["SQL is empty."]},
            }
        ]
    )

    assert report["by_intent"]
    assert report["by_dataset"]
    assert report["by_failure_type"]
    assert report["slot_accuracy"]
    assert report["recommendations"]

