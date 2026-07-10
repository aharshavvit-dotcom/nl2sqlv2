from __future__ import annotations

from quality_gates.thresholds import load_thresholds


def test_load_thresholds_preserves_nested_sections(tmp_path) -> None:
    threshold_file = tmp_path / "thresholds.yaml"
    threshold_file.write_text(
        """
minimums:
  sql_validation_rate: 0.91
controlled_predicted_sql:
  min_row_count_match_rate:
    production_min: 0.88
semantic:
  minimum_applicable_cases: 25
calibration:
  max_expected_calibration_error:
    production_max: 0.07
linking:
  min_filter_column_accuracy_rate:
    production_min: 0.72
""",
        encoding="utf-8",
    )

    thresholds = load_thresholds(threshold_file)

    assert thresholds["minimums"]["sql_validation_rate"] == 0.91
    assert thresholds["minimums"]["unsafe_sql_count_max"] == 0
    assert thresholds["controlled_predicted_sql"]["min_row_count_match_rate"]["production_min"] == 0.88
    assert thresholds["semantic"]["minimum_applicable_cases"] == 25
    assert thresholds["calibration"]["max_expected_calibration_error"]["production_max"] == 0.07
    assert thresholds["linking"]["min_filter_column_accuracy_rate"]["production_min"] == 0.72
