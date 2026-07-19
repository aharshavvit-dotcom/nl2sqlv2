"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from datasets.sql_feature_extractor import SQLFeatureExtractor
from datasets.sql_pattern_classifier import SQLPatternClassifier


def classify(sql: str) -> dict:
    features = SQLFeatureExtractor().extract(sql)
    return SQLPatternClassifier().classify(sql, features)


def test_classifier_count_records() -> None:
    assert classify("SELECT COUNT(*) FROM orders")["template_id"] == "count_records"


def test_classifier_count_by_dimension() -> None:
    assert classify("SELECT city, COUNT(*) FROM customers GROUP BY city")["template_id"] == "count_by_dimension"


def test_classifier_metric_by_dimension() -> None:
    assert classify("SELECT city, SUM(amount) FROM orders GROUP BY city")["template_id"] == "metric_by_dimension"


def test_classifier_top_n_metric_by_dimension() -> None:
    result = classify("SELECT city, SUM(amount) FROM orders GROUP BY city ORDER BY SUM(amount) DESC LIMIT 5")
    assert result["template_id"] == "top_n_metric_by_dimension"


def test_classifier_nested_query_unsupported() -> None:
    result = classify("SELECT * FROM orders WHERE amount > (SELECT AVG(amount) FROM orders)")
    assert result["is_supported"] is False
    assert result["unsupported_reason"] == "nested_query"
