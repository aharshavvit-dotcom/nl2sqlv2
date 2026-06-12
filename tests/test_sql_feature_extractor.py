from __future__ import annotations

from datasets.sql_feature_extractor import SQLFeatureExtractor


def test_sql_feature_extractor_extracts_grouped_metric() -> None:
    features = SQLFeatureExtractor().extract(
        "SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id ORDER BY SUM(amount) DESC LIMIT 5"
    )

    assert features["statement_type"] == "SELECT"
    assert "SUM" in features["aggregations"]
    assert features["group_by"] == ["customer_id"]
    assert features["order_by"][0]["desc"] is True
    assert features["limit"] == 5


def test_sql_feature_extractor_detects_nested_query() -> None:
    features = SQLFeatureExtractor().extract("SELECT * FROM orders WHERE amount > (SELECT AVG(amount) FROM orders)")
    assert features["has_nested_query"] is True
