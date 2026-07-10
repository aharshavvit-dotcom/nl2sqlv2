"""Tests for self_training.gold_comparator."""

from __future__ import annotations

import pytest

from self_training.gold_comparator import (
    BatchComparisonReport,
    ComparisonResult,
    GoldComparator,
    SQLComparisonResult,
)


@pytest.fixture
def comparator():
    return GoldComparator()


# ---------------------------------------------------------------------------
# QueryIR comparison
# ---------------------------------------------------------------------------

class TestCompareQueryIR:
    def test_exact_match(self, comparator):
        gold = {
            "intent": "metric_summary",
            "base_table": "orders",
            "metrics": [{"aggregation": "SUM", "expression": "amount"}],
            "dimensions": [],
            "filters": [],
            "date_filters": [],
            "joins": [],
            "order_by": [],
            "limit": 100,
        }
        result = comparator.compare_query_ir(gold, gold, example_id="ex1")
        assert result.is_exact_match
        assert not result.is_partial_match
        assert result.match_score == 1.0
        assert all(result.field_matches.values())

    def test_wrong_intent(self, comparator):
        pred = {"intent": "show_records", "base_table": "orders"}
        gold = {"intent": "metric_summary", "base_table": "orders"}
        result = comparator.compare_query_ir(pred, gold, example_id="ex2")
        assert not result.is_exact_match
        assert not result.field_matches["intent"]
        assert result.field_matches["base_table"]

    def test_wrong_base_table(self, comparator):
        pred = {"intent": "show_records", "base_table": "products"}
        gold = {"intent": "show_records", "base_table": "orders"}
        result = comparator.compare_query_ir(pred, gold, example_id="ex3")
        assert not result.field_matches["base_table"]
        assert result.field_matches["intent"]

    def test_wrong_metrics(self, comparator):
        pred = {"metrics": [{"aggregation": "COUNT", "expression": "id"}]}
        gold = {"metrics": [{"aggregation": "SUM", "expression": "amount"}]}
        result = comparator.compare_query_ir(pred, gold)
        assert not result.field_matches["metrics"]

    def test_empty_vs_empty(self, comparator):
        result = comparator.compare_query_ir({}, {})
        assert result.is_exact_match
        assert result.match_score == 1.0

    def test_list_filter_values_are_compared_safely(self, comparator):
        pred = {"filters": [{"expression": "status", "operator": "in", "value": ["active", "pending"]}]}
        gold = {"filters": [{"expression": "status", "operator": "in", "value": ["active", "pending"]}]}
        result = comparator.compare_query_ir(pred, gold)
        assert result.field_matches["filters"]

    def test_partial_match_threshold(self, comparator):
        """A partial match needs match_score >= 0.5 and not exact."""
        pred = {
            "intent": "metric_summary",
            "base_table": "orders",
            "metrics": [{"aggregation": "SUM", "expression": "wrong"}],
        }
        gold = {
            "intent": "metric_summary",
            "base_table": "orders",
            "metrics": [{"aggregation": "SUM", "expression": "amount"}],
        }
        result = comparator.compare_query_ir(pred, gold)
        assert not result.is_exact_match
        assert result.match_score >= 0.5
        assert result.is_partial_match

    def test_field_details_populated(self, comparator):
        pred = {"intent": "show_records"}
        gold = {"intent": "count_records"}
        result = comparator.compare_query_ir(pred, gold)
        assert "intent" in result.field_details
        assert result.field_details["intent"]["predicted"] == "show_records"
        assert result.field_details["intent"]["gold"] == "count_records"


# ---------------------------------------------------------------------------
# SQL comparison
# ---------------------------------------------------------------------------

class TestCompareSQL:
    def test_exact_match(self, comparator):
        sql = "SELECT COUNT(*) FROM orders"
        result = comparator.compare_sql(sql, sql)
        assert result.normalized_match
        assert result.structural_match

    def test_whitespace_differences(self, comparator):
        a = "SELECT  COUNT(*)   FROM   orders  "
        b = "SELECT COUNT(*) FROM orders"
        result = comparator.compare_sql(a, b)
        assert result.normalized_match

    def test_case_differences(self, comparator):
        a = "select count(*) from orders"
        b = "SELECT COUNT(*) FROM ORDERS"
        result = comparator.compare_sql(a, b)
        assert result.normalized_match

    def test_different_queries(self, comparator):
        a = "SELECT * FROM orders"
        b = "SELECT COUNT(*) FROM products"
        result = comparator.compare_sql(a, b)
        assert not result.normalized_match
        assert result.keyword_overlap > 0  # share SELECT, FROM

    def test_none_handling(self, comparator):
        result = comparator.compare_sql(None, "SELECT 1")
        assert not result.normalized_match
        assert not result.structural_match

    def test_keyword_overlap(self, comparator):
        a = "SELECT * FROM orders WHERE id = 1"
        b = "SELECT name FROM orders WHERE id = 2"
        result = comparator.compare_sql(a, b)
        assert result.keyword_overlap > 0.5


# ---------------------------------------------------------------------------
# Batch comparison
# ---------------------------------------------------------------------------

class TestCompareBatch:
    def test_batch_comparison(self, comparator):
        predictions = [
            {"example_id": "1", "predicted_query_ir": {"intent": "show_records", "base_table": "orders"}},
            {"example_id": "2", "predicted_query_ir": {"intent": "metric_summary", "base_table": "products"}},
        ]
        gold = [
            {"example_id": "1", "query_ir": {"intent": "show_records", "base_table": "orders"}},
            {"example_id": "2", "query_ir": {"intent": "metric_summary", "base_table": "orders"}},
        ]
        report = comparator.compare_batch(predictions, gold)
        assert isinstance(report, BatchComparisonReport)
        assert report.total == 2
        assert report.exact_matches >= 1
        assert len(report.per_example) == 2

    def test_empty_batch(self, comparator):
        report = comparator.compare_batch([], [])
        assert report.total == 0
        assert report.exact_matches == 0

    def test_missing_gold(self, comparator):
        predictions = [{"example_id": "missing", "predicted_query_ir": {}}]
        gold = [{"example_id": "other", "query_ir": {}}]
        report = comparator.compare_batch(predictions, gold)
        assert report.failures == 1

    def test_field_accuracy_populated(self, comparator):
        predictions = [
            {"example_id": "1", "predicted_query_ir": {"intent": "show_records"}},
        ]
        gold = [
            {"example_id": "1", "query_ir": {"intent": "show_records"}},
        ]
        report = comparator.compare_batch(predictions, gold)
        assert "intent" in report.field_accuracy
        assert report.field_accuracy["intent"] == 1.0
