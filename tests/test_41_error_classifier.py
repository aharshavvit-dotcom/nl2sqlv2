"""Tests for self_training.error_classifier."""

from __future__ import annotations

import pytest

from self_training.error_classifier import ErrorCategory, ErrorClassification, ErrorClassifier, ErrorReport
from self_training.gold_comparator import ComparisonResult


@pytest.fixture
def classifier():
    return ErrorClassifier()


# ---------------------------------------------------------------------------
# Single classification
# ---------------------------------------------------------------------------

class TestClassify:
    def test_wrong_intent(self, classifier):
        comp = ComparisonResult(
            example_id="ex1",
            field_matches={"intent": False, "base_table": True, "metrics": True,
                           "dimensions": True, "filters": True, "date_filters": True,
                           "joins": True, "order_by": True, "limit": True},
            field_details={"intent": {"predicted": "show_records", "gold": "metric_summary"}},
        )
        ec = classifier.classify(comp, {})
        assert ErrorCategory.WRONG_INTENT in ec.categories
        assert ec.severity == "critical"
        assert ec.suggested_fix_type == "intent_correction"

    def test_wrong_base_table(self, classifier):
        comp = ComparisonResult(
            example_id="ex2",
            field_matches={"intent": True, "base_table": False, "metrics": True,
                           "dimensions": True, "filters": True, "date_filters": True,
                           "joins": True, "order_by": True, "limit": True},
            field_details={"base_table": {"predicted": "products", "gold": "orders"}},
        )
        ec = classifier.classify(comp, {})
        assert ErrorCategory.WRONG_BASE_TABLE in ec.categories
        assert ec.severity == "critical"

    def test_wrong_metric(self, classifier):
        comp = ComparisonResult(
            example_id="ex3",
            field_matches={"intent": True, "base_table": True, "metrics": False,
                           "dimensions": True, "filters": True, "date_filters": True,
                           "joins": True, "order_by": True, "limit": True},
            field_details={},
        )
        ec = classifier.classify(comp, {})
        assert ErrorCategory.WRONG_METRIC in ec.categories
        assert ec.severity == "major"

    def test_unnecessary_join(self, classifier):
        comp = ComparisonResult(
            example_id="ex4",
            field_matches={"intent": True, "base_table": True, "metrics": True,
                           "dimensions": True, "filters": True, "date_filters": True,
                           "joins": False, "order_by": True, "limit": True},
            field_details={"joins": {
                "predicted": [{"condition": "a.id = b.a_id"}],
                "gold": [],
            }},
        )
        ec = classifier.classify(comp, {})
        assert ErrorCategory.UNNECESSARY_JOIN in ec.categories
        assert ErrorCategory.WRONG_JOIN not in ec.categories

    def test_missing_join(self, classifier):
        comp = ComparisonResult(
            example_id="ex5",
            field_matches={"intent": True, "base_table": True, "metrics": True,
                           "dimensions": True, "filters": True, "date_filters": True,
                           "joins": False, "order_by": True, "limit": True},
            field_details={"joins": {
                "predicted": [],
                "gold": [{"condition": "a.id = b.a_id"}],
            }},
        )
        ec = classifier.classify(comp, {})
        assert ErrorCategory.MISSING_JOIN in ec.categories

    def test_sql_validation_failure(self, classifier):
        comp = ComparisonResult(example_id="ex6", field_matches={})
        example = {"sql_validation": {"is_valid": False, "issues": ["syntax error"]}}
        ec = classifier.classify(comp, example)
        assert ErrorCategory.SQL_VALIDATION_FAILURE in ec.categories

    def test_ir_validation_failure(self, classifier):
        comp = ComparisonResult(example_id="ex7", field_matches={})
        example = {"ir_validation": {"is_valid": False}}
        ec = classifier.classify(comp, example)
        assert ErrorCategory.IR_VALIDATION_FAILURE in ec.categories

    def test_no_errors_for_exact_match(self, classifier):
        comp = ComparisonResult(
            example_id="ex8",
            field_matches={"intent": True, "base_table": True, "metrics": True,
                           "dimensions": True, "filters": True, "date_filters": True,
                           "joins": True, "order_by": True, "limit": True},
        )
        ec = classifier.classify(comp, {})
        assert len(ec.categories) == 0
        assert ec.severity == "minor"

    def test_minor_severity(self, classifier):
        comp = ComparisonResult(
            example_id="ex9",
            field_matches={"intent": True, "base_table": True, "metrics": True,
                           "dimensions": True, "filters": True, "date_filters": True,
                           "joins": True, "order_by": True, "limit": False},
            field_details={},
        )
        ec = classifier.classify(comp, {})
        assert ErrorCategory.WRONG_LIMIT in ec.categories
        assert ec.severity == "minor"


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------

class TestClassifyBatch:
    def test_batch_report(self, classifier):
        comparisons = [
            ComparisonResult(
                example_id="1", is_exact_match=False,
                field_matches={"intent": False, "base_table": True},
                field_details={"intent": {"predicted": "a", "gold": "b"}},
            ),
            ComparisonResult(
                example_id="2", is_exact_match=True,
                field_matches={"intent": True, "base_table": True},
            ),
            ComparisonResult(
                example_id="3", is_exact_match=False,
                field_matches={"intent": True, "base_table": False},
                field_details={"base_table": {"predicted": "x", "gold": "y"}},
            ),
        ]
        examples = [
            {"example_id": "1", "dataset_name": "wikisql"},
            {"example_id": "2", "dataset_name": "wikisql"},
            {"example_id": "3", "dataset_name": "spider"},
        ]
        report = classifier.classify_batch(comparisons, examples)
        assert isinstance(report, ErrorReport)
        assert report.total_errors == 2
        assert "wrong_intent" in report.by_category
        assert "wrong_base_table" in report.by_category
        assert len(report.top_error_categories) >= 2

    def test_empty_batch(self, classifier):
        report = classifier.classify_batch([], [])
        assert report.total_errors == 0
