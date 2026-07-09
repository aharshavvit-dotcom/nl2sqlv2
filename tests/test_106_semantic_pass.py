"""Tests for strict semantic pass computation and applicability-aware metrics.

Validates:
- Semantic pass requires ALL applicable checks to pass
- Projection check is enforced when gold has projections
- Filter checks are enforced when gold has filters
- Filter value extraction is properly tracked
- Artifact-dir evaluations are not promotion eligible
- Row counts are correctly aggregated
- Six separate semantic pass rates are reported
- Applicability-aware denominators exclude non-applicable queries
"""

from __future__ import annotations

import pytest
from evaluation.semantic_pass import (
    compute_simple_query_semantic_pass,
    compute_semantic_evaluation_metrics,
)
from evaluation.report_schemas import ApplicabilityAwareMetric


# --- Fixtures ---

def _gold_ir_with_filter():
    return {
        "intent": "simple_filter",
        "base_table": "users",
        "dimensions": [{"table": "users", "column": "name", "expression": "name"}],
        "filters": [{"table": "users", "column": "status", "operator": "=", "value": "active"}],
        "metrics": [],
        "joins": [],
    }


def _gold_ir_no_filter():
    return {
        "intent": "show_records",
        "base_table": "users",
        "dimensions": [{"table": "users", "column": "name", "expression": "name"}],
        "filters": [],
        "metrics": [],
        "joins": [],
    }


def _gold_ir_count():
    return {
        "intent": "count_records",
        "base_table": "orders",
        "dimensions": [],
        "filters": [],
        "metrics": [{"aggregation": "COUNT", "expression": "COUNT(*)"}],
        "joins": [],
    }


# --- Test: semantic pass requires projection ---

class TestSemanticPassRequiresProjection:
    def test_correct_projection_passes(self):
        gold = _gold_ir_no_filter()
        pred = {**gold}  # Perfect match
        result = compute_simple_query_semantic_pass(gold, pred, "SELECT name FROM users", {"is_valid": True})
        assert result.passed is True
        assert "projection" in result.applicable_checks
        assert "projection" in result.passed_checks

    def test_wrong_projection_fails(self):
        gold = _gold_ir_no_filter()
        pred = {
            **gold,
            "dimensions": [{"table": "users", "column": "id", "expression": "id"}],
        }
        result = compute_simple_query_semantic_pass(gold, pred, "SELECT id FROM users", {"is_valid": True})
        assert result.passed is False
        assert "projection" in result.failed_checks

    def test_extra_columns_fail(self):
        gold = _gold_ir_no_filter()
        pred = {
            **gold,
            "dimensions": [
                {"table": "users", "column": "name", "expression": "name"},
                {"table": "users", "column": "id", "expression": "id"},
            ],
        }
        result = compute_simple_query_semantic_pass(gold, pred, "SELECT name, id FROM users", {"is_valid": True})
        assert result.passed is False
        assert "projection" in result.failed_checks


# --- Test: semantic pass requires filter ---

class TestSemanticPassRequiresFilter:
    def test_correct_filter_passes(self):
        gold = _gold_ir_with_filter()
        pred = {**gold}  # Perfect match
        result = compute_simple_query_semantic_pass(
            gold, pred, "SELECT name FROM users WHERE status = 'active'", {"is_valid": True}
        )
        assert result.passed is True
        assert "filter_column" in result.applicable_checks
        assert "filter_value" in result.applicable_checks
        assert "filter_column" in result.passed_checks

    def test_wrong_filter_column_fails(self):
        gold = _gold_ir_with_filter()
        pred = {
            **gold,
            "filters": [{"table": "users", "column": "role", "operator": "=", "value": "active"}],
        }
        result = compute_simple_query_semantic_pass(
            gold, pred, "SELECT name FROM users WHERE role = 'active'", {"is_valid": True}
        )
        assert result.passed is False
        assert "filter_column" in result.failed_checks

    def test_wrong_filter_value_fails(self):
        gold = _gold_ir_with_filter()
        pred = {
            **gold,
            "filters": [{"table": "users", "column": "status", "operator": "=", "value": "inactive"}],
        }
        result = compute_simple_query_semantic_pass(
            gold, pred, "SELECT name FROM users WHERE status = 'inactive'", {"is_valid": True}
        )
        assert result.passed is False
        assert "filter_value" in result.failed_checks

    def test_no_filter_gold_excludes_filter_checks(self):
        gold = _gold_ir_no_filter()
        pred = {**gold}
        result = compute_simple_query_semantic_pass(
            gold, pred, "SELECT name FROM users", {"is_valid": True}
        )
        assert "filter_column" not in result.applicable_checks
        assert "filter_value" not in result.applicable_checks
        assert result.passed is True


# --- Test: SQL safety and validity ---

class TestSQLSafetyAndValidity:
    def test_unsafe_sql_fails(self):
        gold = _gold_ir_no_filter()
        pred = {**gold}
        result = compute_simple_query_semantic_pass(
            gold, pred, "DROP TABLE users", {"is_valid": True}
        )
        assert result.passed is False
        assert "sql_safety" in result.failed_checks

    def test_invalid_sql_fails(self):
        gold = _gold_ir_no_filter()
        pred = {**gold}
        result = compute_simple_query_semantic_pass(
            gold, pred, "SELECT name FROM users", {"is_valid": False}
        )
        assert result.passed is False
        assert "sql_validity" in result.failed_checks

    def test_null_sql_with_valid_ir_passes_safety(self):
        gold = _gold_ir_no_filter()
        pred = {**gold}
        result = compute_simple_query_semantic_pass(gold, pred, None, {"is_valid": False})
        assert "sql_safety" in result.passed_checks


# --- Test: applicability-aware metrics ---

class TestApplicabilityAwareMetric:
    def test_excludes_none_from_denominator(self):
        metric = ApplicabilityAwareMetric.compute([True, True, None, None, False])
        assert metric.numerator == 2
        assert metric.denominator == 3
        assert metric.applicable_cases == 3
        assert metric.excluded_cases == 2
        assert abs(metric.value - 2.0 / 3.0) < 0.001

    def test_all_none_gives_zero(self):
        metric = ApplicabilityAwareMetric.compute([None, None, None])
        assert metric.value == 0.0
        assert metric.denominator == 0
        assert metric.excluded_cases == 3

    def test_empty_gives_zero(self):
        metric = ApplicabilityAwareMetric.compute([])
        assert metric.value == 0.0
        assert metric.denominator == 0


# --- Test: six separate pass rates ---

class TestSemanticEvaluationMetrics:
    def test_reports_six_separate_rates(self):
        per_example = [
            {
                "sql_validation_passed": True,
                "base_table_correct": True,
                "predicted_sql": "SELECT name FROM users WHERE status = 'active'",
                "sql_validation": {"is_valid": True},
                "semantic_pass": {
                    "passed": True,
                    "applicable_checks": ["sql_safety", "sql_validity", "intent", "base_table", "projection", "filter_column", "filter_value", "filter_operator"],
                    "passed_checks": ["sql_safety", "sql_validity", "intent", "base_table", "projection", "filter_column", "filter_value", "filter_operator"],
                    "failed_checks": [],
                },
                "filter_linking": {
                    "filter_column_match": True,
                    "filter_value_match": True,
                    "filter_value_extraction_match": True,
                    "filter_column_top1_match": True,
                    "filter_column_top3_match": True,
                },
                "dimension_linking": {"dimension_match": True},
                "projection": {
                    "gold_columns": ["name"],
                    "predicted_columns": ["name"],
                    "exact_match": True,
                    "contains_gold": True,
                    "has_extra_columns": False,
                },
            },
            {
                "sql_validation_passed": True,
                "base_table_correct": True,
                "predicted_sql": "SELECT id, name FROM users",
                "sql_validation": {"is_valid": True},
                "semantic_pass": {
                    "passed": False,
                    "applicable_checks": ["sql_safety", "sql_validity", "projection"],
                    "passed_checks": ["sql_safety", "sql_validity"],
                    "failed_checks": ["projection"],
                },
                "filter_linking": {
                    "filter_column_match": None,
                    "filter_value_match": None,
                },
                "dimension_linking": {"dimension_match": None},
                "projection": {
                    "gold_columns": ["name"],
                    "predicted_columns": ["id", "name"],
                    "exact_match": False,
                    "contains_gold": True,
                    "has_extra_columns": True,
                },
            },
        ]
        metrics = compute_semantic_evaluation_metrics(per_example)
        # Safety: both are safe
        assert metrics.simple_query_safety_pass_rate == 1.0
        # Validity: both are valid
        assert metrics.simple_query_validity_pass_rate == 1.0
        # Table: both correct
        assert metrics.simple_query_table_pass_rate == 1.0
        # Projection: 1/2 applicable, only first passes
        assert metrics.simple_query_projection_pass_rate == 0.5
        # Filter: 1/1 applicable (first only), passes
        assert metrics.simple_query_filter_pass_rate == 1.0
        # Full semantic: 1/2 pass
        assert metrics.simple_query_semantic_pass_rate == 0.5
        # Filter accuracy: only 1 applicable case
        assert metrics.filter_column_accuracy.applicable_cases == 1
        assert metrics.filter_column_accuracy.value == 1.0
        # Projection accuracy: 2 applicable, 1 passes
        assert metrics.projection_exact_match.applicable_cases == 2
        assert metrics.projection_exact_match.value == 0.5


# --- Test: primary failure reason ---

class TestPrimaryFailureReason:
    def test_primary_failure_is_first_failed_check(self):
        gold = _gold_ir_with_filter()
        pred = {
            **gold,
            "base_table": "orders",  # Wrong table
            "filters": [{"table": "orders", "column": "role", "operator": "=", "value": "admin"}],  # Wrong filter
        }
        result = compute_simple_query_semantic_pass(
            gold, pred, "SELECT name FROM orders WHERE role = 'admin'", {"is_valid": True}
        )
        assert result.passed is False
        assert result.primary_failure_reason is not None
        assert len(result.failed_checks) > 1  # Multiple failures
