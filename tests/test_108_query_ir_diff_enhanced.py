"""Tests for enhanced canonical QueryIR diff.

Validates:
- Projection detail (exact match, contains gold, extra/missing columns)
- Separate dimension_match from projection_match
- Failure categories list
- distinct_match
- Backward compatibility for existing fields
"""

from __future__ import annotations

import pytest
from ir.query_ir_models import diff_query_ir


class TestProjectionDetail:
    def test_exact_match(self):
        gold = {
            "dimensions": [
                {"table": "users", "column": "name", "expression": "name"},
                {"table": "users", "column": "email", "expression": "email"},
            ]
        }
        pred = {
            "dimensions": [
                {"table": "users", "column": "name", "expression": "name"},
                {"table": "users", "column": "email", "expression": "email"},
            ]
        }
        diff = diff_query_ir(pred, gold)
        assert diff["projection_exact_match"] is True
        assert diff["projection_contains_gold"] is True
        assert diff["extra_projection_columns"] == []
        assert diff["missing_projection_columns"] == []

    def test_extra_columns(self):
        gold = {"dimensions": [{"table": "users", "column": "name", "expression": "name"}]}
        pred = {
            "dimensions": [
                {"table": "users", "column": "name", "expression": "name"},
                {"table": "users", "column": "id", "expression": "id"},
            ]
        }
        diff = diff_query_ir(pred, gold)
        assert diff["projection_exact_match"] is False
        assert diff["projection_contains_gold"] is True
        assert len(diff["extra_projection_columns"]) == 1
        assert diff["missing_projection_columns"] == []

    def test_missing_columns(self):
        gold = {
            "dimensions": [
                {"table": "users", "column": "name", "expression": "name"},
                {"table": "users", "column": "email", "expression": "email"},
            ]
        }
        pred = {"dimensions": [{"table": "users", "column": "name", "expression": "name"}]}
        diff = diff_query_ir(pred, gold)
        assert diff["projection_exact_match"] is False
        assert diff["projection_contains_gold"] is False
        assert diff["extra_projection_columns"] == []
        assert len(diff["missing_projection_columns"]) == 1


class TestDimensionSeparateFromProjection:
    def test_dimension_match_uses_table_column(self):
        gold = {"dimensions": [{"table": "orders", "column": "status", "expression": "status"}]}
        pred = {"dimensions": [{"table": "orders", "column": "status", "expression": "status"}]}
        diff = diff_query_ir(pred, gold)
        assert diff["dimension_match"] is True
        assert diff["dimension_column_match"] is True

    def test_dimension_mismatch_separate_from_projection(self):
        gold = {"dimensions": [{"table": "orders", "column": "status", "expression": "status"}]}
        pred = {"dimensions": [{"table": "orders", "column": "region", "expression": "region"}]}
        diff = diff_query_ir(pred, gold)
        assert diff["dimension_match"] is False
        assert diff["dimension_column_match"] is False


class TestFailureCategories:
    def test_no_failures_empty_list(self):
        gold = {"intent": "show_records", "base_table": "users"}
        pred = {"intent": "show_records", "base_table": "users"}
        diff = diff_query_ir(pred, gold)
        assert diff["failure_categories"] == []
        assert diff["all_slots_match"] is True

    def test_multiple_failures_listed(self):
        gold = {
            "intent": "show_records",
            "base_table": "users",
            "filters": [{"table": "users", "column": "status", "operator": "=", "value": "active"}],
        }
        pred = {
            "intent": "count_records",
            "base_table": "orders",
            "filters": [{"table": "orders", "column": "role", "operator": "!=", "value": "admin"}],
        }
        diff = diff_query_ir(pred, gold)
        assert "intent_mismatch" in diff["failure_categories"]
        assert "base_table_mismatch" in diff["failure_categories"]
        assert "filter_column_mismatch" in diff["failure_categories"]
        assert "filter_operator_mismatch" in diff["failure_categories"]
        assert "filter_value_mismatch" in diff["failure_categories"]
        assert diff["all_slots_match"] is False

    def test_projection_missing_column_category(self):
        gold = {
            "dimensions": [
                {"table": "users", "column": "name", "expression": "name"},
                {"table": "users", "column": "email", "expression": "email"},
            ]
        }
        pred = {"dimensions": [{"table": "users", "column": "name", "expression": "name"}]}
        diff = diff_query_ir(pred, gold)
        assert "projection_missing_column" in diff["failure_categories"]
        assert "projection_mismatch" in diff["failure_categories"]

    def test_projection_extra_column_category(self):
        gold = {"dimensions": [{"table": "users", "column": "name", "expression": "name"}]}
        pred = {
            "dimensions": [
                {"table": "users", "column": "name", "expression": "name"},
                {"table": "users", "column": "id", "expression": "id"},
            ]
        }
        diff = diff_query_ir(pred, gold)
        assert "projection_extra_column" in diff["failure_categories"]
        assert "projection_mismatch" in diff["failure_categories"]


class TestDistinctMatch:
    def test_both_distinct_match(self):
        gold = {"distinct": True}
        pred = {"distinct": True}
        diff = diff_query_ir(pred, gold)
        assert diff["distinct_match"] is True

    def test_distinct_mismatch(self):
        gold = {"distinct": True}
        pred = {"distinct": False}
        diff = diff_query_ir(pred, gold)
        assert diff["distinct_match"] is False

    def test_both_no_distinct_match(self):
        gold = {}
        pred = {}
        diff = diff_query_ir(pred, gold)
        assert diff["distinct_match"] is True


class TestBackwardCompatibility:
    def test_selected_columns_match_is_projection_alias(self):
        gold = {"dimensions": [{"table": "users", "column": "name", "expression": "name"}]}
        pred = {"dimensions": [{"table": "users", "column": "name", "expression": "name"}]}
        diff = diff_query_ir(pred, gold)
        assert diff["selected_columns_match"] == diff["projection_exact_match"]

    def test_filters_match_is_composite(self):
        gold = {"filters": [{"table": "users", "column": "status", "operator": "=", "value": "active"}]}
        pred = {"filters": [{"table": "users", "column": "status", "operator": "=", "value": "active"}]}
        diff = diff_query_ir(pred, gold)
        assert diff["filters_match"] is True
        assert diff["filters_match"] == (
            diff["filter_column_match"] and diff["filter_value_match"] and diff["filter_operator_match"]
        )

    def test_primary_failure_slot_includes_dimension(self):
        gold = {"dimensions": [{"table": "orders", "column": "status", "expression": "status"}]}
        pred = {"dimensions": [{"table": "orders", "column": "region", "expression": "region"}]}
        diff = diff_query_ir(pred, gold)
        # dimension is in the failure_order now
        assert diff["primary_failure_slot"] is not None
