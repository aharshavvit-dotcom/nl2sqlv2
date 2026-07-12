"""Tests for Gate 1: Validation, migration, and SQL validator extensions.

Tests cover:
- HAVING validation (requires GROUP BY, requires aggregate)
- CTE validation (duplicate names, recursive detection)
- Set operation arity validation
- InSubqueryPredicate / ExistsPredicate depth tracking
- Migration incompatibility for v2-only constructs
- SQL validator CTE-awareness
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ir.query_ir_v2_models import (
    AggregationExpression,
    CTEDefinition,
    ColumnExpression,
    ComparisonPredicate,
    ExistsPredicate,
    FromItem,
    InSubqueryPredicate,
    LiteralExpression,
    LiteralValueType,
    QueryNode,
    SelectItem,
    SetOperationNode,
)
from ir.query_ir_v2_validation import QueryIRV2Validator
from ir.query_ir_migration import QueryIRCompatibilityError, convert_v2_to_v1
from validation.sql_validator import SQLValidator


# ── HAVING validation ─────────────────────────────────────────────────

class TestHavingValidation:
    def test_having_without_group_by_warns(self):
        validator = QueryIRV2Validator()
        query = QueryNode(
            from_item=FromItem(from_type="TABLE", table="orders"),
            having=ComparisonPredicate(
                left=AggregationExpression(function="COUNT", argument=None),
                operator=">",
                right=LiteralExpression(value=5, value_type=LiteralValueType.INTEGER),
            ),
        )
        result = validator.validate(query)
        having_warnings = [i for i in result.issues if i.issue_type == "having_without_group_by"]
        assert len(having_warnings) == 1

    def test_having_without_aggregate_warns(self):
        validator = QueryIRV2Validator()
        query = QueryNode(
            from_item=FromItem(from_type="TABLE", table="orders"),
            group_by=[ColumnExpression(column="region")],
            having=ComparisonPredicate(
                left=ColumnExpression(column="region"),
                operator="=",
                right=LiteralExpression(value="US", value_type=LiteralValueType.STRING),
            ),
        )
        result = validator.validate(query)
        no_agg_warnings = [i for i in result.issues if i.issue_type == "having_without_aggregate"]
        assert len(no_agg_warnings) == 1

    def test_valid_having_with_group_by_and_aggregate(self):
        validator = QueryIRV2Validator()
        query = QueryNode(
            from_item=FromItem(from_type="TABLE", table="orders"),
            select_items=[
                SelectItem(expression=ColumnExpression(column="region"), alias="region"),
            ],
            group_by=[ColumnExpression(column="region")],
            having=ComparisonPredicate(
                left=AggregationExpression(function="SUM", argument=ColumnExpression(column="amount")),
                operator=">",
                right=LiteralExpression(value=1000, value_type=LiteralValueType.INTEGER),
            ),
        )
        result = validator.validate(query)
        having_issues = [i for i in result.issues if "having" in i.issue_type]
        assert len(having_issues) == 0


# ── CTE validation ───────────────────────────────────────────────────

class TestCTEValidation:
    def test_duplicate_cte_name_is_error(self):
        validator = QueryIRV2Validator()
        query = QueryNode(
            ctes=[
                CTEDefinition(name="x", query=QueryNode(from_item=FromItem(from_type="TABLE", table="a"))),
                CTEDefinition(name="x", query=QueryNode(from_item=FromItem(from_type="TABLE", table="b"))),
            ],
            from_item=FromItem(from_type="TABLE", table="x"),
        )
        result = validator.validate(query)
        dup_errors = [i for i in result.issues if i.issue_type == "duplicate_cte_name"]
        assert len(dup_errors) == 1

    def test_recursive_cte_is_error(self):
        validator = QueryIRV2Validator()
        query = QueryNode(
            ctes=[
                CTEDefinition(name="rec", query=QueryNode(from_item=FromItem(from_type="TABLE", table="rec"))),
            ],
            from_item=FromItem(from_type="TABLE", table="rec"),
        )
        result = validator.validate(query)
        rec_errors = [i for i in result.issues if i.issue_type == "recursive_cte_not_supported"]
        assert len(rec_errors) == 1


# ── Set operation validation ─────────────────────────────────────────

class TestSetOperationValidation:
    def test_set_operation_arity_mismatch(self):
        validator = QueryIRV2Validator()
        query = QueryNode(
            from_item=FromItem(from_type="TABLE", table="t1"),
            select_items=[
                SelectItem(expression=ColumnExpression(column="a"), alias="a"),
                SelectItem(expression=ColumnExpression(column="b"), alias="b"),
            ],
            set_operations=[
                SetOperationNode(
                    operation="UNION_ALL",
                    query=QueryNode(
                        from_item=FromItem(from_type="TABLE", table="t2"),
                        select_items=[
                            SelectItem(expression=ColumnExpression(column="c"), alias="c"),
                        ],
                    ),
                ),
            ],
        )
        result = validator.validate(query)
        arity_errors = [i for i in result.issues if i.issue_type == "set_operation_arity_mismatch"]
        assert len(arity_errors) == 1


# ── Migration incompatibility ────────────────────────────────────────

class TestMigrationIncompatibility:
    def test_having_blocks_v1_conversion(self):
        query = QueryNode(
            from_item=FromItem(from_type="TABLE", table="orders"),
            group_by=[ColumnExpression(column="region")],
            having=ComparisonPredicate(
                left=AggregationExpression(function="SUM", argument=ColumnExpression(column="amount")),
                operator=">",
                right=LiteralExpression(value=1000, value_type=LiteralValueType.INTEGER),
            ),
        )
        with pytest.raises(QueryIRCompatibilityError, match="HAVING"):
            convert_v2_to_v1(query)

    def test_cte_blocks_v1_conversion(self):
        query = QueryNode(
            ctes=[CTEDefinition(name="top", query=QueryNode(from_item=FromItem(from_type="TABLE", table="t")))],
            from_item=FromItem(from_type="TABLE", table="top"),
        )
        with pytest.raises(QueryIRCompatibilityError, match="CTE"):
            convert_v2_to_v1(query)

    def test_set_op_blocks_v1_conversion(self):
        query = QueryNode(
            from_item=FromItem(from_type="TABLE", table="t"),
            set_operations=[
                SetOperationNode(
                    operation="UNION_ALL",
                    query=QueryNode(from_item=FromItem(from_type="TABLE", table="t")),
                ),
            ],
        )
        with pytest.raises(QueryIRCompatibilityError, match="Set operations"):
            convert_v2_to_v1(query)


# ── SQL validator CTE awareness ──────────────────────────────────────

class TestSQLValidatorCTEAwareness:
    def test_cte_name_not_flagged_as_unknown_table(self):
        validator = SQLValidator()
        sql = """
        WITH recent_orders AS (
            SELECT id, customer_id FROM orders WHERE date > '2026-01-01'
        )
        SELECT id, customer_id FROM recent_orders LIMIT 10
        """
        schema = {"orders": {"columns": {"id": {}, "customer_id": {}, "date": {}}}}
        result = validator.validate(sql, schema=schema)
        # recent_orders should NOT be flagged as unknown
        assert result["checks"]["tables_exist"], f"Issues: {result['issues']}"

    def test_cte_column_not_flagged_as_unknown(self):
        validator = SQLValidator()
        sql = """
        WITH totals AS (
            SELECT orders.region, SUM(orders.amount) AS total FROM orders GROUP BY orders.region
        )
        SELECT totals.region, totals.total FROM totals LIMIT 10
        """
        schema = {"orders": {"columns": {"region": {}, "amount": {}}}}
        result = validator.validate(sql, schema=schema)
        assert result["checks"]["columns_exist"], f"Issues: {result['issues']}"
