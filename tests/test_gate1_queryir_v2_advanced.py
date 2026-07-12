"""Tests for Gate 1: QueryIR v2 advanced constructs.

Tests cover:
- InSubqueryPredicate and ExistsPredicate models
- CTEDefinition model
- Typed FrameBound and WindowSpecification
- LiteralValueType strict types
- HAVING field on QueryNode
- Scope analyzer (derived analysis)
- Renderer support for GROUP BY, HAVING, CASE, subqueries, windows, CTEs, set ops
- Three-valued logic correctness
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ir.query_ir_v2_models import (
    AggregationExpression,
    BooleanPredicate,
    CTEDefinition,
    CaseExpression,
    CaseWhen,
    ColumnExpression,
    ComparisonPredicate,
    ExistsPredicate,
    FrameBound,
    FrameBoundType,
    FromItem,
    InSubqueryPredicate,
    JoinNode,
    LiteralExpression,
    LiteralValueType,
    OrderByItem,
    QueryNode,
    SelectItem,
    SetOperationNode,
    SubqueryExpression,
    WindowExpression,
    WindowSpecification,
)
from ir.query_ir_v2_boolean_renderer import QueryIRV2NativeRenderer, QueryIRV2RenderingError
from ir.query_ir_v2_scope import QueryScopeAnalyzer
from ir.query_ir_v2_three_valued_logic import TruthValue, sql_and, sql_not, sql_or


# ── InSubqueryPredicate ───────────────────────────────────────────────

class TestInSubqueryPredicate:
    def test_basic_in_subquery(self):
        pred = InSubqueryPredicate(
            expression=ColumnExpression(table="orders", column="customer_id"),
            query=QueryNode(
                from_item=FromItem(from_type="TABLE", table="vip_customers"),
                select_items=[SelectItem(expression=ColumnExpression(table="vip_customers", column="id"), alias="id")],
            ),
        )
        assert pred.predicate_type == "IN_SUBQUERY_PREDICATE"
        assert pred.negated is False

    def test_not_in_subquery(self):
        pred = InSubqueryPredicate(
            expression=ColumnExpression(table="orders", column="customer_id"),
            query=QueryNode(from_item=FromItem(from_type="TABLE", table="blocked")),
            negated=True,
        )
        assert pred.negated is True

    def test_renders_in_subquery(self):
        renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
        pred = InSubqueryPredicate(
            expression=ColumnExpression(table="orders", column="customer_id"),
            query=QueryNode(
                from_item=FromItem(from_type="TABLE", table="vip_customers"),
                select_items=[SelectItem(expression=ColumnExpression(table="vip_customers", column="id"), alias="id")],
                limit=None,
            ),
        )
        sql = renderer.render_predicate(pred, dialect="sqlite")
        assert "IN" in sql
        assert "vip_customers" in sql

    def test_renders_not_in_subquery(self):
        renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
        pred = InSubqueryPredicate(
            expression=ColumnExpression(table="orders", column="customer_id"),
            query=QueryNode(
                from_item=FromItem(from_type="TABLE", table="blocked"),
                select_items=[SelectItem(expression=ColumnExpression(column="id"), alias="id")],
                limit=None,
            ),
            negated=True,
        )
        sql = renderer.render_predicate(pred, dialect="sqlite")
        assert "NOT IN" in sql


# ── ExistsPredicate ───────────────────────────────────────────────────

class TestExistsPredicate:
    def test_basic_exists(self):
        pred = ExistsPredicate(
            query=QueryNode(
                from_item=FromItem(from_type="TABLE", table="orders"),
                select_items=[SelectItem(expression=LiteralExpression(value=1, value_type=LiteralValueType.INTEGER), alias="c")],
            ),
        )
        assert pred.predicate_type == "EXISTS_PREDICATE"
        assert pred.negated is False

    def test_not_exists(self):
        pred = ExistsPredicate(
            query=QueryNode(from_item=FromItem(from_type="TABLE", table="orders")),
            negated=True,
        )
        assert pred.negated is True

    def test_renders_exists(self):
        renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
        pred = ExistsPredicate(
            query=QueryNode(
                from_item=FromItem(from_type="TABLE", table="orders"),
                select_items=[SelectItem(expression=LiteralExpression(value=1, value_type=LiteralValueType.INTEGER), alias="c")],
                limit=None,
            ),
        )
        sql = renderer.render_predicate(pred, dialect="sqlite")
        assert sql.startswith("EXISTS")

    def test_renders_not_exists(self):
        renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
        pred = ExistsPredicate(
            query=QueryNode(
                from_item=FromItem(from_type="TABLE", table="orders"),
                limit=None,
            ),
            negated=True,
        )
        sql = renderer.render_predicate(pred, dialect="sqlite")
        assert sql.startswith("NOT EXISTS")


# ── CTEDefinition ─────────────────────────────────────────────────────

class TestCTEDefinition:
    def test_basic_cte(self):
        cte = CTEDefinition(
            name="regional_sales",
            query=QueryNode(from_item=FromItem(from_type="TABLE", table="orders")),
        )
        assert cte.name == "regional_sales"
        assert cte.columns == []

    def test_cte_with_columns(self):
        cte = CTEDefinition(
            name="ranked",
            columns=["customer_id", "total"],
            query=QueryNode(from_item=FromItem(from_type="TABLE", table="orders")),
        )
        assert len(cte.columns) == 2

    def test_cte_rejects_blank_name(self):
        with pytest.raises(ValidationError):
            CTEDefinition(
                name="  ",
                query=QueryNode(from_item=FromItem(from_type="TABLE", table="t")),
            )

    def test_renders_cte(self):
        renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
        query = QueryNode(
            ctes=[CTEDefinition(
                name="top_customers",
                query=QueryNode(
                    from_item=FromItem(from_type="TABLE", table="customers"),
                    select_items=[SelectItem(expression=ColumnExpression(table="customers", column="id"), alias="id")],
                    limit=10,
                ),
            )],
            from_item=FromItem(from_type="TABLE", table="top_customers"),
            select_items=[SelectItem(expression=ColumnExpression(column="id"), alias="id")],
            limit=None,
        )
        sql = renderer.render(query)
        assert "WITH" in sql
        assert "top_customers" in sql


# ── FrameBound and WindowSpecification ────────────────────────────────

class TestFrameBound:
    def test_unbounded_preceding(self):
        fb = FrameBound(bound_type=FrameBoundType.UNBOUNDED_PRECEDING)
        assert fb.offset is None

    def test_n_preceding_requires_offset(self):
        fb = FrameBound(bound_type=FrameBoundType.N_PRECEDING, offset=3)
        assert fb.offset == 3

    def test_n_preceding_rejects_missing_offset(self):
        with pytest.raises(ValidationError):
            FrameBound(bound_type=FrameBoundType.N_PRECEDING)

    def test_n_preceding_rejects_negative_offset(self):
        with pytest.raises(ValidationError):
            FrameBound(bound_type=FrameBoundType.N_PRECEDING, offset=-1)

    def test_unbounded_rejects_offset(self):
        with pytest.raises(ValidationError):
            FrameBound(bound_type=FrameBoundType.UNBOUNDED_PRECEDING, offset=5)

    def test_current_row(self):
        fb = FrameBound(bound_type=FrameBoundType.CURRENT_ROW)
        assert fb.bound_type == FrameBoundType.CURRENT_ROW


class TestWindowSpecification:
    def test_frame_ordering_enforced(self):
        """Frame start cannot come after frame end."""
        with pytest.raises(ValidationError):
            WindowSpecification(
                frame_type="ROWS",
                frame_start=FrameBound(bound_type=FrameBoundType.UNBOUNDED_FOLLOWING),
                frame_end=FrameBound(bound_type=FrameBoundType.UNBOUNDED_PRECEDING),
            )

    def test_valid_rows_between(self):
        spec = WindowSpecification(
            partition_by=[ColumnExpression(column="region")],
            order_by=[OrderByItem(expression=ColumnExpression(column="amount"), direction="DESC")],
            frame_type="ROWS",
            frame_start=FrameBound(bound_type=FrameBoundType.UNBOUNDED_PRECEDING),
            frame_end=FrameBound(bound_type=FrameBoundType.CURRENT_ROW),
        )
        assert spec.frame_type == "ROWS"


# ── HAVING on QueryNode ──────────────────────────────────────────────

class TestHavingField:
    def test_having_field_is_none_by_default(self):
        q = QueryNode(from_item=FromItem(from_type="TABLE", table="t"))
        assert q.having is None

    def test_having_field_accepts_predicate(self):
        q = QueryNode(
            from_item=FromItem(from_type="TABLE", table="orders"),
            group_by=[ColumnExpression(table="orders", column="region")],
            having=ComparisonPredicate(
                left=AggregationExpression(function="SUM", argument=ColumnExpression(table="orders", column="amount")),
                operator=">",
                right=LiteralExpression(value=1000, value_type=LiteralValueType.INTEGER),
            ),
        )
        assert q.having is not None

    def test_renders_group_by_having(self):
        renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
        query = QueryNode(
            from_item=FromItem(from_type="TABLE", table="orders"),
            select_items=[
                SelectItem(expression=ColumnExpression(table="orders", column="region"), alias="region"),
                SelectItem(
                    expression=AggregationExpression(function="SUM", argument=ColumnExpression(table="orders", column="amount")),
                    alias="total",
                ),
            ],
            group_by=[ColumnExpression(table="orders", column="region")],
            having=ComparisonPredicate(
                left=AggregationExpression(function="SUM", argument=ColumnExpression(table="orders", column="amount")),
                operator=">",
                right=LiteralExpression(value=1000, value_type=LiteralValueType.INTEGER),
            ),
            limit=None,
        )
        sql = renderer.render(query)
        assert "GROUP BY" in sql
        assert "HAVING" in sql
        assert "SUM" in sql


# ── CASE expression rendering ────────────────────────────────────────

class TestCaseRendering:
    def test_renders_case_expression(self):
        renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
        case = CaseExpression(
            cases=[
                CaseWhen(
                    when=ComparisonPredicate(
                        left=ColumnExpression(column="status"),
                        operator="=",
                        right=LiteralExpression(value="active", value_type=LiteralValueType.STRING),
                    ),
                    then=LiteralExpression(value=1, value_type=LiteralValueType.INTEGER),
                ),
            ],
            else_expression=LiteralExpression(value=0, value_type=LiteralValueType.INTEGER),
        )
        sql = renderer.render_expression(case, "sqlite")
        assert "CASE" in sql
        assert "WHEN" in sql
        assert "THEN" in sql
        assert "ELSE" in sql
        assert "END" in sql


# ── Window rendering ─────────────────────────────────────────────────

class TestWindowRendering:
    def test_renders_window_with_partition_and_order(self):
        renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
        win = WindowExpression(
            expression=AggregationExpression(function="SUM", argument=ColumnExpression(column="amount")),
            window=WindowSpecification(
                partition_by=[ColumnExpression(column="region")],
                order_by=[OrderByItem(expression=ColumnExpression(column="date"), direction="ASC")],
            ),
        )
        sql = renderer.render_expression(win, "sqlite")
        assert "OVER" in sql
        assert "PARTITION BY" in sql
        assert "ORDER BY" in sql

    def test_renders_window_with_frame(self):
        renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
        win = WindowExpression(
            expression=AggregationExpression(function="SUM", argument=ColumnExpression(column="amount")),
            window=WindowSpecification(
                order_by=[OrderByItem(expression=ColumnExpression(column="date"), direction="ASC")],
                frame_type="ROWS",
                frame_start=FrameBound(bound_type=FrameBoundType.UNBOUNDED_PRECEDING),
                frame_end=FrameBound(bound_type=FrameBoundType.CURRENT_ROW),
            ),
        )
        sql = renderer.render_expression(win, "sqlite")
        assert "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW" in sql


# ── Set operations rendering ─────────────────────────────────────────

class TestSetOperationRendering:
    def test_renders_union(self):
        renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
        query = QueryNode(
            from_item=FromItem(from_type="TABLE", table="orders_2024"),
            select_items=[SelectItem(expression=ColumnExpression(column="id"), alias="id")],
            set_operations=[
                SetOperationNode(
                    operation="UNION_ALL",
                    query=QueryNode(
                        from_item=FromItem(from_type="TABLE", table="orders_2025"),
                        select_items=[SelectItem(expression=ColumnExpression(column="id"), alias="id")],
                        limit=None,
                    ),
                ),
            ],
            limit=None,
        )
        sql = renderer.render(query)
        assert "UNION ALL" in sql
        assert "orders_2024" in sql
        assert "orders_2025" in sql


# ── Subquery FROM rendering ──────────────────────────────────────────

class TestSubqueryFromRendering:
    def test_renders_subquery_from(self):
        renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
        query = QueryNode(
            from_item=FromItem(
                from_type="SUBQUERY",
                alias="sub",
                query=QueryNode(
                    from_item=FromItem(from_type="TABLE", table="orders"),
                    select_items=[SelectItem(expression=ColumnExpression(column="id"), alias="id")],
                    limit=None,
                ),
            ),
            select_items=[SelectItem(expression=ColumnExpression(table="sub", column="id"), alias="id")],
            limit=None,
        )
        sql = renderer.render(query)
        assert 'AS "sub"' in sql or "AS sub" in sql


# ── Scope analyzer ───────────────────────────────────────────────────

class TestScopeAnalyzer:
    def test_simple_query_depth_zero(self):
        analyzer = QueryScopeAnalyzer()
        query = QueryNode(from_item=FromItem(from_type="TABLE", table="orders"))
        analysis = analyzer.analyze(query)
        assert analysis.subquery_depth == 0
        assert analysis.node_count >= 1

    def test_subquery_increments_depth(self):
        analyzer = QueryScopeAnalyzer()
        query = QueryNode(
            from_item=FromItem(from_type="TABLE", table="orders"),
            where=InSubqueryPredicate(
                expression=ColumnExpression(column="id"),
                query=QueryNode(
                    from_item=FromItem(from_type="TABLE", table="vip"),
                    select_items=[SelectItem(expression=ColumnExpression(column="id"), alias="id")],
                ),
            ),
        )
        analysis = analyzer.analyze(query)
        assert analysis.subquery_depth >= 1

    def test_cte_registers_binding(self):
        analyzer = QueryScopeAnalyzer()
        query = QueryNode(
            ctes=[CTEDefinition(
                name="recent",
                query=QueryNode(from_item=FromItem(from_type="TABLE", table="orders")),
            )],
            from_item=FromItem(from_type="TABLE", table="recent"),
        )
        analysis = analyzer.analyze(query)
        cte_bindings = [b for b in analysis.relation_bindings if b.source_type == "CTE"]
        assert len(cte_bindings) == 1
        assert cte_bindings[0].alias == "recent"

    def test_depth_limit_exceeded_diagnostic(self):
        analyzer = QueryScopeAnalyzer(max_subquery_depth=1)
        deep_query = QueryNode(
            from_item=FromItem(from_type="TABLE", table="t1"),
            where=InSubqueryPredicate(
                expression=ColumnExpression(column="id"),
                query=QueryNode(
                    from_item=FromItem(from_type="TABLE", table="t2"),
                    where=InSubqueryPredicate(
                        expression=ColumnExpression(column="id"),
                        query=QueryNode(from_item=FromItem(from_type="TABLE", table="t3")),
                    ),
                ),
            ),
        )
        analysis = analyzer.analyze(deep_query)
        depth_errors = [d for d in analysis.diagnostics if d.code == "max_subquery_depth_exceeded"]
        assert len(depth_errors) == 1


# ── Three-valued logic ───────────────────────────────────────────────

class TestThreeValuedLogic:
    def test_not_unknown_is_unknown(self):
        assert sql_not(TruthValue.UNKNOWN) is TruthValue.UNKNOWN

    def test_true_or_unknown_is_true(self):
        assert sql_or([TruthValue.TRUE, TruthValue.UNKNOWN]) is TruthValue.TRUE

    def test_false_and_unknown_is_false(self):
        assert sql_and([TruthValue.FALSE, TruthValue.UNKNOWN]) is TruthValue.FALSE

    def test_unknown_and_unknown_is_unknown(self):
        assert sql_and([TruthValue.UNKNOWN, TruthValue.UNKNOWN]) is TruthValue.UNKNOWN

    def test_unknown_or_not_unknown_is_unknown(self):
        """A OR NOT A is NOT TRUE when A is UNKNOWN — no excluded middle."""
        result = sql_or([TruthValue.UNKNOWN, sql_not(TruthValue.UNKNOWN)])
        assert result is TruthValue.UNKNOWN

    def test_classical_logic_holds_for_true_false(self):
        assert sql_or([TruthValue.TRUE, sql_not(TruthValue.TRUE)]) is TruthValue.TRUE
        assert sql_or([TruthValue.FALSE, sql_not(TruthValue.FALSE)]) is TruthValue.TRUE


# ── LiteralValueType strict typing ───────────────────────────────────

class TestLiteralValueType:
    def test_rejects_dict_value(self):
        with pytest.raises(ValidationError):
            LiteralExpression(value={"a": 1})

    def test_rejects_list_value(self):
        with pytest.raises(ValidationError):
            LiteralExpression(value=[1, 2, 3])

    def test_accepts_none(self):
        lit = LiteralExpression(value=None, value_type=LiteralValueType.NULL)
        assert lit.value is None

    def test_accepts_int(self):
        lit = LiteralExpression(value=42, value_type=LiteralValueType.INTEGER)
        assert lit.value == 42

    def test_accepts_bool(self):
        lit = LiteralExpression(value=True, value_type=LiteralValueType.BOOLEAN)
        assert lit.value is True

    def test_accepts_string(self):
        lit = LiteralExpression(value="hello", value_type=LiteralValueType.STRING)
        assert lit.value == "hello"

    def test_decimal_as_string(self):
        lit = LiteralExpression(value="123.456", value_type=LiteralValueType.DECIMAL)
        assert lit.value == "123.456"

    def test_date_as_string(self):
        lit = LiteralExpression(value="2026-01-15", value_type=LiteralValueType.DATE)
        assert lit.value == "2026-01-15"

    def test_timestamp_with_tz(self):
        lit = LiteralExpression(value="2026-01-15T10:30:00+05:30", value_type=LiteralValueType.TIMESTAMP_WITH_TIMEZONE)
        assert lit.value_type == LiteralValueType.TIMESTAMP_WITH_TIMEZONE
