"""Expression rendering for QueryIR v2.

Renders Expression nodes to SQL fragments. Handles all expression types
including CASE, window, subquery, and binary/unary operations.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ir.ir_to_sql_renderer import IRToSQLRenderer, quote_identifier
from ir.query_ir_v2_models import (
    AggregationExpression,
    BinaryOperationExpression,
    BooleanOperationExpression,
    CaseExpression,
    ColumnExpression,
    FrameBound,
    FrameBoundType,
    FunctionExpression,
    LiteralExpression,
    SubqueryExpression,
    UnaryOperationExpression,
    WindowExpression,
    WindowSpecification,
)

if TYPE_CHECKING:
    from .queries import QueryIRV2NativeRenderer


def render_expression(
    expression: Any,
    dialect: str = "sqlite",
    *,
    renderer: QueryIRV2NativeRenderer | None = None,
) -> str:
    """Render a QueryIR v2 Expression node to a SQL string."""
    if isinstance(expression, ColumnExpression):
        return _render_column(expression)

    if isinstance(expression, LiteralExpression):
        return IRToSQLRenderer.render_literal(expression.value)

    if isinstance(expression, BinaryOperationExpression):
        left = render_expression(expression.left, dialect, renderer=renderer)
        right = render_expression(expression.right, dialect, renderer=renderer)
        return f"{left} {expression.operator} {right}"

    if isinstance(expression, UnaryOperationExpression):
        operand = render_expression(expression.operand, dialect, renderer=renderer)
        return f"{expression.operator}({operand})"

    if isinstance(expression, AggregationExpression):
        return _render_aggregation(expression, dialect, renderer)

    if isinstance(expression, FunctionExpression):
        return _render_function(expression, dialect, renderer)

    if isinstance(expression, BooleanOperationExpression):
        rendered = [
            render_expression(operand, dialect, renderer=renderer)
            for operand in expression.operands
        ]
        return f" {expression.operator} ".join(rendered)

    if isinstance(expression, CaseExpression):
        return _render_case(expression, dialect, renderer)

    if isinstance(expression, SubqueryExpression):
        if renderer is None:
            from .queries import QueryIRV2NativeRenderer
            renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
        inner_sql = renderer.render(expression.query, dialect)
        return f"({inner_sql})"

    if isinstance(expression, WindowExpression):
        return _render_window(expression, dialect, renderer)

    from .queries import QueryIRV2RenderingError
    raise QueryIRV2RenderingError(
        "unsupported_v2_rendering_capability",
        f"Unsupported expression node: {type(expression).__name__}",
        capability=type(expression).__name__,
    )


def _render_column(col: ColumnExpression) -> str:
    if col.column == "*":
        return f"{quote_identifier(col.table)}.*" if col.table else "*"
    if col.table:
        return f"{quote_identifier(col.table)}.{quote_identifier(col.column)}"
    return quote_identifier(col.column)


def _render_aggregation(
    agg: AggregationExpression,
    dialect: str,
    renderer: Any,
) -> str:
    if agg.argument is None:
        argument = "*"
    else:
        argument = render_expression(agg.argument, dialect, renderer=renderer)
    distinct_prefix = "DISTINCT " if agg.distinct else ""
    return f"{agg.function.upper()}({distinct_prefix}{argument})"


def _render_function(
    func: FunctionExpression,
    dialect: str,
    renderer: Any,
) -> str:
    # Special handling for DATE_GRAIN (backward compatibility)
    if func.name.upper() == "DATE_GRAIN" and func.arguments:
        grain = "month"
        if len(func.arguments) > 1 and isinstance(func.arguments[1], LiteralExpression):
            grain = str(func.arguments[1].value or "month")
        return IRToSQLRenderer.render_date_grain(
            render_expression(func.arguments[0], dialect, renderer=renderer),
            grain,
            dialect=dialect,
        )
    args = ", ".join(
        render_expression(arg, dialect, renderer=renderer)
        for arg in func.arguments
    )
    return f"{func.name.upper()}({args})"


def _render_case(
    case: CaseExpression,
    dialect: str,
    renderer: Any,
) -> str:
    from .predicates import render_predicate

    parts = ["CASE"]
    for case_when in case.cases:
        when_clause = render_predicate(
            case_when.when,
            dialect=dialect,
            renderer=renderer,
        )
        then_clause = render_expression(case_when.then, dialect, renderer=renderer)
        parts.append(f"WHEN {when_clause} THEN {then_clause}")
    if case.else_expression is not None:
        parts.append(f"ELSE {render_expression(case.else_expression, dialect, renderer=renderer)}")
    parts.append("END")
    return " ".join(parts)


def _render_window(
    window: WindowExpression,
    dialect: str,
    renderer: Any,
) -> str:
    inner = render_expression(window.expression, dialect, renderer=renderer)
    spec = _render_window_spec(window.window, dialect, renderer)
    return f"{inner} OVER ({spec})"


def _render_window_spec(
    spec: WindowSpecification,
    dialect: str,
    renderer: Any,
) -> str:
    parts: list[str] = []

    if spec.partition_by:
        partition_exprs = ", ".join(
            render_expression(expr, dialect, renderer=renderer)
            for expr in spec.partition_by
        )
        parts.append(f"PARTITION BY {partition_exprs}")

    if spec.order_by:
        order_items = ", ".join(
            _render_order_by_item(item, dialect, renderer)
            for item in spec.order_by
        )
        parts.append(f"ORDER BY {order_items}")

    frame_clause = _render_frame(spec)
    if frame_clause:
        parts.append(frame_clause)

    return " ".join(parts)


def _render_order_by_item(item: Any, dialect: str, renderer: Any) -> str:
    expr = render_expression(item.expression, dialect, renderer=renderer)
    return f"{expr} {item.direction}"


def _render_frame(spec: WindowSpecification) -> str:
    if spec.frame_type is None or spec.frame_start is None:
        return ""

    start = _render_frame_bound(spec.frame_start)
    if spec.frame_end is not None:
        end = _render_frame_bound(spec.frame_end)
        return f"{spec.frame_type} BETWEEN {start} AND {end}"
    return f"{spec.frame_type} {start}"


def _render_frame_bound(bound: FrameBound) -> str:
    if bound.bound_type == FrameBoundType.UNBOUNDED_PRECEDING:
        return "UNBOUNDED PRECEDING"
    if bound.bound_type == FrameBoundType.N_PRECEDING:
        return f"{bound.offset} PRECEDING"
    if bound.bound_type == FrameBoundType.CURRENT_ROW:
        return "CURRENT ROW"
    if bound.bound_type == FrameBoundType.N_FOLLOWING:
        return f"{bound.offset} FOLLOWING"
    if bound.bound_type == FrameBoundType.UNBOUNDED_FOLLOWING:
        return "UNBOUNDED FOLLOWING"
    return ""
