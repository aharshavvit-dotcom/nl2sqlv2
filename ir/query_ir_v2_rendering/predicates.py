"""Predicate rendering for QueryIR v2.

Renders Predicate nodes to SQL WHERE/HAVING clause fragments.
Handles all predicate types including IN SUBQUERY, EXISTS, and boolean combinations.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ir.query_ir_v2_models import (
    BetweenPredicate,
    BooleanPredicate,
    ComparisonPredicate,
    ExistsPredicate,
    InLiteralPredicate,
    InSubqueryPredicate,
    LiteralExpression,
    NotPredicate,
    NullPredicate,
)

from .expressions import render_expression

if TYPE_CHECKING:
    from .queries import QueryIRV2NativeRenderer


def render_predicate(
    predicate: Any,
    *,
    dialect: str = "sqlite",
    parent_operator: str | None = None,
    inside_not: bool = False,
    renderer: Any | None = None,
    enable_or_rendering: bool = True,
) -> str:
    """Render a QueryIR v2 Predicate node to a SQL string."""

    if isinstance(predicate, ComparisonPredicate):
        return _render_comparison(predicate, dialect, renderer)

    if isinstance(predicate, InLiteralPredicate):
        return _render_in_literal(predicate, dialect, renderer)

    if isinstance(predicate, InSubqueryPredicate):
        return _render_in_subquery(predicate, dialect, renderer)

    if isinstance(predicate, ExistsPredicate):
        return _render_exists(predicate, dialect, renderer)

    if isinstance(predicate, BetweenPredicate):
        return _render_between(predicate, dialect, renderer)

    if isinstance(predicate, NullPredicate):
        return _render_null(predicate, dialect, renderer)

    if isinstance(predicate, BooleanPredicate):
        return _render_boolean(
            predicate,
            dialect,
            parent_operator=parent_operator,
            inside_not=inside_not,
            renderer=renderer,
            enable_or_rendering=enable_or_rendering,
        )

    if isinstance(predicate, NotPredicate):
        inner = render_predicate(
            predicate.operand,
            dialect=dialect,
            inside_not=True,
            renderer=renderer,
            enable_or_rendering=enable_or_rendering,
        )
        return f"NOT ({inner})"

    from .queries import QueryIRV2RenderingError
    raise QueryIRV2RenderingError(
        "unsupported_v2_rendering_capability",
        f"Unsupported predicate node: {type(predicate).__name__}",
        capability=type(predicate).__name__,
    )


def _render_comparison(pred: ComparisonPredicate, dialect: str, renderer: Any) -> str:
    from .queries import QueryIRV2RenderingError

    if isinstance(pred.right, LiteralExpression) and pred.right.value is None:
        raise QueryIRV2RenderingError(
            "unsupported_v2_rendering_capability",
            "Use NULL_PREDICATE instead of comparison to NULL.",
            capability="NULL_COMPARISON",
        )
    left = render_expression(pred.left, dialect, renderer=renderer)
    right = render_expression(pred.right, dialect, renderer=renderer)
    return f"{left} {pred.operator} {right}"


def _render_in_literal(pred: InLiteralPredicate, dialect: str, renderer: Any) -> str:
    values = ", ".join(
        render_expression(item, dialect, renderer=renderer) for item in pred.values
    )
    operator = "NOT IN" if pred.negated else "IN"
    expr = render_expression(pred.expression, dialect, renderer=renderer)
    return f"{expr} {operator} ({values})"


def _render_in_subquery(pred: InSubqueryPredicate, dialect: str, renderer: Any) -> str:
    if renderer is None:
        from .queries import QueryIRV2NativeRenderer
        renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
    expr = render_expression(pred.expression, dialect, renderer=renderer)
    inner_sql = renderer.render(pred.query, dialect)
    operator = "NOT IN" if pred.negated else "IN"
    return f"{expr} {operator} ({inner_sql})"


def _render_exists(pred: ExistsPredicate, dialect: str, renderer: Any) -> str:
    if renderer is None:
        from .queries import QueryIRV2NativeRenderer
        renderer = QueryIRV2NativeRenderer(enable_or_rendering=True)
    inner_sql = renderer.render(pred.query, dialect)
    prefix = "NOT EXISTS" if pred.negated else "EXISTS"
    return f"{prefix} ({inner_sql})"


def _render_between(pred: BetweenPredicate, dialect: str, renderer: Any) -> str:
    expr = render_expression(pred.expression, dialect, renderer=renderer)
    lower = render_expression(pred.lower, dialect, renderer=renderer)
    upper = render_expression(pred.upper, dialect, renderer=renderer)
    operator = "NOT BETWEEN" if pred.negated else "BETWEEN"
    return f"{expr} {operator} {lower} AND {upper}"


def _render_null(pred: NullPredicate, dialect: str, renderer: Any) -> str:
    expr = render_expression(pred.expression, dialect, renderer=renderer)
    operator = "IS NOT NULL" if pred.negated else "IS NULL"
    return f"{expr} {operator}"


def _render_boolean(
    pred: BooleanPredicate,
    dialect: str,
    *,
    parent_operator: str | None,
    inside_not: bool,
    renderer: Any,
    enable_or_rendering: bool,
) -> str:
    from .queries import QueryIRV2RenderingError

    if pred.operator == "OR" and not enable_or_rendering:
        raise QueryIRV2RenderingError(
            "v2_or_rendering_disabled",
            "QueryIR v2 OR rendering is disabled outside tests/diagnostics.",
            capability="OR_FILTER",
        )
    rendered = [
        render_predicate(
            item,
            dialect=dialect,
            parent_operator=pred.operator,
            renderer=renderer,
            enable_or_rendering=enable_or_rendering,
        )
        for item in pred.operands
    ]
    text = f" {pred.operator} ".join(rendered)
    if parent_operator is not None or inside_not:
        return f"({text})"
    return text
