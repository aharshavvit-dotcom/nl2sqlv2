"""Full QueryIR v2 query-level renderer.

Renders complete QueryNode trees to SQL including:
SELECT, FROM, JOIN, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT/OFFSET,
CTEs, set operations, and subquery FROM items.

This is the primary renderer class — exposed publicly through
`ir/query_ir_v2_boolean_renderer.py` for backward compatibility.
"""

from __future__ import annotations

from typing import Any

from ir.ir_to_sql_renderer import IRToSQLRenderer, quote_identifier
from ir.query_ir_v2_models import (
    BooleanPredicate,
    CTEDefinition,
    FromItem,
    JoinNode,
    NotPredicate,
    OrderByItem,
    QueryNode,
    SelectItem,
    SetOperationNode,
)

from .expressions import render_expression
from .predicates import render_predicate


class QueryIRV2RenderingError(ValueError):
    """Raised when a QueryIR v2 node cannot be rendered to SQL."""

    def __init__(self, code: str, message: str, *, capability: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.capability = capability


class QueryIRV2NativeRenderer:
    """Full native QueryIR v2 renderer.

    Renders QueryNode trees to SQL. Supports all v2 constructs:
    SELECT, FROM (table/subquery), JOIN (table/subquery), WHERE, GROUP BY,
    HAVING, CASE, window expressions, ORDER BY, LIMIT/OFFSET, CTEs, set operations.
    """

    def __init__(
        self,
        *,
        enable_or_rendering: bool = False,
        max_limit: int = 1000,
    ) -> None:
        self.enable_or_rendering = enable_or_rendering
        self.max_limit = max_limit

    def render(self, query: QueryNode | dict[str, Any], dialect: str | None = None) -> str:
        """Render a QueryNode to SQL."""
        query_node = query if isinstance(query, QueryNode) else QueryNode.model_validate(query)
        resolved_dialect = dialect or query_node.dialect or "sqlite"

        # OR gate check
        if (
            query_node.where is not None
            and _contains_or(query_node.where)
            and not self.enable_or_rendering
        ):
            raise QueryIRV2RenderingError(
                "v2_or_rendering_disabled",
                "QueryIR v2 OR rendering is disabled outside tests/diagnostics.",
                capability="OR_FILTER",
            )

        parts: list[str] = []

        # CTEs
        cte_clause = self._render_ctes(query_node, resolved_dialect)
        if cte_clause:
            parts.append(cte_clause)

        # Main query body
        parts.append(self._render_select(query_node, resolved_dialect))
        from_clause = self._render_from(query_node, resolved_dialect)
        if from_clause:
            parts.append(from_clause)
        joins_clause = self._render_joins(query_node, resolved_dialect)
        if joins_clause:
            parts.append(joins_clause)
        where_clause = self._render_where(query_node, resolved_dialect)
        if where_clause:
            parts.append(where_clause)
        group_clause = self._render_group_by(query_node, resolved_dialect)
        if group_clause:
            parts.append(group_clause)
        having_clause = self._render_having(query_node, resolved_dialect)
        if having_clause:
            parts.append(having_clause)

        # Set operations BEFORE ORDER BY (ORDER BY applies to combined result)
        set_ops = self._render_set_operations(query_node, resolved_dialect)
        if set_ops:
            parts.append(set_ops)

        order_clause = self._render_order_by(query_node, resolved_dialect)
        if order_clause:
            parts.append(order_clause)
        limit_clause = self._render_limit(query_node)
        if limit_clause:
            parts.append(limit_clause)
        offset_clause = self._render_offset(query_node)
        if offset_clause:
            parts.append(offset_clause)

        return IRToSQLRenderer.clean_sql("\n".join(parts))

    # ── Backward-compatible public API ────────────────────────────────

    def render_select(self, query: QueryNode, dialect: str) -> str:
        return self._render_select(query, dialect)

    def render_from(self, query: QueryNode) -> str:
        return self._render_from(query, "sqlite")

    def render_joins(self, query: QueryNode, dialect: str) -> str:
        return self._render_joins(query, dialect)

    def render_where(self, query: QueryNode, dialect: str) -> str:
        return self._render_where(query, dialect)

    def render_order_by(self, query: QueryNode, dialect: str) -> str:
        return self._render_order_by(query, dialect)

    def render_limit(self, query: QueryNode) -> str:
        return self._render_limit(query)

    def render_predicate(
        self,
        predicate: Any,
        *,
        dialect: str = "sqlite",
        parent_operator: str | None = None,
        inside_not: bool = False,
    ) -> str:
        return render_predicate(
            predicate,
            dialect=dialect,
            parent_operator=parent_operator,
            inside_not=inside_not,
            renderer=self,
            enable_or_rendering=self.enable_or_rendering,
        )

    def render_expression(self, expression: Any, dialect: str = "sqlite") -> str:
        return render_expression(expression, dialect, renderer=self)

    # ── Internal rendering methods ────────────────────────────────────

    def _render_ctes(self, query: QueryNode, dialect: str) -> str:
        if not query.ctes:
            return ""
        cte_defs: list[str] = []
        for cte in query.ctes:
            inner_sql = self.render(cte.query, dialect)
            if cte.columns:
                cols = ", ".join(quote_identifier(c) for c in cte.columns)
                cte_defs.append(f"{quote_identifier(cte.name)} ({cols}) AS (\n{inner_sql}\n)")
            else:
                cte_defs.append(f"{quote_identifier(cte.name)} AS (\n{inner_sql}\n)")
        return "WITH " + ",\n".join(cte_defs)

    def _render_select(self, query: QueryNode, dialect: str) -> str:
        if not query.select_items:
            return "SELECT\n  *"
        rendered = [self._render_select_item(item, dialect) for item in query.select_items]
        return "SELECT\n  " + ",\n  ".join(rendered)

    def _render_from(self, query: QueryNode, dialect: str) -> str:
        if query.from_item is None:
            return ""
        return "FROM " + self._render_from_item(query.from_item, dialect)

    def _render_from_item(self, from_item: FromItem, dialect: str) -> str:
        if from_item.from_type == "TABLE" and from_item.table:
            base = quote_identifier(from_item.table)
            if from_item.alias and from_item.alias != from_item.table:
                return f"{base} AS {quote_identifier(from_item.alias)}"
            return base
        if from_item.from_type == "SUBQUERY" and from_item.query is not None:
            inner_sql = self.render(from_item.query, dialect)
            alias = from_item.alias or "_subquery"
            return f"(\n{inner_sql}\n) AS {quote_identifier(alias)}"
        return ""

    def _render_joins(self, query: QueryNode, dialect: str) -> str:
        if not query.joins:
            return ""
        lines = [self._render_join(join, dialect) for join in sorted(query.joins, key=lambda j: j.path_order)]
        return "\n".join(line for line in lines if line)

    def _render_join(self, join: JoinNode, dialect: str) -> str:
        right_sql = self._render_from_item(join.right, dialect)
        line = f"{join.join_type} JOIN {right_sql}"
        if join.on is not None:
            on_clause = self.render_predicate(join.on, dialect=dialect)
            line += "\n  ON " + on_clause
        return line

    def _render_where(self, query: QueryNode, dialect: str) -> str:
        predicate = query.where
        if predicate is None and query.predicates:
            if len(query.predicates) > 1:
                predicate = BooleanPredicate(operator="AND", operands=query.predicates)
            else:
                predicate = query.predicates[0]
        if predicate is None:
            return ""
        return "WHERE " + self.render_predicate(predicate, dialect=dialect)

    def _render_group_by(self, query: QueryNode, dialect: str) -> str:
        if not query.group_by:
            return ""
        exprs = ", ".join(
            render_expression(expr, dialect, renderer=self) for expr in query.group_by
        )
        return f"GROUP BY {exprs}"

    def _render_having(self, query: QueryNode, dialect: str) -> str:
        if query.having is None:
            return ""
        return "HAVING " + self.render_predicate(query.having, dialect=dialect)

    def _render_set_operations(self, query: QueryNode, dialect: str) -> str:
        if not query.set_operations:
            return ""
        parts: list[str] = []
        for set_op in query.set_operations:
            op_keyword = set_op.operation.replace("_", " ")
            inner_sql = self.render(set_op.query, dialect)
            parts.append(f"{op_keyword}\n{inner_sql}")
        return "\n".join(parts)

    def _render_order_by(self, query: QueryNode, dialect: str) -> str:
        if not query.order_by:
            return ""
        items = ", ".join(self._render_order_by_item(item, dialect) for item in query.order_by)
        return f"ORDER BY {items}"

    def _render_limit(self, query: QueryNode) -> str:
        if query.limit is None:
            return ""
        return f"LIMIT {min(max(int(query.limit), 1), self.max_limit)}"

    def _render_offset(self, query: QueryNode) -> str:
        if query.offset is None or query.offset <= 0:
            return ""
        return f"OFFSET {int(query.offset)}"

    def _render_select_item(self, item: SelectItem, dialect: str) -> str:
        expr = render_expression(item.expression, dialect, renderer=self)
        return f"{expr} AS {quote_identifier(item.alias)}" if item.alias else expr

    def _render_order_by_item(self, item: OrderByItem, dialect: str) -> str:
        expr = quote_identifier(item.alias) if item.alias else render_expression(item.expression, dialect, renderer=self)
        return f"{expr} {item.direction}"


def _contains_or(predicate: Any) -> bool:
    """Check if a predicate tree contains OR operators."""
    if isinstance(predicate, BooleanPredicate):
        return predicate.operator == "OR" or any(_contains_or(item) for item in predicate.operands)
    if isinstance(predicate, NotPredicate):
        return _contains_or(predicate.operand)
    return False
