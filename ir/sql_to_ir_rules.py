from __future__ import annotations

import re
from typing import Any

from sqlglot import exp


AGGREGATE_TYPES = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)
SET_OPERATION_TYPES = (exp.Union, exp.Intersect, exp.Except)
DATE_GRAIN_RE = re.compile(
    r"strftime\(\s*['\"](?P<format>%Y(?:-%m)?)['\"]\s*,\s*(?P<table>[A-Za-z_][\w]*)\.(?P<column>[A-Za-z_][\w]*)\s*\)",
    re.IGNORECASE,
)


def is_select_query(ast: exp.Expression) -> bool:
    return isinstance(ast, exp.Select)


def has_nested_query(ast: exp.Expression) -> bool:
    if any(True for _ in ast.find_all(exp.Subquery)):
        return True
    selects = list(ast.find_all(exp.Select))
    return len(selects) > (1 if isinstance(ast, exp.Select) else 0)


def has_set_operation(ast: exp.Expression) -> bool:
    return isinstance(ast, SET_OPERATION_TYPES) or any(True for _ in ast.find_all(*SET_OPERATION_TYPES))


def has_window_function(ast: exp.Expression) -> bool:
    return any(True for _ in ast.find_all(exp.Window))


def extract_tables(ast: exp.Expression) -> list[str]:
    tables: list[str] = []
    for table in ast.find_all(exp.Table):
        name = normalize_table_ref(table)
        if name and name not in tables:
            tables.append(name)
    return tables


def extract_select_expressions(ast: exp.Expression) -> list[dict[str, Any]]:
    expressions: list[dict[str, Any]] = []
    for item in getattr(ast, "expressions", []) or []:
        inner = unwrap_alias(item)
        date_grain = date_grain_from_expression(inner)
        expressions.append(
            {
                "sql": expression_sql(inner),
                "alias": alias_for(item),
                "is_aggregation": contains_aggregation(inner),
                "is_column": isinstance(inner, exp.Column),
                "is_star": isinstance(inner, exp.Star),
                "column": normalize_column_ref(inner) if isinstance(inner, exp.Column) else None,
                "date_grain": date_grain,
            }
        )
    return expressions


def extract_aggregations(ast: exp.Expression) -> list[dict[str, Any]]:
    aggregations: list[dict[str, Any]] = []
    for item in getattr(ast, "expressions", []) or []:
        inner = unwrap_alias(item)
        for aggregation in iter_aggregations(inner):
            argument = aggregation.this
            aggregations.append(
                {
                    "function": aggregation.key.upper(),
                    "alias": alias_for(item),
                    "argument_sql": "*" if argument is None else expression_sql(argument),
                    "expression": expression_sql(aggregation),
                    "argument": argument,
                    "node": aggregation,
                }
            )
    return aggregations


def extract_group_by(ast: exp.Expression) -> list[str]:
    group = ast.args.get("group")
    if group is None:
        return []
    return [expression_sql(item) for item in group.expressions]


def extract_order_by(ast: exp.Expression) -> list[dict[str, Any]]:
    order = ast.args.get("order")
    if order is None:
        return []
    values: list[dict[str, Any]] = []
    for item in order.expressions:
        target = item.this if item.this is not None else item
        values.append(
            {
                "expression": expression_sql(target),
                "alias": target.name if isinstance(target, exp.Column) and not target.table else None,
                "direction": "DESC" if bool(item.args.get("desc")) else "ASC",
                "desc": bool(item.args.get("desc")),
            }
        )
    return values


def extract_limit(ast: exp.Expression) -> int | None:
    limit = ast.args.get("limit")
    if limit is None:
        return None
    expression = getattr(limit, "expression", None)
    if expression is None:
        return None
    try:
        return int(expression.name)
    except (TypeError, ValueError):
        return None


def extract_where_filters(ast: exp.Expression) -> list[dict[str, Any]]:
    where = ast.find(exp.Where)
    if where is None or where.this is None:
        return []
    filters: list[dict[str, Any]] = []
    for condition in flatten_and(where.this):
        parsed = normalize_filter_condition(condition)
        if parsed:
            filters.append(parsed)
    return filters


def extract_joins(ast: exp.Expression) -> list[dict[str, Any]]:
    joins: list[dict[str, Any]] = []
    for index, join in enumerate(ast.args.get("joins") or [], start=1):
        table = normalize_table_ref(join.this) if join.this is not None else None
        on = join.args.get("on")
        left = right = None
        if isinstance(on, exp.EQ):
            left_ref = normalize_column_ref(on.this)
            right_ref = normalize_column_ref(on.expression)
            if left_ref and right_ref:
                left, right = left_ref, right_ref
        joins.append(
            {
                "table": table,
                "join_type": str(join.args.get("kind") or "INNER").upper(),
                "condition": expression_sql(on) if on is not None else None,
                "left": left,
                "right": right,
                "path_order": index,
            }
        )
    return joins


def detect_intent_from_ast(ast: exp.Expression, features: dict[str, Any]) -> str:
    aggregations = {str(item.get("function", "")).upper() for item in features.get("aggregations", [])}
    group_by = features.get("group_by") or []
    order_by = features.get("order_by") or []
    where_filters = features.get("where_filters") or []
    limit = features.get("limit")

    if any(date_grain_from_sql(str(item)) for item in group_by):
        return "trend_by_date"
    if aggregations == {"COUNT"} and not group_by:
        return "count_records"
    if "COUNT" in aggregations and group_by:
        return "count_by_dimension"
    if aggregations and group_by:
        if order_by and limit is not None and limit <= 20:
            return "top_n_metric_by_dimension" if order_by[0].get("direction") == "DESC" else "bottom_n_metric_by_dimension"
        return "metric_by_dimension"
    if aggregations:
        return "metric_summary"
    if where_filters:
        return "simple_filter"
    return "show_records"


def normalize_column_ref(expr: exp.Expression | None) -> dict[str, str | None] | None:
    if not isinstance(expr, exp.Column):
        return None
    table = str(expr.table) if expr.table else None
    column = str(expr.name) if expr.name else None
    expression = f"{table}.{column}" if table and column else column
    return {"table": table, "column": column, "expression": expression}


def normalize_table_ref(expr: exp.Expression | None) -> str:
    if isinstance(expr, exp.Table):
        return str(expr.name)
    if expr is None:
        return ""
    return str(getattr(expr, "name", None) or expr.sql(dialect="sqlite"))


def unwrap_alias(expr: exp.Expression) -> exp.Expression:
    return expr.this if isinstance(expr, exp.Alias) and expr.this is not None else expr


def alias_for(expr: exp.Expression) -> str | None:
    alias = getattr(expr, "alias", None)
    return str(alias) if alias else None


def expression_sql(expr: exp.Expression | None) -> str:
    if expr is None:
        return ""
    return re.sub(r"\s+", " ", expr.sql(dialect="sqlite")).strip()


def contains_aggregation(expr: exp.Expression) -> bool:
    return any(True for _ in iter_aggregations(expr))


def iter_aggregations(expr: exp.Expression) -> list[exp.Expression]:
    aggregations: list[exp.Expression] = []
    if isinstance(expr, AGGREGATE_TYPES):
        aggregations.append(expr)
    aggregations.extend(item for item in expr.find_all(*AGGREGATE_TYPES) if item is not expr)
    return aggregations


def date_grain_from_expression(expr: exp.Expression) -> dict[str, str] | None:
    return date_grain_from_sql(expression_sql(expr))


def date_grain_from_sql(sql: str) -> dict[str, str] | None:
    match = DATE_GRAIN_RE.search(sql)
    if not match:
        return None
    fmt = match.group("format")
    grain = "month" if fmt == "%Y-%m" else "year"
    table = match.group("table")
    column = match.group("column")
    return {
        "date_grain": grain,
        "date_table": table,
        "date_column": column,
        "date_expression": f"{table}.{column}",
    }


def flatten_and(expr: exp.Expression) -> list[exp.Expression]:
    if isinstance(expr, exp.And):
        return [*flatten_and(expr.this), *flatten_and(expr.expression)]
    return [expr]


def normalize_filter_condition(condition: exp.Expression) -> dict[str, Any] | None:
    operator_map = {
        exp.EQ: "equals",
        exp.NEQ: "not_equals",
        exp.GT: "greater_than",
        exp.GTE: "greater_equal",
        exp.LT: "less_than",
        exp.LTE: "less_equal",
        exp.Like: "contains",
    }
    if isinstance(condition, exp.In):
        left = normalize_column_ref(condition.this)
        values = [literal_value(item) for item in condition.expressions]
        if left is None or any(value is None for value in values):
            return None
        return {
            "expression": expression_sql(condition),
            "operator": "in",
            "left": left,
            "value": values,
            "value_sql": [expression_sql(item) for item in condition.expressions],
        }

    for klass, operator in operator_map.items():
        if not isinstance(condition, klass):
            continue
        left = normalize_column_ref(condition.this)
        right = condition.expression
        if left is None and isinstance(right, exp.Column):
            left = normalize_column_ref(right)
            right = condition.this
            operator = reverse_operator(operator)
        value = literal_value(right)
        if left is None or value is None:
            return None
        return {
            "expression": expression_sql(condition),
            "operator": operator,
            "left": left,
            "value": value,
            "value_sql": expression_sql(right),
        }
    return None


def literal_value(expr: exp.Expression | None) -> Any:
    if expr is None:
        return None
    if isinstance(expr, exp.Literal):
        raw = expr.this
        if expr.is_string:
            return str(raw)
        text = str(raw)
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text
    if isinstance(expr, exp.Boolean):
        return bool(expr.this)
    if isinstance(expr, exp.Null):
        return None
    return None


def reverse_operator(operator: str) -> str:
    return {
        "greater_than": "less_than",
        "greater_equal": "less_equal",
        "less_than": "greater_than",
        "less_equal": "greater_equal",
    }.get(operator, operator)

