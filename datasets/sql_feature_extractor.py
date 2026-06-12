from __future__ import annotations

from typing import Any

import sqlglot
from sqlglot import exp


class SQLFeatureExtractor:
    def extract(self, sql: str, dialect: str | None = None) -> dict[str, Any]:
        tree, parse_error = self._parse(sql, dialect=dialect)
        if tree is None:
            return {
                "statement_type": None,
                "selected_columns": [],
                "aggregations": [],
                "aggregation_expressions": [],
                "tables": [],
                "joins": [],
                "where_conditions": [],
                "group_by": [],
                "order_by": [],
                "limit": None,
                "has_nested_query": False,
                "has_set_operation": False,
                "has_having": False,
                "has_window_function": False,
                "complexity": "unknown",
                "parse_error": parse_error,
            }

        selected_columns = [item.sql() for item in getattr(tree, "expressions", [])]
        aggregations = self._aggregations(tree)
        group_by = self._group_by(tree)
        order_by = self._order_by(tree)
        joins = self._joins(tree)
        where_conditions = self._where_conditions(tree)
        has_nested = any(True for _ in tree.find_all(exp.Subquery))
        has_set_operation = isinstance(tree, (exp.Union, exp.Intersect, exp.Except)) or any(
            True for _ in tree.find_all(exp.Union, exp.Intersect, exp.Except)
        )
        has_having = tree.find(exp.Having) is not None
        has_window = any(True for _ in tree.find_all(exp.Window))

        return {
            "statement_type": self._statement_type(tree),
            "selected_columns": selected_columns,
            "aggregations": [item["function"] for item in aggregations],
            "aggregation_expressions": aggregations,
            "tables": [table.name for table in tree.find_all(exp.Table) if table.name],
            "joins": joins,
            "where_conditions": where_conditions,
            "group_by": group_by,
            "order_by": order_by,
            "limit": self._limit(tree),
            "has_nested_query": has_nested,
            "has_set_operation": has_set_operation,
            "has_having": has_having,
            "has_window_function": has_window,
            "complexity": self._complexity(
                joins=joins,
                where_conditions=where_conditions,
                group_by=group_by,
                order_by=order_by,
                has_nested=has_nested,
                has_set_operation=has_set_operation,
                has_having=has_having,
                has_window=has_window,
            ),
        }

    @staticmethod
    def _parse(sql: str, dialect: str | None) -> tuple[exp.Expression | None, str | None]:
        dialects = [dialect] if dialect else [None, "sqlite"]
        last_error: str | None = None
        for candidate in dialects:
            try:
                return sqlglot.parse_one(sql, read=candidate), None
            except Exception as exc:
                last_error = str(exc)
        return None, last_error

    @staticmethod
    def _statement_type(tree: exp.Expression) -> str:
        if isinstance(tree, exp.Select):
            return "SELECT"
        if isinstance(tree, (exp.Union, exp.Intersect, exp.Except)):
            return "SELECT"
        return tree.key.upper() if getattr(tree, "key", None) else type(tree).__name__.upper()

    @staticmethod
    def _aggregations(tree: exp.Expression) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        for agg_type in [exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max]:
            for item in tree.find_all(agg_type):
                column_expr = item.this.sql() if item.this is not None else "*"
                results.append(
                    {
                        "function": item.key.upper(),
                        "column": column_expr,
                        "expression": item.sql(),
                    }
                )
        return results

    @staticmethod
    def _joins(tree: exp.Expression) -> list[dict[str, str | None]]:
        joins: list[dict[str, str | None]] = []
        for join in tree.find_all(exp.Join):
            joins.append(
                {
                    "table": join.this.sql() if join.this is not None else None,
                    "on": join.args.get("on").sql() if join.args.get("on") is not None else None,
                    "kind": str(join.args.get("kind") or "").upper() or None,
                }
            )
        return joins

    @staticmethod
    def _where_conditions(tree: exp.Expression) -> list[dict[str, str]]:
        where = tree.find(exp.Where)
        if where is None or where.this is None:
            return []
        conditions: list[dict[str, str]] = []
        for item in where.find_all(exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NEQ, exp.Like, exp.In):
            conditions.append({"expression": item.sql(), "operator": item.key.upper()})
        if not conditions:
            conditions.append({"expression": where.this.sql(), "operator": "WHERE"})
        return conditions

    @staticmethod
    def _group_by(tree: exp.Expression) -> list[str]:
        group = tree.args.get("group")
        if group is None:
            return []
        return [item.sql() for item in group.expressions]

    @staticmethod
    def _order_by(tree: exp.Expression) -> list[dict[str, Any]]:
        order = tree.args.get("order")
        if order is None:
            return []
        values: list[dict[str, Any]] = []
        for item in order.expressions:
            values.append(
                {
                    "expression": item.this.sql() if item.this is not None else item.sql(),
                    "desc": bool(item.args.get("desc")),
                }
            )
        return values

    @staticmethod
    def _limit(tree: exp.Expression) -> int | None:
        limit = tree.args.get("limit")
        if limit is None:
            return None
        expression = limit.expression
        if expression is None:
            return None
        try:
            return int(expression.name)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _complexity(
        joins: list[dict[str, Any]],
        where_conditions: list[dict[str, Any]],
        group_by: list[str],
        order_by: list[dict[str, Any]],
        has_nested: bool,
        has_set_operation: bool,
        has_having: bool,
        has_window: bool,
    ) -> str:
        if has_nested or has_set_operation or has_having or has_window:
            return "complex"
        if joins or group_by or order_by:
            return "medium"
        if where_conditions:
            return "simple_filter"
        return "simple"
