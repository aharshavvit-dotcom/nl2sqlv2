from __future__ import annotations

import re
from typing import Any

import sqlglot
from sqlglot import exp


class SQLCanonicalizer:
    def canonicalize(self, sql: str, dialect: str = "sqlite") -> dict[str, Any]:
        warnings: list[str] = []
        try:
            ast = sqlglot.parse_one(sql, read=_dialect(dialect))
            canonical_sql = ast.sql(dialect=_dialect(dialect), pretty=False).rstrip(";")
            return {
                "canonical_sql": canonical_sql,
                "base_table": self._base_table(ast),
                "tables": sorted({table.name for table in ast.find_all(exp.Table) if table.name}),
                "columns": sorted({self._column_name(column) for column in ast.find_all(exp.Column) if column.name}),
                "joins": self._joins(ast),
                "filters": self._filters(ast, dialect),
                "group_by": self._group_by(ast, dialect),
                "order_by": self._order_by(ast, dialect),
                "aggregations": self._aggregations(ast, dialect),
                "limit": self._limit(ast),
                "parse_warnings": warnings,
            }
        except Exception as exc:
            warnings.append(f"sqlglot_parse_failed: {exc}")
            return self._fallback(sql, warnings)

    @staticmethod
    def _column_name(column: exp.Column) -> str:
        if column.table:
            return f"{column.table}.{column.name}"
        return column.name

    @staticmethod
    def _joins(ast: exp.Expression) -> list[dict[str, Any]]:
        joins = []
        for join in ast.find_all(exp.Join):
            table = join.this.name if isinstance(join.this, exp.Table) else join.this.sql(dialect="sqlite") if join.this else ""
            joins.append({"table": table, "kind": (join.args.get("kind") or "JOIN"), "on": join.args.get("on").sql(dialect="sqlite") if join.args.get("on") else ""})
        return joins

    @staticmethod
    def _filters(ast: exp.Expression, dialect: str) -> list[str]:
        where = ast.find(exp.Where)
        if where is None or where.this is None:
            return []
        return sorted(_split_and(where.this, dialect))

    @staticmethod
    def _group_by(ast: exp.Expression, dialect: str) -> list[str]:
        group = ast.args.get("group")
        return sorted(item.sql(dialect=_dialect(dialect)) for item in getattr(group, "expressions", []) or [])

    @staticmethod
    def _order_by(ast: exp.Expression, dialect: str) -> list[str]:
        order = ast.args.get("order")
        return [item.sql(dialect=_dialect(dialect)) for item in getattr(order, "expressions", []) or []]

    @staticmethod
    def _aggregations(ast: exp.Expression, dialect: str) -> list[str]:
        agg_types = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)
        return sorted(item.sql(dialect=_dialect(dialect)).lower() for item in ast.find_all(*agg_types))

    @staticmethod
    def _limit(ast: exp.Expression) -> int | None:
        limit = ast.args.get("limit")
        expression = getattr(limit, "expression", None)
        if expression is None:
            return None
        try:
            return int(expression.name)
        except Exception:
            return None

    @staticmethod
    def _base_table(ast: exp.Expression) -> str | None:
        from_expr = ast.args.get("from") or ast.args.get("from_")
        table = getattr(from_expr, "this", None)
        return table.name if isinstance(table, exp.Table) else None

    @staticmethod
    def _fallback(sql: str, warnings: list[str]) -> dict[str, Any]:
        text = " ".join(str(sql or "").strip().rstrip(";").split())
        tables = re.findall(r"\bFROM\s+([A-Za-z_][\w.\"]*)", text, flags=re.IGNORECASE)
        tables.extend(re.findall(r"\bJOIN\s+([A-Za-z_][\w.\"]*)", text, flags=re.IGNORECASE))
        columns = []
        select_match = re.search(r"\bSELECT\s+(.*?)\s+FROM\b", text, flags=re.IGNORECASE)
        if select_match:
            columns = [item.strip().strip('"') for item in select_match.group(1).split(",") if item.strip()]
        limit_match = re.search(r"\bLIMIT\s+(\d+)\b", text, flags=re.IGNORECASE)
        return {
            "canonical_sql": text,
            "base_table": tables[0].strip('"') if tables else None,
            "tables": sorted({item.strip('"') for item in tables}),
            "columns": sorted(columns),
            "joins": [{"table": item.strip('"'), "kind": "JOIN", "on": ""} for item in re.findall(r"\bJOIN\s+([A-Za-z_][\w.\"]*)", text, flags=re.IGNORECASE)],
            "filters": re.findall(r"\bWHERE\s+(.*?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|$)", text, flags=re.IGNORECASE),
            "group_by": [],
            "order_by": [],
            "aggregations": sorted(re.findall(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", text, flags=re.IGNORECASE)),
            "limit": int(limit_match.group(1)) if limit_match else None,
            "parse_warnings": warnings,
        }


def _split_and(expression: exp.Expression, dialect: str) -> list[str]:
    if isinstance(expression, exp.And):
        return [*_split_and(expression.left, dialect), *_split_and(expression.right, dialect)]
    return [expression.sql(dialect=_dialect(dialect))]


def _dialect(dialect: str) -> str:
    normalized = (dialect or "sqlite").lower()
    return "postgres" if normalized == "postgresql" else normalized
