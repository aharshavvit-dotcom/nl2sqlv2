from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any

import sqlglot
from sqlglot import exp

from nl2sql_v1.schema import SchemaGraph


BLOCKED_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "MERGE",
    "TRUNCATE",
    "CREATE",
    "REPLACE",
    "GRANT",
    "REVOKE",
    "CALL",
    "EXEC",
    "EXECUTE",
    "COPY",
    "UNLOAD",
}
SENSITIVE_MARKERS = ("email", "phone", "password", "token", "secret", "ssn", "address", "dob", "birth_date", "credit_card", "api_key", "auth")
DANGEROUS_FUNCTIONS = {"load_extension", "readfile", "writefile"}

POLICY_FAILURE_TYPES = (
    "select_star_blocked",
    "limit_policy_failed",
    "non_select_statement",
    "unsafe_keyword",
    "syntax_error",
    "unknown",
)


def policy_failure_type(validation: dict[str, Any]) -> str | None:
    """Map central-validator failures to stable lifecycle categories.

    The ordering is intentional: a mutating statement containing a blocked
    keyword is first classified as non-SELECT, while other unsafe keyword
    failures outrank syntax and query-shape policy failures.
    """
    if bool(validation.get("is_valid", validation.get("ok", False))):
        return None
    checks = validation.get("checks") or {}
    if not checks.get("select_only", False):
        return "non_select_statement" if checks.get("parse", False) else (
            "unsafe_keyword" if not checks.get("no_blocked_keywords", True) else "syntax_error"
        )
    if not checks.get("no_blocked_keywords", True):
        return "unsafe_keyword"
    if not checks.get("parse", False):
        return "syntax_error"
    if not checks.get("no_select_star", True):
        return "select_star_blocked"
    if not checks.get("limit_present", True) or not checks.get("limit_within_bounds", True):
        return "limit_policy_failed"
    return "unknown"


class SQLValidator:
    def validate(
        self,
        sql: str | None,
        schema: dict[str, Any] | SchemaGraph | None = None,
        max_limit: int = 1000,
        dialect: str = "sqlite",
    ) -> dict[str, Any]:
        checks = {
            "parse": False,
            "select_only": False,
            "single_statement": False,
            "no_blocked_keywords": False,
            "no_comments": False,
            "no_select_star": False,
            "tables_exist": True,
            "columns_exist": True,
            "no_sensitive_columns": True,
            "limit_present": False,
            "limit_within_bounds": False,
            "no_dangerous_functions": True,
        }
        issues: list[str] = []
        referenced_tables: list[str] = []
        referenced_columns: list[str] = []

        if not sql or not sql.strip():
            issues.append("SQL is empty.")
            return self._payload(False, checks, issues, referenced_tables, referenced_columns)

        upper_sql = sql.upper()
        checks["no_comments"] = "--" not in sql and "/*" not in sql and "*/" not in sql
        if not checks["no_comments"]:
            issues.append("SQL comments are not allowed.")

        blocked = sorted(keyword for keyword in BLOCKED_KEYWORDS if re.search(rf"\b{keyword}\b", upper_sql))
        checks["no_blocked_keywords"] = not blocked
        if blocked:
            issues.append("Blocked SQL keyword(s): " + ", ".join(blocked))

        try:
            parsed = sqlglot.parse(sql, read=self._sqlglot_dialect(dialect))
        except Exception as exc:
            issues.append(f"SQL parse failed: {exc}")
            return self._payload(False, checks, issues, referenced_tables, referenced_columns)

        checks["parse"] = True
        checks["single_statement"] = len(parsed) == 1
        if not checks["single_statement"]:
            issues.append("Only one SQL statement is allowed.")
        statement = parsed[0] if parsed else None
        checks["select_only"] = isinstance(statement, exp.Select)
        if not checks["select_only"]:
            issues.append("Only SELECT statements are allowed.")
            return self._payload(False, checks, issues, referenced_tables, referenced_columns)

        select = statement
        checks["no_select_star"] = not self._has_select_star(select)
        if not checks["no_select_star"]:
            issues.append("SELECT * is not allowed.")
        select_aliases = {expression.alias for expression in select.expressions if expression.alias}

        for table in select.find_all(exp.Table):
            if table.name:
                referenced_tables.append(table.name)

        for column in select.find_all(exp.Column):
            name = column.name
            if not name or name == "*":
                continue
            qualified = f"{column.table}.{name}" if column.table else name
            referenced_columns.append(qualified)
            if self._is_sensitive(name):
                checks["no_sensitive_columns"] = False

        if not checks["no_sensitive_columns"]:
            issues.append("Sensitive columns are not allowed.")

        functions = {item.name.lower() for item in select.find_all(exp.Func) if item.name}
        dangerous = sorted(functions & DANGEROUS_FUNCTIONS)
        checks["no_dangerous_functions"] = not dangerous
        if dangerous:
            issues.append("Dangerous SQL function(s): " + ", ".join(dangerous))

        limit = select.args.get("limit")
        checks["limit_present"] = limit is not None
        if not checks["limit_present"]:
            issues.append("SELECT must include LIMIT.")
        checks["limit_within_bounds"] = self._limit_ok(limit, max_limit)
        if checks["limit_present"] and not checks["limit_within_bounds"]:
            issues.append(f"LIMIT must be <= {max_limit}.")

        schema_tables = self._schema_tables(schema)
        if schema_tables:
            unknown_tables = [table for table in referenced_tables if table not in schema_tables]
            checks["tables_exist"] = not unknown_tables
            if unknown_tables:
                issues.append("Unknown table(s): " + ", ".join(sorted(set(unknown_tables))))

            unknown_columns = []
            for qualified in referenced_columns:
                if "." in qualified:
                    table, column = qualified.split(".", 1)
                    if table in schema_tables and column not in schema_tables[table]:
                        unknown_columns.append(qualified)
                    continue
                if qualified in select_aliases:
                    continue
                candidate_tables = referenced_tables or list(schema_tables)
                if not any(qualified in schema_tables.get(table, set()) for table in candidate_tables):
                    unknown_columns.append(qualified)
            checks["columns_exist"] = not unknown_columns
            if unknown_columns:
                issues.append("Unknown column(s): " + ", ".join(sorted(set(unknown_columns))))

        valid = all(checks.values())
        return self._payload(valid, checks, issues, referenced_tables, referenced_columns)

    @staticmethod
    def _payload(
        is_valid: bool,
        checks: dict[str, bool],
        issues: list[str],
        referenced_tables: list[str],
        referenced_columns: list[str],
    ) -> dict[str, Any]:
        return {
            "is_valid": is_valid,
            "ok": is_valid,
            "message": "ok" if is_valid else "SQL validation failed",
            "checks": checks,
            "issues": issues,
            "referenced_tables": sorted(set(referenced_tables)),
            "referenced_columns": sorted(set(referenced_columns)),
        }

    @staticmethod
    def _has_select_star(statement: exp.Select) -> bool:
        for item in statement.expressions:
            text = item.sql(dialect="sqlite").strip()
            if text == "*" or text.endswith(".*"):
                return True
        return False

    @staticmethod
    def _limit_ok(limit: Any, max_limit: int) -> bool:
        if limit is None:
            return False
        expression = getattr(limit, "expression", None)
        if expression is None:
            return False
        try:
            return int(expression.name) <= max_limit
        except Exception:
            return False

    @staticmethod
    def _schema_tables(schema: dict[str, Any] | SchemaGraph | None) -> dict[str, set[str]]:
        if schema is None:
            return {}
        if isinstance(schema, SchemaGraph):
            return {table: set(info.columns) for table, info in schema.tables.items()}
        raw_tables = schema.get("tables", schema)
        normalized: dict[str, set[str]] = {}
        for table, info in raw_tables.items():
            columns = info.get("columns", info) if isinstance(info, dict) else getattr(info, "columns", {})
            if isinstance(columns, dict):
                normalized[str(table)] = {str(column) for column in columns}
            else:
                values = []
                for column in columns:
                    raw = asdict(column) if is_dataclass(column) else column
                    values.append(str(raw.get("name", raw)) if isinstance(raw, dict) else str(raw))
                normalized[str(table)] = set(values)
        return normalized

    @staticmethod
    def _is_sensitive(column: str) -> bool:
        name = column.lower()
        return any(marker in name for marker in SENSITIVE_MARKERS)

    @staticmethod
    def _sqlglot_dialect(dialect: str) -> str:
        normalized = (dialect or "sqlite").lower()
        if normalized == "postgresql":
            return "postgres"
        if normalized in {"sqlite", "mysql", "postgres"}:
            return normalized
        return "sqlite"
