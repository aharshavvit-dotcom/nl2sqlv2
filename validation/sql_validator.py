from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any

import sqlglot
from sqlglot import exp

from db.schema_graph import SchemaGraph


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


def _unquoted_sql(sql: str) -> str:
    """Remove quoted strings/identifiers before keyword policy scanning."""
    return re.sub(
        r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[(?:\]\]|[^\]])*\]",
        " ",
        sql,
    )


def root_cause_hint(sql: str | None, validation: dict[str, Any]) -> str:
    """Return a stable, deterministic hint for an invalid SQL example."""
    failure = policy_failure_type(validation)
    if failure == "unsafe_keyword" or not (validation.get("checks") or {}).get("no_blocked_keywords", True):
        return "unsafe_keyword"
    text = str(sql or "")
    if re.search(r'(?is)\bAS\s+(?!["`\[])(?:#|\d\S*|[^,\n]+[\s()/#.][^,\n]*)(?=\s*(?:,|FROM\b|LIMIT\b|$))', text):
        return "unquoted_alias"
    if re.search(r'\.\s*["`][^"`]+["`]\s*\.', text):
        return "malformed_identifier"
    unquoted_refs = re.finditer(
        r'(?is)(?:"(?:""|[^"])*"|`(?:``|[^`])*`)\s*\.\s*'
        r'(?!["`\[])(?P<identifier>.+?)(?=\s*(?:\bAS\b|,|\bFROM\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|$))',
        text,
    )
    if any(re.search(r'[\s./#()\-]', match.group("identifier").strip()) for match in unquoted_refs):
        return "malformed_identifier"
    if failure == "syntax_error":
        return "parse_error"
    return "unknown"


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

        policy_sql = _unquoted_sql(sql).upper()
        checks["no_comments"] = "--" not in sql and "/*" not in sql and "*/" not in sql
        if not checks["no_comments"]:
            issues.append("SQL comments are not allowed.")

        blocked = sorted(keyword for keyword in BLOCKED_KEYWORDS if re.search(rf"\b{keyword}\b", policy_sql))
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

        # Collect CTE names (these are valid table references, not physical tables)
        cte_names: set[str] = set()
        with_clause = select.args.get("with_") or select.args.get("with")
        if with_clause is not None:
            for cte in getattr(with_clause, "expressions", []):
                if cte.alias:
                    cte_names.add(str(cte.alias).lower())

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

        # LIMIT check — only enforced on the outermost query, not inner CTEs/subqueries
        limit = select.args.get("limit")
        checks["limit_present"] = limit is not None
        if not checks["limit_present"]:
            issues.append("SELECT must include LIMIT.")
        checks["limit_within_bounds"] = self._limit_ok(limit, max_limit)
        if checks["limit_present"] and not checks["limit_within_bounds"]:
            issues.append(f"LIMIT must be <= {max_limit}.")

        schema_tables = self._schema_tables(schema)
        if schema_tables:
            # CTE names are valid references — don't flag them as unknown
            unknown_tables = [
                table for table in referenced_tables
                if table not in schema_tables and table.lower() not in cte_names
            ]
            checks["tables_exist"] = not unknown_tables
            if unknown_tables:
                issues.append("Unknown table(s): " + ", ".join(sorted(set(unknown_tables))))

            unknown_columns = []
            for qualified in referenced_columns:
                if "." in qualified:
                    table, column = qualified.split(".", 1)
                    # Skip CTE-referenced columns (we don't have their schema)
                    if table.lower() in cte_names:
                        continue
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

    def validate_with_repair(
        self,
        sql: str | None,
        schema: dict[str, Any] | SchemaGraph | None = None,
        max_limit: int = 1000,
        dialect: str = "sqlite",
        default_limit: int = 100,
        query_ir_valid: bool = True,
    ) -> dict[str, Any]:
        """Validate SQL and apply only deterministic, semantics-preserving repairs."""
        original_validation = self.validate(sql, schema=schema, max_limit=max_limit, dialect=dialect)
        result: dict[str, Any] = {
            "original_sql": sql,
            "original_validation": original_validation,
            "repair_attempted": False,
            "repair_succeeded": False,
            "repair_actions": [],
            "repaired_sql": None,
            "final_sql": sql if original_validation.get("is_valid") else None,
            "final_validation": original_validation,
        }
        if original_validation.get("is_valid") or not query_ir_valid or not sql or not sql.strip():
            return result

        repaired = self._safe_repair_sql(
            sql,
            schema=schema,
            dialect=dialect,
            default_limit=min(default_limit, max_limit),
        )
        result["repair_attempted"] = bool(repaired.get("attempted", False))
        result["repair_actions"] = list(repaired.get("actions") or [])
        candidate = repaired.get("sql")
        if not candidate:
            return result
        repaired_validation = self.validate(candidate, schema=schema, max_limit=max_limit, dialect=dialect)
        result["repaired_sql"] = candidate
        result["final_validation"] = repaired_validation
        if repaired_validation.get("is_valid"):
            result["repair_succeeded"] = True
            result["final_sql"] = candidate
        return result

    def _safe_repair_sql(
        self,
        sql: str,
        schema: dict[str, Any] | SchemaGraph | None,
        dialect: str,
        default_limit: int,
    ) -> dict[str, Any]:
        policy_sql = _unquoted_sql(sql).upper()
        if any(re.search(rf"\b{keyword}\b", policy_sql) for keyword in BLOCKED_KEYWORDS):
            return {"attempted": False, "sql": None, "actions": []}

        actions: list[str] = []
        candidate = sql.strip()
        without_comments = re.sub(r"\s*(?:--[^\r\n]*|/\*.*?\*/)\s*$", "", candidate, flags=re.DOTALL).strip()
        without_terminator = without_comments.rstrip().removesuffix(";").rstrip()
        if without_terminator != candidate:
            candidate = without_terminator
            actions.append("removed_trailing_comment_or_semicolon")
        # Renderer bugs from older bundles can leave display aliases unquoted.
        # This rewrite changes only the alias token and is done before parsing,
        # because malformed aliases are exactly what prevents sqlglot parsing.
        alias_repaired = self._quote_unquoted_aliases(candidate)
        if alias_repaired != candidate:
            candidate = alias_repaired
            actions.append("quoted_unquoted_alias")
        try:
            parsed = sqlglot.parse(candidate, read=self._sqlglot_dialect(dialect))
        except Exception:
            return {"attempted": bool(actions), "sql": candidate if actions else None, "actions": actions}
        if len(parsed) != 1 or not isinstance(parsed[0], exp.Select):
            return {"attempted": False, "sql": None, "actions": []}
        select = parsed[0]

        if self._has_select_star(select):
            tables = [table.name for table in select.find_all(exp.Table) if table.name]
            schema_tables = self._schema_tables(schema)
            if len(set(tables)) != 1 or not schema_tables.get(tables[0]):
                return {"attempted": bool(actions), "sql": candidate if actions else None, "actions": actions}
            columns = [
                column for column in sorted(schema_tables[tables[0]])
                if not self._is_sensitive(column)
            ]
            if not columns:
                return {"attempted": bool(actions), "sql": candidate if actions else None, "actions": actions}
            select.set("expressions", [exp.column(column) for column in columns])
            actions.append("replaced_select_star_with_explicit_columns")

        if select.args.get("limit") is None:
            select.set("limit", exp.Limit(expression=exp.Literal.number(default_limit)))
            actions.append(f"added_limit_{default_limit}")
        normalized = select.sql(dialect=self._sqlglot_dialect(dialect))
        if normalized != candidate and not actions:
            actions.append("normalized_identifier_quoting")
        return {"attempted": bool(actions), "sql": normalized, "actions": actions}

    @staticmethod
    def _quote_unquoted_aliases(sql: str) -> str:
        if not re.match(r"(?is)^\s*SELECT\b", sql):
            return sql

        pattern = re.compile(
            r'(?is)\bAS\s+(?!["`\[])(?P<alias>.+?)(?=\s*(?:,|\bFROM\b|\bLIMIT\b|$))'
        )

        def replace(match: re.Match[str]) -> str:
            alias = match.group("alias").strip()
            # Do not rewrite SQL type names in CAST(... AS TYPE).
            prefix = sql[: match.start()]
            if prefix.count("(") > prefix.count(")"):
                return match.group(0)
            if not alias or re.search(r"\b(?:JOIN|WHERE|GROUP|ORDER|HAVING)\b", alias, re.IGNORECASE):
                return match.group(0)
            return 'AS "' + alias.replace('"', '""') + '"'

        return pattern.sub(replace, sql)

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
