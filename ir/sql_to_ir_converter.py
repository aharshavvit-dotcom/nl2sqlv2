from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

import sqlglot
from sqlglot import exp

from datasets.models import DatabaseSchema
from nl2sql_v1.schema import SchemaGraph
from validation.sql_validator import SQLValidator

from .ir_roundtrip_validator import IRRoundtripValidator
from .ir_to_sql_renderer import IRToSQLRenderer
from .ir_validator import IRValidator
from .query_ir_models import IRDateFilter, IRDimension, IRFilter, IRJoin, IRMetric, IROrderBy, QueryIR
from .sql_to_ir_errors import (
    IRConstructionFailure,
    MISSING_BASE_TABLE,
    NESTED_QUERY,
    NON_SELECT,
    PARSE_ERROR,
    SET_OPERATION,
    UNKNOWN_SCHEMA_REFERENCE,
    UNSUPPORTED_CASE,
    UNSUPPORTED_EXPRESSION,
    UNSUPPORTED_HAVING,
    WINDOW_FUNCTION,
    SQLParseFailure,
    SQLToIRError,
    SchemaResolutionFailure,
    UnsupportedSQLPattern,
)
from .sql_to_ir_rules import (
    date_grain_from_expression,
    date_grain_from_sql,
    detect_intent_from_ast,
    extract_aggregations,
    extract_group_by,
    extract_joins,
    extract_limit,
    extract_order_by,
    extract_select_expressions,
    extract_tables,
    extract_where_filters,
    has_nested_query,
    has_set_operation,
    has_window_function,
    is_select_query,
    normalize_column_ref,
    unwrap_alias,
)


SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DATE_VALUE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
APPROVED_REVENUE_EXPR = "order_items.quantity * order_items.price"


class SQLToIRConverter:
    def __init__(self, dialect: str = "sqlite", max_limit: int = 1000):
        self.dialect = dialect or "sqlite"
        self.max_limit = max_limit
        self.ir_validator = IRValidator(max_limit=max_limit)
        self.sql_renderer = IRToSQLRenderer(max_limit=max_limit)
        self.sql_validator = SQLValidator()
        self.roundtrip_validator = IRRoundtripValidator(dialect=self.dialect, max_limit=max_limit)

    def convert(
        self,
        question: str,
        sql: str,
        schema: dict[str, Any] | DatabaseSchema | SchemaGraph | None,
        dataset_name: str | None = None,
        db_id: str | None = None,
        example_id: str | None = None,
        split: str | None = None,
    ) -> dict[str, Any]:
        metadata = self._metadata(dataset_name, db_id, example_id, split, sql, schema)
        warnings: list[str] = []
        try:
            ast = self._parse(sql)
            self._reject_unsupported(ast, sql)
            query_ir = self._construct_query_ir(question, ast, schema, metadata, warnings)
            validation_schema = self._schema_to_validation(schema)
            ir_validation = self.ir_validator.validate(query_ir, schema=validation_schema)
            if not ir_validation.is_valid:
                raise SchemaResolutionFailure(UNKNOWN_SCHEMA_REFERENCE, "; ".join(ir_validation.errors), sql)
            rendered_sql = self.sql_renderer.render(query_ir)
            sql_validation = self.sql_validator.validate(
                rendered_sql,
                schema=validation_schema,
                max_limit=self.max_limit,
                dialect=self.dialect,
            )
            if not sql_validation.get("is_valid"):
                raise IRConstructionFailure(UNSUPPORTED_EXPRESSION, "; ".join(sql_validation.get("issues", [])), sql)
            roundtrip = self.roundtrip_validator.validate_roundtrip(sql, query_ir, rendered_sql, schema=validation_schema)
            if not roundtrip.get("is_valid"):
                raise IRConstructionFailure(UNSUPPORTED_EXPRESSION, "; ".join(roundtrip.get("issues", [])), sql)
            return {
                "success": True,
                "query_ir": query_ir.model_dump(),
                "ir_validation": ir_validation.model_dump(),
                "roundtrip_sql": rendered_sql,
                "sql_validation": sql_validation,
                "roundtrip_validation": roundtrip,
                "unsupported_reason": None,
                "warnings": warnings,
                "metadata": metadata,
            }
        except SQLToIRError as exc:
            return self._unsupported(exc, metadata, warnings)
        except Exception as exc:
            return self._unsupported(IRConstructionFailure(UNSUPPORTED_EXPRESSION, str(exc), sql), metadata, warnings)

    def _construct_query_ir(
        self,
        question: str,
        ast: exp.Expression,
        schema: dict[str, Any] | DatabaseSchema | SchemaGraph | None,
        metadata: dict[str, Any],
        warnings: list[str],
    ) -> QueryIR:
        schema_tables = self._schema_tables(schema)
        table_aliases = self._table_aliases(ast)
        tables = extract_tables(ast)
        self._validate_identifiers(tables)
        source_from = self._source_from_table(ast)
        select_expressions = extract_select_expressions(ast)
        aggregations = extract_aggregations(ast)
        group_by_sql = extract_group_by(ast)
        order_by_sql = extract_order_by(ast)
        where_filters = extract_where_filters(ast)
        limit = extract_limit(ast)
        if limit is None:
            limit = 100
            warnings.append("LIMIT was missing in source SQL; default limit was added to QueryIR.")
        if limit > self.max_limit:
            limit = self.max_limit
            warnings.append(f"LIMIT exceeded {self.max_limit}; QueryIR limit was capped.")

        features = {
            "aggregations": aggregations,
            "group_by": group_by_sql,
            "order_by": order_by_sql,
            "where_filters": where_filters,
            "limit": limit,
        }
        intent = detect_intent_from_ast(ast, features)
        metrics = self._metrics(aggregations, tables, table_aliases, schema_tables, source_from)
        date_filters = self._date_grain_filters(ast, table_aliases, schema_tables)
        dimensions = self._dimensions(intent, ast, select_expressions, group_by_sql, table_aliases, schema_tables, source_from)
        filters, absolute_date_filters = self._filters(where_filters, table_aliases, schema_tables, source_from)
        date_filters.extend(absolute_date_filters)
        joins = self._joins(extract_joins(ast), table_aliases, schema_tables)
        base_table = self._base_table(source_from, tables, metrics, intent)
        if not base_table:
            raise SchemaResolutionFailure(MISSING_BASE_TABLE, "Could not determine a base table.", metadata.get("source_sql"))
        required_tables = self._required_tables(base_table, tables, metrics, dimensions, filters, date_filters, joins)
        group_by = self._query_ir_group_by(intent, dimensions, date_filters)
        order_by = self._order_by(order_by_sql, metrics, dimensions, date_filters)
        select_mode = self._select_mode(intent)

        query_ir = QueryIR(
            query_ir_id=f"sqlir_{uuid4().hex}",
            question=question,
            normalized_question=self._normalize_question(question),
            intent=intent,
            template_id=intent,
            dialect=self.dialect,
            base_table=base_table,
            required_tables=required_tables,
            metrics=metrics,
            dimensions=dimensions,
            filters=filters,
            date_filters=date_filters,
            joins=joins,
            group_by=group_by,
            order_by=order_by,
            limit=limit,
            select_mode=select_mode,
            warnings=warnings,
            metadata={
                **metadata,
                "tables": tables,
                "validation_context": {
                    "schema_context": self._schema_context_metadata(schema),
                },
            },
        )
        self._validate_schema_refs(query_ir, schema_tables)
        return query_ir

    def _parse(self, sql: str) -> exp.Expression:
        last_error: Exception | None = None
        for dialect in [None, self.dialect]:
            try:
                return sqlglot.parse_one(sql, read=dialect)
            except Exception as exc:
                last_error = exc
        raise SQLParseFailure(PARSE_ERROR, str(last_error), sql)

    @staticmethod
    def _reject_unsupported(ast: exp.Expression, sql: str) -> None:
        if has_set_operation(ast):
            raise UnsupportedSQLPattern(SET_OPERATION, "Set operations are not supported for SQL-to-IR conversion.", sql)
        if not is_select_query(ast):
            raise UnsupportedSQLPattern(NON_SELECT, "Only SELECT statements are supported.", sql)
        if has_nested_query(ast):
            raise UnsupportedSQLPattern(NESTED_QUERY, "Nested queries are not supported for SQL-to-IR conversion.", sql)
        if has_window_function(ast):
            raise UnsupportedSQLPattern(WINDOW_FUNCTION, "Window functions are not supported for SQL-to-IR conversion.", sql)
        if ast.find(exp.Having) is not None:
            raise UnsupportedSQLPattern(UNSUPPORTED_HAVING, "HAVING clauses are not supported in this phase.", sql)
        if ast.find(exp.Case) is not None:
            raise UnsupportedSQLPattern(UNSUPPORTED_CASE, "CASE expressions are not supported in this phase.", sql)
        if ast.find(exp.Or) is not None:
            raise UnsupportedSQLPattern(UNSUPPORTED_EXPRESSION, "OR filters are not supported in this phase.", sql)
        for item in getattr(ast, "expressions", []) or []:
            inner = unwrap_alias(item)
            if isinstance(inner, (exp.Column, exp.Star)):
                continue
            if date_grain_from_expression(inner):
                continue
            if any(True for _ in inner.find_all(*rules_aggregation_types())):
                continue
            raise UnsupportedSQLPattern(UNSUPPORTED_EXPRESSION, f"Unsupported SELECT expression: {inner.sql()}", sql)

    def _metrics(
        self,
        aggregations: list[dict[str, Any]],
        tables: list[str],
        table_aliases: dict[str, str],
        schema_tables: dict[str, set[str]],
        source_from: str | None,
    ) -> list[IRMetric]:
        metrics: list[IRMetric] = []
        for aggregation in aggregations:
            function = str(aggregation["function"]).upper()
            alias = self._safe_alias(aggregation.get("alias") or ("record_count" if function == "COUNT" else function.lower()))
            argument = aggregation.get("argument")
            if function == "COUNT" and (argument is None or isinstance(argument, exp.Star)):
                table = source_from or (tables[0] if tables else None)
                metrics.append(
                    IRMetric(
                        name=alias,
                        aggregation="COUNT",
                        table=table,
                        column="*",
                        expression="*",
                        alias=alias,
                        source_slot="metric",
                        confidence=1.0,
                    )
                )
                continue
            expression, table, column = self._metric_expression(argument, tables, table_aliases, schema_tables, source_from)
            metrics.append(
                IRMetric(
                    name=alias,
                    aggregation=function,
                    table=table,
                    column=column,
                    expression=expression,
                    alias=alias,
                    source_slot="metric",
                    confidence=1.0,
                )
            )
        return metrics

    def _metric_expression(
        self,
        argument: exp.Expression,
        tables: list[str],
        table_aliases: dict[str, str],
        schema_tables: dict[str, set[str]],
        source_from: str | None,
    ) -> tuple[str, str | None, str | None]:
        if isinstance(argument, exp.Column):
            table, column = self._resolve_column(normalize_column_ref(argument), tables, table_aliases, schema_tables, source_from)
            return f"{table}.{column}", table, column
        if isinstance(argument, exp.Mul):
            refs = [
                self._resolve_column(normalize_column_ref(item), tables, table_aliases, schema_tables, source_from)
                for item in argument.find_all(exp.Column)
            ]
            expression = " * ".join(f"{table}.{column}" for table, column in refs)
            if expression != APPROVED_REVENUE_EXPR:
                raise UnsupportedSQLPattern(UNSUPPORTED_EXPRESSION, f"Unsupported arithmetic metric expression: {expression}", argument.sql())
            return expression, "order_items", None
        raise UnsupportedSQLPattern(UNSUPPORTED_EXPRESSION, f"Unsupported metric expression: {argument.sql()}", argument.sql())

    def _dimensions(
        self,
        intent: str,
        ast: exp.Expression,
        select_expressions: list[dict[str, Any]],
        group_by_sql: list[str],
        table_aliases: dict[str, str],
        schema_tables: dict[str, set[str]],
        source_from: str | None,
    ) -> list[IRDimension]:
        if intent not in {"metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension", "count_by_dimension"}:
            return []
        dimensions: list[IRDimension] = []
        group_lookup = {self._normalize_sql(item): item for item in group_by_sql}
        alias_by_expression = {self._normalize_sql(item["sql"]): item.get("alias") for item in select_expressions}
        for group_item in group_lookup.values():
            if date_grain_from_sql(group_item):
                continue
            group_expr = self._find_group_expression(ast, group_item)
            ref = normalize_column_ref(group_expr)
            table, column = self._resolve_column(ref, list(table_aliases.values()), table_aliases, schema_tables, source_from)
            expression = f"{table}.{column}"
            alias = self._safe_alias(alias_by_expression.get(self._normalize_sql(group_item)) or self._infer_dimension_name(column))
            dimensions.append(
                IRDimension(
                    name=alias,
                    table=table,
                    column=column,
                    expression=expression,
                    alias=alias,
                    source_slot="dimension",
                    confidence=1.0,
                )
            )
        return dimensions

    @staticmethod
    def _find_group_expression(ast: exp.Expression, group_sql: str) -> exp.Expression | None:
        group = ast.args.get("group")
        if group is None:
            return None
        normalized = SQLToIRConverter._normalize_sql(group_sql)
        for item in group.expressions:
            if SQLToIRConverter._normalize_sql(item.sql(dialect="sqlite")) == normalized:
                return item
        return None

    def _date_grain_filters(
        self,
        ast: exp.Expression,
        table_aliases: dict[str, str],
        schema_tables: dict[str, set[str]],
    ) -> list[IRDateFilter]:
        for item in [*(getattr(ast, "expressions", []) or []), *self._group_expressions(ast)]:
            inner = unwrap_alias(item)
            grain = date_grain_from_expression(inner)
            if not grain:
                continue
            table = table_aliases.get(grain["date_table"], grain["date_table"])
            column = grain["date_column"]
            self._ensure_identifier(table)
            self._ensure_identifier(column)
            if schema_tables and (table not in schema_tables or column not in schema_tables[table]):
                raise SchemaResolutionFailure(UNKNOWN_SCHEMA_REFERENCE, f"Unknown date column: {table}.{column}", inner.sql())
            return [
                IRDateFilter(
                    date_table=table,
                    date_column=column,
                    date_expression=f"{table}.{column}",
                    filter_type="grain",
                    start_date=None,
                    end_date=None,
                    date_grain=grain["date_grain"],
                    raw_text=inner.sql(dialect="sqlite"),
                    confidence=1.0,
                )
            ]
        return []

    @staticmethod
    def _group_expressions(ast: exp.Expression) -> list[exp.Expression]:
        group = ast.args.get("group")
        return list(group.expressions) if group is not None else []

    def _filters(
        self,
        where_filters: list[dict[str, Any]],
        table_aliases: dict[str, str],
        schema_tables: dict[str, set[str]],
        source_from: str | None,
    ) -> tuple[list[IRFilter], list[IRDateFilter]]:
        filters: list[IRFilter] = []
        date_ranges: dict[str, dict[str, Any]] = {}
        tables = list(dict.fromkeys(table_aliases.values()))
        for item in where_filters:
            table, column = self._resolve_column(item.get("left"), tables, table_aliases, schema_tables, source_from)
            expression = f"{table}.{column}"
            value = item.get("value")
            operator = str(item.get("operator") or "equals")
            if self._is_date_filter(column, value):
                range_info = date_ranges.setdefault(expression, {"table": table, "column": column, "start_date": None, "end_date": None, "raw": []})
                range_info["raw"].append(item.get("expression"))
                if operator in {"greater_than", "greater_equal"}:
                    range_info["start_date"] = str(value)
                elif operator in {"less_than", "less_equal"}:
                    range_info["end_date"] = str(value)
                continue
            filters.append(
                IRFilter(
                    name=self._infer_dimension_name(column),
                    table=table,
                    column=column,
                    expression=expression,
                    operator=operator,  # type: ignore[arg-type]
                    value=value,
                    value_type="number" if isinstance(value, (int, float)) else "string",
                    raw_text=str(value),
                    confidence=1.0,
                )
            )
        date_filters = [
            IRDateFilter(
                date_table=info["table"],
                date_column=info["column"],
                date_expression=expression,
                filter_type="absolute_range",
                start_date=info.get("start_date"),
                end_date=info.get("end_date"),
                date_grain=None,
                raw_text=" AND ".join(str(item) for item in info.get("raw", []) if item),
                confidence=1.0,
            )
            for expression, info in date_ranges.items()
        ]
        return filters, date_filters

    def _joins(self, raw_joins: list[dict[str, Any]], table_aliases: dict[str, str], schema_tables: dict[str, set[str]]) -> list[IRJoin]:
        joins: list[IRJoin] = []
        tables = list(dict.fromkeys(table_aliases.values()))
        for item in raw_joins:
            left_table, left_column = self._resolve_column(item.get("left"), tables, table_aliases, schema_tables, None)
            right_table, right_column = self._resolve_column(item.get("right"), tables, table_aliases, schema_tables, None)
            condition = f"{left_table}.{left_column} = {right_table}.{right_column}"
            joins.append(
                IRJoin(
                    left_table=left_table,
                    left_column=left_column,
                    right_table=right_table,
                    right_column=right_column,
                    join_type=item.get("join_type") or "INNER",
                    condition=condition,
                    path_order=int(item.get("path_order") or len(joins) + 1),
                    confidence=1.0,
                )
            )
        return joins

    @staticmethod
    def _base_table(source_from: str | None, tables: list[str], metrics: list[IRMetric], intent: str) -> str | None:
        for metric in metrics:
            if metric.expression == APPROVED_REVENUE_EXPR:
                return "order_items"
        if source_from:
            return source_from
        for metric in metrics:
            if metric.table:
                return metric.table
        return tables[0] if tables else None

    @staticmethod
    def _required_tables(
        base_table: str,
        tables: list[str],
        metrics: list[IRMetric],
        dimensions: list[IRDimension],
        filters: list[IRFilter],
        date_filters: list[IRDateFilter],
        joins: list[IRJoin],
    ) -> list[str]:
        values = [
            base_table,
            *tables,
            *[metric.table for metric in metrics],
            *[dimension.table for dimension in dimensions],
            *[item.table for item in filters],
            *[item.date_table for item in date_filters],
            *[join.left_table for join in joins],
            *[join.right_table for join in joins],
        ]
        return [str(item) for item in dict.fromkeys(values) if item]

    @staticmethod
    def _query_ir_group_by(intent: str, dimensions: list[IRDimension], date_filters: list[IRDateFilter]) -> list[str]:
        if intent == "trend_by_date":
            grain = next((item for item in date_filters if item.filter_type == "grain"), None)
            return [f"DATE_GRAIN({grain.date_expression}, {grain.date_grain or 'month'})"] if grain else []
        if intent in {"metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension", "count_by_dimension"}:
            return [dimension.expression for dimension in dimensions]
        return []

    @staticmethod
    def _order_by(
        raw_order_by: list[dict[str, Any]],
        metrics: list[IRMetric],
        dimensions: list[IRDimension],
        date_filters: list[IRDateFilter],
    ) -> list[IROrderBy]:
        metric_aliases = {metric.alias for metric in metrics}
        dimension_aliases = {dimension.alias for dimension in dimensions}
        values: list[IROrderBy] = []
        for item in raw_order_by:
            expression = str(item.get("expression") or "")
            alias = item.get("alias") or expression
            source = "explicit"
            if expression in metric_aliases or alias in metric_aliases:
                source = "count" if expression == "record_count" or alias == "record_count" else "metric"
            elif expression in dimension_aliases or alias in dimension_aliases:
                source = "dimension"
            elif expression == "period" or any(filter_item.filter_type == "grain" for filter_item in date_filters):
                source = "date"
            values.append(
                IROrderBy(
                    expression=expression,
                    alias=alias,
                    direction=item.get("direction") or "ASC",
                    source=source,  # type: ignore[arg-type]
                )
            )
        return values

    @staticmethod
    def _select_mode(intent: str) -> str:
        if intent in {"count_records", "count_by_dimension"}:
            return "count"
        if intent == "trend_by_date":
            return "trend"
        if intent in {"metric_summary", "metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension"}:
            return "aggregate"
        return "records"

    def _resolve_column(
        self,
        ref: dict[str, str | None] | None,
        tables: list[str],
        table_aliases: dict[str, str],
        schema_tables: dict[str, set[str]],
        preferred_table: str | None,
    ) -> tuple[str, str]:
        if ref is None or not ref.get("column"):
            raise SchemaResolutionFailure(UNKNOWN_SCHEMA_REFERENCE, "Could not resolve SQL column reference.")
        raw_table = ref.get("table")
        column = str(ref["column"])
        table = table_aliases.get(str(raw_table), str(raw_table)) if raw_table else None
        self._ensure_identifier(column)
        if table:
            self._ensure_identifier(table)
            if schema_tables and (table not in schema_tables or column not in schema_tables[table]):
                raise SchemaResolutionFailure(UNKNOWN_SCHEMA_REFERENCE, f"Unknown column: {table}.{column}")
            return table, column
        candidates = []
        search_tables = tables or list(schema_tables)
        if schema_tables:
            candidates = [candidate for candidate in search_tables if candidate in schema_tables and column in schema_tables[candidate]]
        elif len(search_tables) == 1:
            candidates = [search_tables[0]]
        if preferred_table and preferred_table in candidates:
            return preferred_table, column
        if len(candidates) == 1:
            table = candidates[0]
            self._ensure_identifier(table)
            return table, column
        raise SchemaResolutionFailure(UNKNOWN_SCHEMA_REFERENCE, f"Ambiguous or unknown column: {column}")

    @staticmethod
    def _source_from_table(ast: exp.Expression) -> str | None:
        from_expr = ast.args.get("from") or ast.args.get("from_")
        if from_expr is None:
            return None
        table = from_expr.this
        if isinstance(table, exp.Table):
            return str(table.name)
        return None

    @staticmethod
    def _table_aliases(ast: exp.Expression) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for table in ast.find_all(exp.Table):
            name = str(table.name)
            aliases[name] = name
            if table.alias:
                aliases[str(table.alias)] = name
        return aliases

    def _validate_identifiers(self, tables: list[str]) -> None:
        for table in tables:
            self._ensure_identifier(table)

    @staticmethod
    def _validate_schema_refs(query_ir: QueryIR, schema_tables: dict[str, set[str]]) -> None:
        if not schema_tables:
            return
        for table in query_ir.required_tables:
            if table not in schema_tables:
                raise SchemaResolutionFailure(UNKNOWN_SCHEMA_REFERENCE, f"Unknown table: {table}")

    @staticmethod
    def _ensure_identifier(value: str | None) -> None:
        if not value or not SAFE_IDENTIFIER_RE.match(value):
            raise UnsupportedSQLPattern(UNSUPPORTED_EXPRESSION, f"Unsupported identifier for QueryIR rendering: {value}")

    @staticmethod
    def _safe_alias(value: str) -> str:
        alias = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip().lower()).strip("_")
        return alias or "value"

    @staticmethod
    def _infer_dimension_name(column: str) -> str:
        name = column.lower()
        for suffix in ["_name", "_id"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return SQLToIRConverter._safe_alias(name)

    @staticmethod
    def _is_date_filter(column: str, value: Any) -> bool:
        lowered = column.lower()
        return "date" in lowered or "time" in lowered or (isinstance(value, str) and DATE_VALUE_RE.match(value) is not None)

    @staticmethod
    def _schema_tables(schema: dict[str, Any] | DatabaseSchema | SchemaGraph | None) -> dict[str, set[str]]:
        if schema is None:
            return {}
        if isinstance(schema, SchemaGraph):
            return {table: set(info.columns) for table, info in schema.tables.items()}
        if isinstance(schema, DatabaseSchema):
            raw_tables = schema.tables
        else:
            raw_tables = schema.get("tables", schema)
        normalized: dict[str, set[str]] = {}
        for table, info in raw_tables.items():
            columns = info.get("columns", info) if isinstance(info, dict) else getattr(info, "columns", {})
            if isinstance(columns, dict):
                normalized[str(table)] = {str(column) for column in columns}
            else:
                normalized[str(table)] = {
                    str(column.get("name", column)) if isinstance(column, dict) else str(column)
                    for column in columns
                }
        return normalized

    @staticmethod
    def _schema_to_validation(schema: dict[str, Any] | DatabaseSchema | SchemaGraph | None) -> dict[str, Any] | SchemaGraph | None:
        if isinstance(schema, DatabaseSchema):
            return schema.to_dict()
        return schema

    @staticmethod
    def _schema_context_metadata(schema: dict[str, Any] | DatabaseSchema | SchemaGraph | None) -> dict[str, Any]:
        tables = SQLToIRConverter._schema_tables(schema)
        return {
            "tables": {
                table: {"columns": {column: {} for column in sorted(columns)}}
                for table, columns in tables.items()
            }
        }

    @staticmethod
    def _metadata(
        dataset_name: str | None,
        db_id: str | None,
        example_id: str | None,
        split: str | None,
        sql: str,
        schema: dict[str, Any] | DatabaseSchema | SchemaGraph | None,
    ) -> dict[str, Any]:
        serialized_schema = getattr(schema, "serialized_schema", None) if schema is not None else None
        if not serialized_schema and isinstance(schema, dict):
            serialized_schema = schema.get("serialized_schema")
        return {
            "dataset_name": dataset_name,
            "db_id": db_id,
            "example_id": example_id,
            "split": split,
            "source_sql": sql,
            "serialized_schema": serialized_schema,
        }

    @staticmethod
    def _normalize_question(question: str) -> str:
        return re.sub(r"\s+", " ", question.strip().lower())

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        return re.sub(r"\s+", " ", sql.lower()).strip()

    @staticmethod
    def _unsupported(error: SQLToIRError, metadata: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
        return {
            "success": False,
            "query_ir": None,
            "ir_validation": None,
            "roundtrip_sql": None,
            "sql_validation": None,
            "roundtrip_validation": None,
            "unsupported_reason": error.reason,
            "error_message": error.message,
            "warnings": warnings,
            "metadata": metadata,
        }


def rules_aggregation_types() -> tuple[type[exp.Expression], ...]:
    return (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)

