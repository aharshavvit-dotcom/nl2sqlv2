from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .query_ir_models import IRDateFilter, IRDimension, IRFilter, IRJoin, IRMetric, IROrderBy, QueryIR
from .query_ir_v2_models import (
    AggregationExpression,
    BetweenPredicate,
    BinaryOperationExpression,
    BooleanPredicate,
    CaseExpression,
    ColumnExpression,
    ComparisonPredicate,
    CTEDefinition,
    DateFilterNode,
    ExistsPredicate,
    FromItem,
    FunctionExpression,
    InLiteralPredicate,
    InSubqueryPredicate,
    JoinNode,
    LiteralExpression,
    NotPredicate,
    NullPredicate,
    OrderByItem,
    QueryNode,
    SelectItem,
    SetOperationNode,
    SubqueryExpression,
    WindowExpression,
)
from .query_ir_v2_validation import unsupported_v2_rendering_capabilities


V1_TO_V2_WARNING = "legacy_query_ir_v1_migrated_to_query_ir_v2"
V2_TO_V1_WARNING = "query_ir_v2_converted_to_v1_compatibility_subset"


class QueryIRCompatibilityError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        capability: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        self.capability = capability

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "path": self.path,
            "capability": self.capability,
        }


def migrate_v1_to_v2(value: QueryIR | QueryNode | dict[str, Any]) -> QueryNode:
    if isinstance(value, QueryNode):
        return value
    if isinstance(value, dict) and value.get("query_ir_version") == "2.0":
        return QueryNode.model_validate(value)

    query_ir = coerce_query_ir_v1(value)
    v1_payload = query_ir.model_dump(mode="json")
    metadata = dict(query_ir.metadata or {})
    metadata["v1_payload"] = _stable_json_dict(v1_payload)
    metadata["migration"] = {
        "from_query_ir_version": "1",
        "to_query_ir_version": "2.0",
        "deterministic": True,
        "source_v1_fingerprint": _payload_hash(v1_payload),
    }

    select_items = [
        _dimension_to_select_item(dimension)
        for dimension in query_ir.dimensions
    ]
    select_items.extend(_metric_to_select_item(metric) for metric in query_ir.metrics)

    migrated_predicates = [_filter_to_predicate(item) for item in query_ir.filters]
    where = _and_predicates(migrated_predicates)

    return QueryNode(
        query_ir_version="2.0",
        query_type="SELECT",
        query_ir_id=query_ir.query_ir_id,
        question=query_ir.question,
        normalized_question=query_ir.normalized_question,
        intent=query_ir.intent,
        template_id=query_ir.template_id,
        dialect=query_ir.dialect,
        from_item=FromItem(table=query_ir.base_table) if query_ir.base_table else None,
        required_tables=list(query_ir.required_tables),
        select_items=select_items,
        joins=[_join_to_v2(join) for join in query_ir.joins],
        where=where,
        predicates=migrated_predicates,
        date_filters=[_date_filter_to_v2(item) for item in query_ir.date_filters],
        group_by=[_expression_from_v1_expression(item) for item in query_ir.group_by],
        order_by=[_order_by_to_v2(item) for item in query_ir.order_by],
        limit=query_ir.limit,
        select_mode=query_ir.select_mode,
        warnings=list(query_ir.warnings),
        capability_metadata={
            "required_capabilities": _capabilities_from_v1(query_ir),
            "unsupported_capabilities": [],
            "renderer_capabilities": ["query_ir_v1_compatibility_renderer"],
            "source_capability_labels": list((query_ir.metadata or {}).get("capability_labels") or []),
        },
        confidence_metadata={
            "overall": _overall_confidence(query_ir),
            "slots": _slot_confidences(query_ir),
            "source": "query_ir_v1_migration",
        },
        metadata=metadata,
    )


def convert_v2_to_v1(value: QueryNode | dict[str, Any]) -> QueryIR:
    query = value if isinstance(value, QueryNode) else QueryNode.model_validate(value)

    # Explicit incompatibility errors for v2-only constructs
    if query.having is not None:
        raise QueryIRCompatibilityError(
            "v2_only_having",
            "HAVING clause cannot be represented in QueryIR v1. Use v2 renderer.",
            path="having",
            capability="having",
        )
    if query.ctes:
        raise QueryIRCompatibilityError(
            "v2_only_cte",
            f"CTE definitions ({', '.join(c.name for c in query.ctes)}) cannot be represented in QueryIR v1.",
            path="ctes",
            capability="common_table_expression",
        )
    if query.set_operations:
        raise QueryIRCompatibilityError(
            "v2_only_set_operation",
            "Set operations (UNION/INTERSECT/EXCEPT) cannot be represented in QueryIR v1.",
            path="set_operations",
            capability="set_operation",
        )

    predicate_roots = [query.where] if query.where is not None else list(query.predicates)
    predicate_error = _first_v1_predicate_compatibility_error(predicate_roots)
    if predicate_error is not None:
        raise predicate_error
    unsupported = unsupported_v2_rendering_capabilities(query)
    if unsupported:
        capability, path = unsupported[0]
        raise QueryIRCompatibilityError(
            "unsupported_v2_rendering_capability",
            f"Current renderer cannot safely render QueryIR v2 capability {capability!r}.",
            path=path,
            capability=capability,
        )

    legacy_payload = query.metadata.get("v1_payload") if isinstance(query.metadata, dict) else None
    base_payload = dict(legacy_payload) if isinstance(legacy_payload, dict) else {}
    metadata = dict(base_payload.get("metadata") or {})
    warnings = list(base_payload.get("warnings") or [])
    if V2_TO_V1_WARNING not in warnings:
        warnings.append(V2_TO_V1_WARNING)

    metrics = _metrics_from_select_items(query.select_items)
    dimensions = _dimensions_from_select_items(query.select_items)
    filters = _filters_from_predicate_roots(predicate_roots)
    date_filters = [_date_filter_from_v2(item) for item in query.date_filters]
    joins = [_join_from_v2(item) for item in query.joins]
    order_by = [_order_by_from_v2(item) for item in query.order_by]

    payload = {
        "query_ir_id": query.query_ir_id or base_payload.get("query_ir_id") or _stable_id(query.model_dump(mode="json")),
        "question": query.question or base_payload.get("question", ""),
        "normalized_question": query.normalized_question or base_payload.get("normalized_question", ""),
        "intent": query.intent or base_payload.get("intent", "show_records"),
        "template_id": query.template_id if query.template_id is not None else base_payload.get("template_id"),
        "dialect": query.dialect or base_payload.get("dialect", "sqlite"),
        "base_table": query.from_item.table if query.from_item and query.from_item.from_type == "TABLE" else base_payload.get("base_table"),
        "required_tables": list(query.required_tables or base_payload.get("required_tables") or []),
        "metrics": metrics if metrics else list(base_payload.get("metrics") or []),
        "dimensions": dimensions if dimensions else list(base_payload.get("dimensions") or []),
        "filters": filters if filters else list(base_payload.get("filters") or []),
        "date_filters": date_filters if date_filters else list(base_payload.get("date_filters") or []),
        "joins": joins if joins else list(base_payload.get("joins") or []),
        "group_by": [_expression_to_v1_sql(item) for item in query.group_by] if query.group_by else list(base_payload.get("group_by") or []),
        "order_by": order_by if order_by else list(base_payload.get("order_by") or []),
        "limit": query.limit if query.limit is not None else int(base_payload.get("limit") or 100),
        "select_mode": query.select_mode or base_payload.get("select_mode", "records"),
        "warnings": warnings,
        "metadata": metadata,
    }
    return coerce_query_ir_v1(payload)


def coerce_query_ir_v1(value: QueryIR | dict[str, Any]) -> QueryIR:
    if isinstance(value, QueryIR):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"Expected QueryIR v1 object or dict, got {type(value).__name__}")
    payload = _normalize_legacy_v1_payload(value)
    return QueryIR.model_validate(payload)


def _normalize_legacy_v1_payload(value: dict[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    base_table = payload.get("base_table")
    payload.setdefault("query_ir_id", _stable_id(payload))
    payload.setdefault("question", "")
    payload.setdefault("normalized_question", str(payload.get("question") or "").strip().lower())
    payload.setdefault("intent", payload.get("template_id") or "show_records")
    payload.setdefault("template_id", payload.get("intent"))
    payload.setdefault("dialect", "sqlite")
    payload.setdefault("required_tables", [base_table] if base_table else [])
    payload.setdefault("metrics", [])
    payload.setdefault("dimensions", [])
    payload.setdefault("filters", [])
    payload.setdefault("date_filters", [])
    payload.setdefault("joins", [])
    payload.setdefault("group_by", [])
    payload.setdefault("order_by", [])
    payload.setdefault("limit", 100)
    payload.setdefault("select_mode", _select_mode_for(payload))
    payload.setdefault("warnings", [])
    payload.setdefault("metadata", {})

    payload["metrics"] = [_normalize_metric(item, base_table) for item in payload.get("metrics") or []]
    payload["dimensions"] = [_normalize_dimension(item, base_table) for item in payload.get("dimensions") or []]
    payload["filters"] = [_normalize_filter(item, base_table) for item in payload.get("filters") or []]
    payload["date_filters"] = [_normalize_date_filter(item, base_table) for item in payload.get("date_filters") or []]
    payload["joins"] = [_normalize_join(item, idx, base_table) for idx, item in enumerate(payload.get("joins") or [])]
    payload["order_by"] = [_normalize_order_by(item) for item in payload.get("order_by") or []]
    return payload


def _dimension_to_select_item(dimension: IRDimension) -> SelectItem:
    return SelectItem(
        expression=ColumnExpression(table=dimension.table, column=dimension.column, raw_expression=dimension.expression),
        alias=dimension.alias,
        name=dimension.name,
        role="dimension",
        source_slot=dimension.source_slot,
        confidence=dimension.confidence,
        legacy_v1=dimension.model_dump(mode="json"),
    )


def _metric_to_select_item(metric: IRMetric) -> SelectItem:
    expression = AggregationExpression(
        function=metric.aggregation,
        argument=_expression_from_v1_expression(metric.expression, table=metric.table, column=metric.column),
    )
    return SelectItem(
        expression=expression,
        alias=metric.alias,
        name=metric.name,
        role="count" if metric.aggregation.upper() == "COUNT" else "metric",
        source_slot=metric.source_slot,
        confidence=metric.confidence,
        legacy_v1=metric.model_dump(mode="json"),
    )


def _filter_to_predicate(item: IRFilter) -> ComparisonPredicate | InLiteralPredicate:
    left = ColumnExpression(table=item.table, column=item.column, raw_expression=item.expression)
    operator = _v1_operator_to_v2(item.operator)
    if item.operator in {"in", "not_in"}:
        values = item.value if isinstance(item.value, list) else [item.value]
        return InLiteralPredicate(
            expression=left,
            values=[LiteralExpression(value=value, value_type=item.value_type) for value in values],
            negated=item.operator == "not_in",
            legacy_v1=item.model_dump(mode="json"),
        )
    value = f"%{item.value}%" if item.operator == "contains" else item.value
    return ComparisonPredicate(
        left=left,
        operator=operator,
        right=LiteralExpression(value=value, value_type=item.value_type),
        legacy_v1=item.model_dump(mode="json"),
    )


def _and_predicates(predicates: list[Any]) -> Any | None:
    if not predicates:
        return None
    if len(predicates) == 1:
        return predicates[0]
    return BooleanPredicate(operator="AND", operands=predicates)


def _date_filter_to_v2(item: IRDateFilter) -> DateFilterNode:
    return DateFilterNode(
        date_expression=ColumnExpression(table=item.date_table, column=item.date_column, raw_expression=item.date_expression),
        filter_type=item.filter_type,
        start_date=item.start_date,
        end_date=item.end_date,
        date_grain=item.date_grain,
        raw_text=item.raw_text,
        confidence=item.confidence,
        legacy_v1=item.model_dump(mode="json"),
    )


def _join_to_v2(join: IRJoin) -> JoinNode:
    return JoinNode(
        join_type=_join_type(join.join_type),
        right=FromItem(table=join.right_table),
        on=ComparisonPredicate(
            left=ColumnExpression(table=join.left_table, column=join.left_column),
            operator="=",
            right=ColumnExpression(table=join.right_table, column=join.right_column),
        ),
        condition=join.condition,
        path_order=join.path_order,
        confidence=join.confidence,
        legacy_v1=join.model_dump(mode="json"),
    )


def _order_by_to_v2(item: IROrderBy) -> OrderByItem:
    return OrderByItem(
        expression=_expression_from_v1_expression(item.alias or item.expression),
        alias=item.alias,
        direction=item.direction,
        source=item.source,
        legacy_v1=item.model_dump(mode="json"),
    )


def _expression_from_v1_expression(expression: str | None, *, table: str | None = None, column: str | None = None) -> Any:
    if column and (not expression or expression in {column, "*", f"{table}.{column}" if table else column}):
        return ColumnExpression(table=table, column=column, raw_expression=expression)
    text = str(expression or "").strip()
    if not text:
        return LiteralExpression(value=None, value_type="null")
    if text == "*":
        return ColumnExpression(column="*", raw_expression="*")
    if text.startswith("DATE_GRAIN("):
        inner = text[len("DATE_GRAIN(") : -1] if text.endswith(")") else text
        parts = [part.strip() for part in inner.split(",", 1)]
        args = [_expression_from_v1_expression(parts[0])]
        if len(parts) > 1:
            args.append(LiteralExpression(value=parts[1], value_type="string"))
        return FunctionExpression(name="DATE_GRAIN", arguments=args)
    binary = _split_binary_expression(text)
    if binary:
        left, operator, right = binary
        return BinaryOperationExpression(
            operator=operator,
            left=_expression_from_v1_expression(left),
            right=_expression_from_v1_expression(right),
        )
    parsed = _parse_qualified(text)
    if parsed:
        return ColumnExpression(table=parsed[0], column=parsed[1], raw_expression=text)
    return ColumnExpression(column=text, raw_expression=text)


def _metrics_from_select_items(items: list[SelectItem]) -> list[dict[str, Any]]:
    metrics = []
    for item in items:
        if item.role not in {"metric", "count"} or not isinstance(item.expression, AggregationExpression):
            continue
        legacy = dict(item.legacy_v1 or {})
        argument = item.expression.argument
        if isinstance(argument, ColumnExpression):
            table, column, expression = _column_parts(argument)
        else:
            table = legacy.get("table")
            column = legacy.get("column")
            expression = _expression_to_v1_sql(argument) if argument is not None else legacy.get("expression", "*")
        legacy.update(
            {
                "name": item.name or legacy.get("name") or item.alias or "metric",
                "aggregation": item.expression.function.upper(),
                "table": table,
                "column": column,
                "expression": expression,
                "alias": item.alias or legacy.get("alias") or item.name or "metric",
                "source_slot": item.source_slot if item.source_slot is not None else legacy.get("source_slot"),
                "confidence": item.confidence,
            }
        )
        metrics.append(legacy)
    return metrics


def _dimensions_from_select_items(items: list[SelectItem]) -> list[dict[str, Any]]:
    dimensions = []
    for item in items:
        if item.role not in {"dimension", "record"}:
            continue
        legacy = dict(item.legacy_v1 or {})
        table, column, expression = _column_parts(item.expression)
        legacy.update(
            {
                "name": item.name or legacy.get("name") or item.alias or column or "dimension",
                "table": table,
                "column": column,
                "expression": expression,
                "alias": item.alias or legacy.get("alias") or item.name or column or "dimension",
                "source_slot": item.source_slot if item.source_slot is not None else legacy.get("source_slot"),
                "confidence": item.confidence,
            }
        )
        dimensions.append(legacy)
    return dimensions


def _filters_from_predicate_roots(predicates: list[Any]) -> list[dict[str, Any]]:
    filters = []
    for predicate in _flatten_v1_compatible_predicates(predicates):
        if isinstance(predicate, ComparisonPredicate):
            legacy = dict(predicate.legacy_v1 or {})
            table, column, expression = _column_parts(predicate.left)
            legacy.update(
                {
                    "name": legacy.get("name") or column,
                    "table": table,
                    "column": column,
                    "expression": expression,
                    "operator": legacy.get("operator") or _v2_operator_to_v1(predicate.operator),
                    "value": legacy.get("value", _literal_value(predicate.right)),
                    "value_type": legacy.get("value_type") or _literal_value_type(predicate.right),
                    "raw_text": legacy.get("raw_text"),
                    "confidence": legacy.get("confidence", 1.0),
                }
            )
            filters.append(legacy)
        elif isinstance(predicate, InLiteralPredicate):
            legacy = dict(predicate.legacy_v1 or {})
            table, column, expression = _column_parts(predicate.expression)
            legacy.update(
                {
                    "name": legacy.get("name") or column,
                    "table": table,
                    "column": column,
                    "expression": expression,
                    "operator": "not_in" if predicate.negated else "in",
                    "value": [_literal_value(item) for item in predicate.values],
                    "value_type": legacy.get("value_type", "string"),
                    "raw_text": legacy.get("raw_text"),
                    "confidence": legacy.get("confidence", 1.0),
                }
            )
            filters.append(legacy)
    return filters


def _flatten_v1_compatible_predicates(predicates: list[Any]) -> list[Any]:
    values: list[Any] = []
    for predicate in predicates:
        if isinstance(predicate, BooleanPredicate) and predicate.operator == "AND":
            values.extend(_flatten_v1_compatible_predicates(predicate.operands))
        elif predicate is not None:
            values.append(predicate)
    return values


def _first_v1_predicate_compatibility_error(predicates: list[Any]) -> QueryIRCompatibilityError | None:
    for index, predicate in enumerate(predicates):
        error = _predicate_compatibility_error(predicate, f"where.{index}" if len(predicates) > 1 else "where")
        if error is not None:
            return error
    return None


def _predicate_compatibility_error(predicate: Any, path: str) -> QueryIRCompatibilityError | None:
    if predicate is None:
        return None
    if isinstance(predicate, ComparisonPredicate):
        return None
    if isinstance(predicate, InLiteralPredicate):
        return None
    if isinstance(predicate, BooleanPredicate):
        if predicate.operator != "AND":
            return QueryIRCompatibilityError(
                "v2_predicate_not_representable_in_v1",
                "OR predicates cannot be represented by QueryIR v1 filters.",
                path=path,
                capability="OR_FILTER",
            )
        for idx, operand in enumerate(predicate.operands):
            error = _predicate_compatibility_error(operand, f"{path}.operands.{idx}")
            if error is not None:
                return error
        return None
    if isinstance(predicate, NotPredicate):
        return QueryIRCompatibilityError(
            "v2_predicate_not_representable_in_v1",
            "NOT predicates cannot be represented by QueryIR v1 filters.",
            path=path,
            capability="NOT_PREDICATE",
        )
    if isinstance(predicate, NullPredicate):
        return QueryIRCompatibilityError(
            "v2_predicate_not_representable_in_v1",
            "NULL predicates cannot be represented by QueryIR v1 filters.",
            path=path,
            capability="NULL_PREDICATE",
        )
    if isinstance(predicate, BetweenPredicate):
        return QueryIRCompatibilityError(
            "v2_predicate_not_representable_in_v1",
            "BETWEEN predicates cannot be represented by QueryIR v1 filters.",
            path=path,
            capability="BETWEEN_PREDICATE",
        )
    return QueryIRCompatibilityError(
        "v2_predicate_not_representable_in_v1",
        f"Predicate {type(predicate).__name__} cannot be represented by QueryIR v1.",
        path=path,
        capability=type(predicate).__name__,
    )


def _date_filter_from_v2(item: DateFilterNode) -> dict[str, Any]:
    legacy = dict(item.legacy_v1 or {})
    table, column, expression = _column_parts(item.date_expression)
    legacy.update(
        {
            "date_table": table,
            "date_column": column,
            "date_expression": expression,
            "filter_type": item.filter_type,
            "start_date": item.start_date,
            "end_date": item.end_date,
            "date_grain": item.date_grain,
            "raw_text": item.raw_text,
            "confidence": item.confidence,
        }
    )
    return legacy


def _join_from_v2(item: JoinNode) -> dict[str, Any]:
    legacy = dict(item.legacy_v1 or {})
    left_table = legacy.get("left_table")
    left_column = legacy.get("left_column")
    right_table = legacy.get("right_table") or item.right.table
    right_column = legacy.get("right_column")
    if isinstance(item.on, ComparisonPredicate):
        left_table, left_column, _ = _column_parts(item.on.left)
        right_table, right_column, _ = _column_parts(item.on.right)
    legacy.update(
        {
            "left_table": left_table,
            "left_column": left_column,
            "right_table": right_table,
            "right_column": right_column,
            "join_type": item.join_type,
            "condition": item.condition or _join_condition(left_table, left_column, right_table, right_column),
            "path_order": item.path_order,
            "confidence": item.confidence,
        }
    )
    return legacy


def _order_by_from_v2(item: OrderByItem) -> dict[str, Any]:
    legacy = dict(item.legacy_v1 or {})
    legacy.update(
        {
            "expression": _expression_to_v1_sql(item.expression),
            "alias": item.alias,
            "direction": item.direction,
            "source": item.source if item.source != "unknown" else legacy.get("source", "explicit"),
        }
    )
    return legacy


def _expression_to_v1_sql(expression: Any) -> str:
    if isinstance(expression, ColumnExpression):
        if expression.raw_expression:
            return expression.raw_expression
        if expression.column == "*":
            return "*"
        return f"{expression.table}.{expression.column}" if expression.table else expression.column
    if isinstance(expression, LiteralExpression):
        return str(expression.value)
    if isinstance(expression, BinaryOperationExpression):
        return f"{_expression_to_v1_sql(expression.left)} {expression.operator} {_expression_to_v1_sql(expression.right)}"
    if isinstance(expression, FunctionExpression):
        args = ", ".join(_expression_to_v1_sql(item) for item in expression.arguments)
        return f"{expression.name}({args})"
    if isinstance(expression, AggregationExpression) and expression.argument is not None:
        return _expression_to_v1_sql(expression.argument)
    raise QueryIRCompatibilityError(
        "v2_not_representable_in_v1",
        f"Expression {type(expression).__name__} cannot be represented by QueryIR v1.",
        capability=type(expression).__name__,
    )


def _column_parts(expression: Any) -> tuple[str | None, str | None, str]:
    if isinstance(expression, ColumnExpression):
        sql = _expression_to_v1_sql(expression)
        return expression.table, expression.column, sql
    sql = _expression_to_v1_sql(expression)
    parsed = _parse_qualified(sql)
    if parsed:
        return parsed[0], parsed[1], sql
    return None, sql, sql


def _literal_value(expression: Any) -> Any:
    if isinstance(expression, LiteralExpression):
        return expression.value
    return _expression_to_v1_sql(expression)


def _literal_value_type(expression: Any) -> str:
    if isinstance(expression, LiteralExpression):
        return expression.value_type
    return "string"


def _normalize_metric(item: Any, base_table: str | None) -> dict[str, Any]:
    raw = dict(item or {})
    aggregation = str(raw.get("aggregation") or raw.get("function") or "COUNT").upper()
    table, column = _table_column(raw.get("table"), raw.get("column"), base_table)
    expression = raw.get("expression") or ("*" if column == "*" else f"{table}.{column}" if table and column else column or "*")
    alias = raw.get("alias") or raw.get("name") or ("record_count" if aggregation == "COUNT" else "metric")
    return {
        "name": raw.get("name") or alias,
        "aggregation": aggregation,
        "table": table,
        "column": column,
        "expression": expression,
        "alias": alias,
        "source_slot": raw.get("source_slot"),
        "confidence": float(raw.get("confidence", 1.0)),
    }


def _normalize_dimension(item: Any, base_table: str | None) -> dict[str, Any]:
    raw = dict(item or {})
    table, column = _table_column(raw.get("table"), raw.get("column") or raw.get("expression"), base_table)
    expression = raw.get("expression") or (f"{table}.{column}" if table and column else column)
    alias = raw.get("alias") or raw.get("name") or _alias_from_column(column)
    return {
        "name": raw.get("name") or alias,
        "table": table or base_table or "",
        "column": column or "",
        "expression": expression or "",
        "alias": alias,
        "source_slot": raw.get("source_slot"),
        "confidence": float(raw.get("confidence", 1.0)),
    }


def _normalize_filter(item: Any, base_table: str | None) -> dict[str, Any]:
    raw = dict(item or {})
    table, column = _table_column(raw.get("table"), raw.get("column") or raw.get("expression"), base_table)
    expression = raw.get("expression") or (f"{table}.{column}" if table and column else column)
    return {
        "name": raw.get("name") or _alias_from_column(column),
        "table": table or base_table or "",
        "column": column or "",
        "expression": expression or "",
        "operator": _symbol_operator_to_v1(raw.get("operator") or "equals"),
        "value": raw.get("value", ""),
        "value_type": raw.get("value_type", "string"),
        "raw_text": raw.get("raw_text"),
        "confidence": float(raw.get("confidence", 1.0)),
    }


def _normalize_date_filter(item: Any, base_table: str | None) -> dict[str, Any]:
    raw = dict(item or {})
    table, column = _table_column(raw.get("date_table"), raw.get("date_column") or raw.get("date_expression"), base_table)
    expression = raw.get("date_expression") or (f"{table}.{column}" if table and column else column)
    return {
        "date_table": table or base_table or "",
        "date_column": column or "",
        "date_expression": expression or "",
        "filter_type": raw.get("filter_type", "absolute_range"),
        "start_date": raw.get("start_date"),
        "end_date": raw.get("end_date"),
        "date_grain": raw.get("date_grain"),
        "raw_text": raw.get("raw_text"),
        "confidence": float(raw.get("confidence", 1.0)),
    }


def _normalize_join(item: Any, path_order: int, base_table: str | None) -> dict[str, Any]:
    raw = dict(item or {})
    left_table = raw.get("left_table")
    left_column = raw.get("left_column")
    right_table = raw.get("right_table") or raw.get("table")
    right_column = raw.get("right_column")
    condition = raw.get("condition")
    if condition and not all([left_table, left_column, right_table, right_column]):
        parsed = _parse_join_condition(condition)
        if parsed:
            left_table, left_column, right_table, right_column = parsed
    return {
        "left_table": left_table or base_table or "",
        "left_column": left_column or "",
        "right_table": right_table or "",
        "right_column": right_column or "",
        "join_type": _join_type(raw.get("join_type", "INNER")),
        "condition": condition or _join_condition(left_table, left_column, right_table, right_column),
        "path_order": int(raw.get("path_order", path_order)),
        "confidence": float(raw.get("confidence", 1.0)),
    }


def _normalize_order_by(item: Any) -> dict[str, Any]:
    raw = dict(item or {})
    return {
        "expression": str(raw.get("expression") or raw.get("alias") or ""),
        "alias": raw.get("alias"),
        "direction": str(raw.get("direction") or "ASC").upper(),
        "source": raw.get("source", "explicit"),
    }


def _select_mode_for(payload: dict[str, Any]) -> str:
    intent = str(payload.get("intent") or payload.get("template_id") or "")
    if intent == "count_records":
        return "count"
    if intent == "trend_by_date":
        return "trend"
    if payload.get("metrics"):
        return "aggregate"
    return "records"


def _capabilities_from_v1(query_ir: QueryIR) -> list[str]:
    capabilities = ["select", "limit"]
    if query_ir.joins:
        capabilities.append("join")
    if query_ir.metrics:
        capabilities.append("aggregation")
    if query_ir.dimensions:
        capabilities.append("projection")
    if query_ir.filters:
        capabilities.append("filter")
    if query_ir.date_filters:
        capabilities.append("date_filter")
    if query_ir.group_by:
        capabilities.append("group_by")
    if query_ir.order_by:
        capabilities.append("order_by")
    return sorted(set(capabilities))


def _overall_confidence(query_ir: QueryIR) -> float | None:
    values: list[float] = []
    values.extend(metric.confidence for metric in query_ir.metrics)
    values.extend(dimension.confidence for dimension in query_ir.dimensions)
    values.extend(item.confidence for item in query_ir.filters)
    values.extend(item.confidence for item in query_ir.date_filters)
    values.extend(item.confidence for item in query_ir.joins)
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _slot_confidences(query_ir: QueryIR) -> dict[str, float]:
    slots: dict[str, float] = {}
    if query_ir.metrics:
        slots["metrics"] = round(sum(item.confidence for item in query_ir.metrics) / len(query_ir.metrics), 6)
    if query_ir.dimensions:
        slots["dimensions"] = round(sum(item.confidence for item in query_ir.dimensions) / len(query_ir.dimensions), 6)
    if query_ir.filters:
        slots["filters"] = round(sum(item.confidence for item in query_ir.filters) / len(query_ir.filters), 6)
    if query_ir.date_filters:
        slots["date_filters"] = round(sum(item.confidence for item in query_ir.date_filters) / len(query_ir.date_filters), 6)
    if query_ir.joins:
        slots["joins"] = round(sum(item.confidence for item in query_ir.joins) / len(query_ir.joins), 6)
    return slots


def _v1_operator_to_v2(operator: str) -> str:
    return {
        "equals": "=",
        "not_equals": "<>",
        "greater_than": ">",
        "greater_equal": ">=",
        "less_than": "<",
        "less_equal": "<=",
        "contains": "LIKE",
        "in": "IN",
        "not_in": "NOT IN",
    }.get(operator, operator)


def _v2_operator_to_v1(operator: str) -> str:
    return {
        "=": "equals",
        "==": "equals",
        "<>": "not_equals",
        "!=": "not_equals",
        ">": "greater_than",
        ">=": "greater_equal",
        "<": "less_than",
        "<=": "less_equal",
        "CONTAINS": "contains",
        "contains": "contains",
        "LIKE": "contains",
        "ILIKE": "contains",
        "IN": "in",
        "NOT IN": "not_in",
    }.get(operator.upper(), operator)


def _symbol_operator_to_v1(operator: Any) -> str:
    return _v2_operator_to_v1(str(operator))


def _join_type(value: Any) -> str:
    text = str(value or "INNER").upper().replace(" JOIN", "")
    aliases = {"LEFT OUTER": "LEFT", "RIGHT OUTER": "RIGHT", "FULL OUTER": "FULL", "JOIN": "INNER"}
    return aliases.get(text, text if text in {"INNER", "LEFT", "RIGHT", "FULL", "CROSS"} else "INNER")


def _split_binary_expression(value: str) -> tuple[str, str, str] | None:
    for operator in ["*", "/", "+", "-"]:
        parts = re.split(rf"\s+\{operator}\s+", value, maxsplit=1)
        if len(parts) == 2:
            return parts[0], operator, parts[1]
    return None


def _parse_qualified(value: str | None) -> tuple[str, str] | None:
    if not value or "." not in str(value):
        return None
    table, column = str(value).split(".", 1)
    if table and column:
        return table.strip('"`[]'), column.strip('"`[]')
    return None


def _table_column(table: Any, column: Any, default_table: str | None) -> tuple[str | None, str | None]:
    parsed = _parse_qualified(str(column)) if column is not None else None
    if parsed:
        return parsed
    return (str(table) if table else default_table, str(column) if column is not None else None)


def _parse_join_condition(condition: str) -> tuple[str, str, str, str] | None:
    match = re.search(r"([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\s*=\s*([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)", condition)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3), match.group(4)


def _join_condition(
    left_table: Any,
    left_column: Any,
    right_table: Any,
    right_column: Any,
) -> str:
    if all([left_table, left_column, right_table, right_column]):
        return f"{left_table}.{left_column} = {right_table}.{right_column}"
    return ""


def _alias_from_column(column: Any) -> str:
    text = str(column or "column").split(".")[-1].strip().lower()
    return re.sub(r"[^a-z0-9_]+", "_", text).strip("_") or "column"


def _stable_id(payload: dict[str, Any]) -> str:
    return "query_ir_" + _payload_hash(payload)[:16]


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _stable_json_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, sort_keys=True, default=str))


__all__ = [
    "QueryIRCompatibilityError",
    "V1_TO_V2_WARNING",
    "V2_TO_V1_WARNING",
    "coerce_query_ir_v1",
    "convert_v2_to_v1",
    "migrate_v1_to_v2",
]
