from __future__ import annotations

import hashlib
import re
from typing import Any

import sqlglot
from sqlglot import exp

from capabilities.sql_capability_extractor import SQLCapabilityExtractor
from .query_ir_v2_models import (
    AggregationExpression,
    BinaryOperationExpression,
    BooleanPredicate,
    CTEDefinition,
    CaseExpression,
    CaseWhen,
    ColumnExpression,
    ComparisonPredicate,
    ExistsPredicate,
    FromItem,
    FunctionExpression,
    InLiteralPredicate,
    InSubqueryPredicate,
    JoinNode,
    LiteralExpression,
    LiteralValueType,
    NotPredicate,
    NullPredicate,
    BetweenPredicate,
    OrderByItem,
    QueryNode,
    SelectItem,
    SetOperationNode,
    SubqueryExpression,
    UnaryOperationExpression,
    WindowExpression,
    WindowSpecification,
)
from .query_ir_v2_validation import QueryIRV2Validator
from .sql_to_ir_rules import (
    has_case_expression,
    has_complex_having,
    has_nested_query,
    has_set_operation,
    has_window_function,
)


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


class SQLToQueryIRV2Error(ValueError):
    pass


class SQLToQueryIRV2Converter:
    def __init__(
        self,
        *,
        dialect: str = "sqlite",
        max_recursive_depth: int = 32,
        max_predicate_nodes: int = 256,
        max_in_literal_values: int = 100,
    ) -> None:
        self.dialect = dialect or "sqlite"
        self.validator = QueryIRV2Validator(
            max_recursive_depth=max_recursive_depth,
            max_predicate_nodes=max_predicate_nodes,
            max_in_literal_values=max_in_literal_values,
        )

    def convert(
        self,
        sql: str,
        *,
        question: str = "",
        schema: dict[str, Any] | None = None,
    ) -> QueryNode:
        ast = self._parse(sql)
        self._reject_non_goals(ast)
        table_aliases = _table_aliases(ast)
        source_table = _source_from_table(ast)
        where_expr = ast.args.get("where")
        where = self._predicate_from_ast(where_expr.this, table_aliases) if where_expr is not None and where_expr.this is not None else None

        # GROUP BY
        group_by = self._group_by(ast, table_aliases)

        # HAVING
        having_expr = ast.args.get("having")
        having = self._predicate_from_ast(having_expr.this, table_aliases) if having_expr is not None and having_expr.this is not None else None

        # CTEs
        ctes = self._ctes(ast)

        # Set operations
        set_operations = self._set_operations(ast)

        # Determine intent
        has_agg = bool(group_by) or having is not None
        intent = "aggregate" if has_agg else ("show_records" if where is None else "simple_filter")

        annotation = SQLCapabilityExtractor(dialect=self.dialect).extract(sql, schema=schema, full_query_ir_supported=True)
        query = QueryNode(
            query_ir_id="sqlv2_" + hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16],
            question=question,
            normalized_question=" ".join(question.strip().lower().split()),
            intent=intent,
            template_id=intent,
            dialect=self.dialect,
            from_item=FromItem(table=source_table) if source_table else None,
            required_tables=_tables(ast),
            select_items=self._select_items(ast, table_aliases),
            joins=self._joins(ast, table_aliases),
            where=where,
            having=having,
            predicates=[where] if where is not None else [],
            group_by=group_by,
            order_by=self._order_by(ast, table_aliases),
            limit=_limit(ast),
            offset=_offset(ast),
            ctes=ctes,
            set_operations=set_operations,
            select_mode="aggregate" if has_agg else "records",
            capability_metadata={
                "required_capabilities": list(annotation.required_capabilities),
                "unsupported_capabilities": [],
                "renderer_capabilities": ["query_ir_v2_boolean_renderer"],
                "source_capability_labels": list(annotation.required_capabilities),
            },
            metadata={"source_sql": sql, "converter": "SQLToQueryIRV2Converter"},
        )
        validation = self.validator.validate(query)
        if not validation.is_valid:
            raise SQLToQueryIRV2Error("; ".join(validation.errors))
        return query

    def _parse(self, sql: str) -> exp.Expression:
        try:
            return sqlglot.parse_one(sql, read=self.dialect)
        except Exception:
            return sqlglot.parse_one(sql)

    @staticmethod
    def _reject_non_goals(ast: exp.Expression) -> None:
        if not isinstance(ast, exp.Select):
            raise SQLToQueryIRV2Error("Only SELECT statements can be converted to QueryIR v2.")

    def _select_items(self, ast: exp.Expression, table_aliases: dict[str, str]) -> list[SelectItem]:
        items: list[SelectItem] = []
        for expression in getattr(ast, "expressions", []) or []:
            inner = expression.this if isinstance(expression, exp.Alias) and expression.this is not None else expression
            alias = str(expression.alias) if getattr(expression, "alias", None) else None
            if isinstance(inner, exp.Star):
                continue
            rendered = self._expression_from_ast(inner, table_aliases)
            # Determine role from expression type
            if isinstance(rendered, AggregationExpression):
                role = "metric"
            elif isinstance(rendered, WindowExpression):
                role = "metric"
            else:
                role = "dimension"
            actual_alias = alias or (rendered.column if isinstance(rendered, ColumnExpression) else None)
            items.append(SelectItem(role=role, expression=rendered, alias=actual_alias))
        return items

    def _joins(self, ast: exp.Expression, table_aliases: dict[str, str]) -> list[JoinNode]:
        joins: list[JoinNode] = []
        for index, join in enumerate(ast.args.get("joins") or [], start=1):
            if not isinstance(join.this, exp.Table):
                raise SQLToQueryIRV2Error("Only table joins are supported in Phase 2B.")
            on = join.args.get("on")
            joins.append(
                JoinNode(
                    join_type=str(join.args.get("kind") or "INNER").upper(),
                    right=FromItem(table=str(join.this.name)),
                    on=self._predicate_from_ast(on, table_aliases) if on is not None else None,
                    path_order=index,
                )
            )
        return joins

    def _predicate_from_ast(self, node: exp.Expression, table_aliases: dict[str, str]) -> Any:
        node = _unwrap_paren(node)
        if isinstance(node, exp.And):
            return BooleanPredicate(
                operator="AND",
                operands=[
                    self._predicate_from_ast(node.this, table_aliases),
                    self._predicate_from_ast(node.expression, table_aliases),
                ],
            )
        if isinstance(node, exp.Or):
            return BooleanPredicate(
                operator="OR",
                operands=[
                    self._predicate_from_ast(node.this, table_aliases),
                    self._predicate_from_ast(node.expression, table_aliases),
                ],
            )
        if isinstance(node, exp.Not):
            child = _unwrap_paren(node.this)
            negated = self._negated_predicate_from_ast(child, table_aliases)
            return negated if negated is not None else NotPredicate(operand=self._predicate_from_ast(child, table_aliases))
        if isinstance(node, exp.Is):
            right = _unwrap_paren(node.expression)
            if isinstance(right, exp.Null):
                return NullPredicate(expression=self._expression_from_ast(node.this, table_aliases))
        if isinstance(node, exp.In):
            selects = list(node.find_all(exp.Select))
            if selects:
                # IN with subquery
                subquery = node.args.get("query") or node.args.get("unnest")
                if subquery is not None and hasattr(subquery, "this") and isinstance(subquery.this, exp.Select):
                    inner_query = self._convert_select(subquery.this)
                else:
                    # Try to find the Select directly
                    inner_query = self._convert_select(selects[0])
                return InSubqueryPredicate(
                    expression=self._expression_from_ast(node.this, table_aliases),
                    query=inner_query,
                )
            values = [self._literal_from_ast(item) for item in node.expressions]
            return InLiteralPredicate(expression=self._expression_from_ast(node.this, table_aliases), values=values)
        if isinstance(node, exp.Between):
            return BetweenPredicate(
                expression=self._expression_from_ast(node.this, table_aliases),
                lower=self._literal_from_ast(node.args["low"]),
                upper=self._literal_from_ast(node.args["high"]),
            )
        # Exists predicate
        if isinstance(node, exp.Exists):
            inner = node.this
            if isinstance(inner, exp.Subquery) and isinstance(inner.this, exp.Select):
                inner_query = self._convert_select(inner.this)
            elif isinstance(inner, exp.Select):
                inner_query = self._convert_select(inner)
            else:
                raise SQLToQueryIRV2Error(f"Unsupported EXISTS body: {node.sql(dialect=self.dialect)}")
            return ExistsPredicate(query=inner_query)
        for klass, operator in _COMPARISON_OPERATORS.items():
            if isinstance(node, klass):
                return ComparisonPredicate(
                    left=self._expression_from_ast(node.this, table_aliases),
                    operator=operator,
                    right=self._expression_from_ast(node.expression, table_aliases),
                )
        raise SQLToQueryIRV2Error(f"Unsupported WHERE predicate: {node.sql(dialect=self.dialect)}")

    def _negated_predicate_from_ast(self, node: exp.Expression, table_aliases: dict[str, str]) -> Any | None:
        if isinstance(node, exp.Is) and isinstance(_unwrap_paren(node.expression), exp.Null):
            return NullPredicate(expression=self._expression_from_ast(node.this, table_aliases), negated=True)
        if isinstance(node, exp.In):
            if any(True for _ in node.find_all(exp.Select)):
                # NOT IN with subquery
                selects = list(node.find_all(exp.Select))
                inner_query = self._convert_select(selects[0])
                return InSubqueryPredicate(
                    expression=self._expression_from_ast(node.this, table_aliases),
                    query=inner_query,
                    negated=True,
                )
            return InLiteralPredicate(
                expression=self._expression_from_ast(node.this, table_aliases),
                values=[self._literal_from_ast(item) for item in node.expressions],
                negated=True,
            )
        if isinstance(node, exp.Between):
            return BetweenPredicate(
                expression=self._expression_from_ast(node.this, table_aliases),
                lower=self._literal_from_ast(node.args["low"]),
                upper=self._literal_from_ast(node.args["high"]),
                negated=True,
            )
        return None

    def _expression_from_ast(self, node: exp.Expression, table_aliases: dict[str, str]) -> Any:
        node = _unwrap_paren(node)
        if isinstance(node, exp.Column):
            raw_table = str(node.table) if node.table else None
            table = table_aliases.get(raw_table, raw_table) if raw_table else None
            return ColumnExpression(table=table, column=str(node.name))
        if isinstance(node, exp.Literal):
            return self._literal_from_ast(node)
        if isinstance(node, exp.Boolean):
            return LiteralExpression(value=bool(node.this), value_type=LiteralValueType.BOOLEAN)
        if isinstance(node, exp.Null):
            return LiteralExpression(value=None, value_type=LiteralValueType.NULL)

        # Aggregation functions
        if isinstance(node, (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)):
            return self._aggregation_from_ast(node, table_aliases)
        if isinstance(node, exp.AggFunc):
            return self._aggregation_from_ast(node, table_aliases)

        # CASE expression
        if isinstance(node, exp.Case):
            return self._case_from_ast(node, table_aliases)

        # Subquery expression
        if isinstance(node, exp.Subquery):
            inner = node.this
            if isinstance(inner, exp.Select):
                inner_query = self._convert_select(inner)
                return SubqueryExpression(query=inner_query)

        # Window function
        if isinstance(node, exp.Window):
            return self._window_from_ast(node, table_aliases)

        # Binary operations (arithmetic)
        if isinstance(node, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
            op_map = {exp.Add: "+", exp.Sub: "-", exp.Mul: "*", exp.Div: "/"}
            return BinaryOperationExpression(
                operator=op_map.get(type(node), "+"),
                left=self._expression_from_ast(node.this, table_aliases),
                right=self._expression_from_ast(node.expression, table_aliases),
            )

        # Unary negation
        if isinstance(node, exp.Neg):
            return UnaryOperationExpression(
                operator="-",
                operand=self._expression_from_ast(node.this, table_aliases),
            )

        # Generic function call
        if isinstance(node, exp.Func):
            return self._function_from_ast(node, table_aliases)

        # Alias (unwrap)
        if isinstance(node, exp.Alias) and node.this is not None:
            return self._expression_from_ast(node.this, table_aliases)

        return self._literal_from_ast(node)

    def _aggregation_from_ast(self, node: exp.Expression, table_aliases: dict[str, str]) -> AggregationExpression:
        func_name_map = {
            exp.Count: "COUNT", exp.Sum: "SUM", exp.Avg: "AVG",
            exp.Min: "MIN", exp.Max: "MAX",
        }
        func_name = func_name_map.get(type(node), type(node).__name__.upper())
        arg = node.this
        if isinstance(arg, exp.Star) or arg is None:
            return AggregationExpression(function=func_name, argument=None)
        distinct = getattr(node, "args", {}).get("distinct", False)
        return AggregationExpression(
            function=func_name,
            argument=self._expression_from_ast(arg, table_aliases),
            distinct=bool(distinct),
        )

    def _case_from_ast(self, node: exp.Case, table_aliases: dict[str, str]) -> CaseExpression:
        cases: list[CaseWhen] = []
        for if_node in node.args.get("ifs", []):
            when_pred = self._predicate_from_ast(if_node.this, table_aliases)
            then_expr = self._expression_from_ast(if_node.args.get("true", if_node.expression), table_aliases)
            cases.append(CaseWhen(when=when_pred, then=then_expr))
        else_expr = None
        default = node.args.get("default")
        if default is not None:
            else_expr = self._expression_from_ast(default, table_aliases)
        return CaseExpression(cases=cases, else_expression=else_expr)

    def _window_from_ast(self, node: exp.Window, table_aliases: dict[str, str]) -> WindowExpression:
        inner = self._expression_from_ast(node.this, table_aliases)
        partition_by: list[Any] = []
        order_by_items: list[OrderByItem] = []
        for part_expr in node.args.get("partition_by", []) or []:
            partition_by.append(self._expression_from_ast(part_expr, table_aliases))
        order = node.args.get("order")
        if order is not None:
            for item in order.expressions if hasattr(order, "expressions") else [order]:
                direction = "DESC" if getattr(item, "args", {}).get("desc") else "ASC"
                order_by_items.append(OrderByItem(
                    expression=self._expression_from_ast(item.this, table_aliases),
                    direction=direction,
                ))
        return WindowExpression(
            expression=inner,
            window=WindowSpecification(
                partition_by=partition_by,
                order_by=order_by_items,
            ),
        )

    def _function_from_ast(self, node: exp.Func, table_aliases: dict[str, str]) -> FunctionExpression:
        name = type(node).__name__.upper()
        if hasattr(node, "sql_name"):
            name = node.sql_name()
        args = []
        for key in ["this", "expression"]:
            arg = node.args.get(key)
            if arg is not None:
                args.append(self._expression_from_ast(arg, table_aliases))
        for extra in node.args.get("expressions", []) or []:
            args.append(self._expression_from_ast(extra, table_aliases))
        return FunctionExpression(name=name, arguments=args)

    def _group_by(self, ast: exp.Expression, table_aliases: dict[str, str]) -> list[Any]:
        group = ast.args.get("group")
        if group is None:
            return []
        return [
            self._expression_from_ast(expr, table_aliases)
            for expr in (group.expressions if hasattr(group, "expressions") else [group])
        ]

    def _order_by(self, ast: exp.Expression, table_aliases: dict[str, str]) -> list[OrderByItem]:
        order = ast.args.get("order")
        if order is None:
            return []
        items: list[OrderByItem] = []
        for item in order.expressions if hasattr(order, "expressions") else [order]:
            direction = "DESC" if getattr(item, "args", {}).get("desc") else "ASC"
            items.append(OrderByItem(
                expression=self._expression_from_ast(item.this, table_aliases),
                direction=direction,
            ))
        return items

    def _ctes(self, ast: exp.Expression) -> list[CTEDefinition]:
        cte_list: list[CTEDefinition] = []
        with_clause = ast.args.get("with_") or ast.args.get("with")
        if with_clause is None:
            return cte_list
        for cte in with_clause.expressions if hasattr(with_clause, "expressions") else []:
            name = str(cte.alias) if cte.alias else ""
            inner = cte.this
            if isinstance(inner, exp.Select):
                cte_query = self._convert_select(inner)
                cte_list.append(CTEDefinition(name=name, query=cte_query))
        return cte_list

    def _set_operations(self, ast: exp.Expression) -> list[SetOperationNode]:
        ops: list[SetOperationNode] = []
        # sqlglot represents UNION as a Union node wrapping the select
        # Walk the tree for Union/Intersect/Except nodes
        return ops  # Will be extended when needed per specific sqlglot AST

    def _convert_select(self, ast: exp.Select) -> QueryNode:
        """Recursively convert a sub-SELECT AST to a QueryNode."""
        table_aliases = _table_aliases(ast)
        source_table = _source_from_table(ast)
        where_expr = ast.args.get("where")
        where = self._predicate_from_ast(where_expr.this, table_aliases) if where_expr is not None and where_expr.this is not None else None
        group_by = self._group_by(ast, table_aliases)
        having_expr = ast.args.get("having")
        having = self._predicate_from_ast(having_expr.this, table_aliases) if having_expr is not None and having_expr.this is not None else None
        return QueryNode(
            query_ir_id="sub_" + hashlib.sha256(ast.sql().encode("utf-8")).hexdigest()[:16],
            dialect=self.dialect,
            from_item=FromItem(table=source_table) if source_table else None,
            required_tables=_tables(ast),
            select_items=self._select_items(ast, table_aliases),
            joins=self._joins(ast, table_aliases),
            where=where,
            having=having,
            group_by=group_by,
            order_by=self._order_by(ast, table_aliases),
            limit=_limit(ast),
            offset=_offset(ast),
            metadata={"converter": "SQLToQueryIRV2Converter", "is_subquery": True},
        )

    def _literal_from_ast(self, node: exp.Expression) -> LiteralExpression:
        node = _unwrap_paren(node)
        if isinstance(node, exp.Literal):
            if node.is_string:
                text = str(node.this)
                vtype = LiteralValueType.DATE if DATE_RE.match(text) else LiteralValueType.STRING
                return LiteralExpression(value=text, value_type=vtype, source_text=f"'{text}'")
            text = str(node.this)
            try:
                return LiteralExpression(value=int(text), value_type=LiteralValueType.INTEGER, source_text=text)
            except ValueError:
                return LiteralExpression(value=text, value_type=LiteralValueType.DECIMAL, source_text=text)
        if isinstance(node, exp.Boolean):
            return LiteralExpression(value=bool(node.this), value_type=LiteralValueType.BOOLEAN)
        if isinstance(node, exp.Null):
            return LiteralExpression(value=None, value_type=LiteralValueType.NULL)
        raise SQLToQueryIRV2Error(f"Unsupported literal expression: {node.sql(dialect=self.dialect)}")


_COMPARISON_OPERATORS: dict[type[exp.Expression], str] = {
    exp.EQ: "=",
    exp.NEQ: "<>",
    exp.GT: ">",
    exp.GTE: ">=",
    exp.LT: "<",
    exp.LTE: "<=",
    exp.Like: "LIKE",
}


def _unwrap_paren(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren) and node.this is not None:
        node = node.this
    return node


def _tables(ast: exp.Expression) -> list[str]:
    return list(dict.fromkeys(str(table.name) for table in ast.find_all(exp.Table) if table.name))


def _table_aliases(ast: exp.Expression) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for table in ast.find_all(exp.Table):
        name = str(table.name)
        aliases[name] = name
        if table.alias:
            aliases[str(table.alias)] = name
    return aliases


def _source_from_table(ast: exp.Expression) -> str | None:
    from_expr = ast.args.get("from") or ast.args.get("from_")
    table = getattr(from_expr, "this", None)
    return str(table.name) if isinstance(table, exp.Table) else None


def _limit(ast: exp.Expression) -> int | None:
    limit = ast.args.get("limit")
    expression = getattr(limit, "expression", None)
    if expression is None:
        return None
    try:
        return int(expression.name)
    except Exception:
        return None


def _offset(ast: exp.Expression) -> int | None:
    offset = ast.args.get("offset")
    if offset is None:
        return None
    expression = getattr(offset, "expression", None)
    if expression is None:
        return None
    try:
        return int(expression.name)
    except Exception:
        return None


__all__ = ["SQLToQueryIRV2Converter", "SQLToQueryIRV2Error"]
