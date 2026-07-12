from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .query_ir_v2_models import (
    AggregationExpression,
    BetweenPredicate,
    BinaryOperationExpression,
    BooleanOperationExpression,
    BooleanPredicate,
    CTEDefinition,
    CaseExpression,
    ColumnExpression,
    ComparisonPredicate,
    DateFilterNode,
    ExistsPredicate,
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
    UnaryOperationExpression,
    WindowExpression,
)


class QueryIRV2ValidationIssue(BaseModel):
    severity: str
    issue_type: str
    message: str
    path: str | None = None
    suggested_action: str | None = None


class QueryIRV2ValidationResult(BaseModel):
    is_valid: bool
    issues: list[QueryIRV2ValidationIssue] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class QueryIRV2Validator:
    def __init__(
        self,
        max_recursive_depth: int = 32,
        *,
        max_predicate_nodes: int = 256,
        max_in_literal_values: int = 100,
        enforce_renderer_support: bool = False,
    ):
        self.max_recursive_depth = max_recursive_depth
        self.max_predicate_nodes = max_predicate_nodes
        self.max_in_literal_values = max_in_literal_values
        self.enforce_renderer_support = enforce_renderer_support

    def validate(self, query: QueryNode | dict[str, Any]) -> QueryIRV2ValidationResult:
        issues: list[QueryIRV2ValidationIssue] = []
        raw_payload = query if isinstance(query, dict) else None
        if raw_payload and str(raw_payload.get("query_type", "SELECT")).upper() != "SELECT":
            self._issue(
                issues,
                "error",
                "forbidden_mutation_query_type",
                f"QueryIR v2 only permits SELECT query_type, got {raw_payload.get('query_type')!r}.",
                "query_type",
            )

        try:
            query_node = query if isinstance(query, QueryNode) else QueryNode.model_validate(query)
        except ValidationError as exc:
            for error in exc.errors():
                self._issue(
                    issues,
                    "error",
                    "query_ir_v2_model_validation",
                    str(error.get("msg", "QueryIR v2 validation failed.")),
                    ".".join(str(part) for part in error.get("loc", ())) or None,
                )
            return self._result(issues)

        self._validate_aliases(issues, query_node)
        self._validate_depth(issues, query_node)
        self._validate_predicate_limits(issues, query_node)
        self._validate_predicate_semantics(issues, query_node)
        self._validate_references(issues, query_node)
        self._validate_capability_consistency(issues, query_node)
        self._validate_renderer_support(issues, query_node)
        self._validate_having(issues, query_node)
        self._validate_ctes(issues, query_node)
        self._validate_set_operations(issues, query_node)
        return self._result(issues)

    def _validate_aliases(self, issues: list[QueryIRV2ValidationIssue], query: QueryNode) -> None:
        aliases = [item.alias for item in query.select_items if item.alias]
        duplicates = sorted({alias for alias in aliases if aliases.count(alias) > 1})
        for alias in duplicates:
            self._issue(
                issues,
                "error",
                "duplicate_select_alias",
                f"Select alias is not unique: {alias}.",
                "select_items",
            )

    def _validate_depth(self, issues: list[QueryIRV2ValidationIssue], query: QueryNode) -> None:
        depth = self._query_depth(query)
        if depth > self.max_recursive_depth:
            self._issue(
                issues,
                "error",
                "recursive_depth_exceeded",
                f"QueryIR v2 recursive depth {depth} exceeds max {self.max_recursive_depth}.",
                None,
                "Reduce expression nesting or increase the configured limit for trusted payloads.",
            )

    def _validate_references(self, issues: list[QueryIRV2ValidationIssue], query: QueryNode) -> None:
        if query.from_item is not None and query.from_item.from_type == "TABLE" and not query.from_item.table:
            self._issue(issues, "error", "missing_from_table", "TABLE from_item requires a table.", "from_item.table")
        for path, expression in self._iter_expressions(query):
            if isinstance(expression, ColumnExpression):
                if not expression.column:
                    self._issue(issues, "error", "missing_column_reference", "Column expression requires column.", path)
                if expression.table is not None and not expression.table.strip():
                    self._issue(issues, "error", "invalid_table_reference", "Column expression table is blank.", path)
        if query.limit is not None and query.limit < 0:
            self._issue(issues, "error", "invalid_limit", "Limit must be non-negative.", "limit")
        if query.offset is not None and query.offset < 0:
            self._issue(issues, "error", "invalid_offset", "Offset must be non-negative.", "offset")

    def _validate_capability_consistency(self, issues: list[QueryIRV2ValidationIssue], query: QueryNode) -> None:
        required = set(query.capability_metadata.required_capabilities)
        unsupported = set(query.capability_metadata.unsupported_capabilities)
        for capability in sorted(required & unsupported):
            self._issue(
                issues,
                "error",
                "capability_metadata_conflict",
                f"Capability is both required and unsupported: {capability}.",
                "capability_metadata",
            )
        labels = {str(item).upper() for item in query.capability_metadata.source_capability_labels}
        labels.update(str(item).upper() for item in query.capability_metadata.required_capabilities)
        has_or_label = "OR_FILTER" in labels
        has_or_tree = _predicate_tree_contains_or(query.where) or any(
            _predicate_tree_contains_or(predicate) for predicate in query.predicates
        )
        if has_or_label != has_or_tree:
            self._issue(
                issues,
                "error",
                "or_filter_capability_mismatch",
                "OR_FILTER capability label must match the v2 predicate tree.",
                "capability_metadata",
            )

    def _validate_renderer_support(self, issues: list[QueryIRV2ValidationIssue], query: QueryNode) -> None:
        if not self.enforce_renderer_support:
            return
        for capability, path in unsupported_v2_rendering_capabilities(query):
            self._issue(
                issues,
                "error",
                "unsupported_v2_rendering_capability",
                f"Current renderer does not support QueryIR v2 capability: {capability}.",
                path,
            )

    def _query_depth(self, query: QueryNode) -> int:
        depths = [1]
        if query.from_item and query.from_item.query:
            depths.append(1 + self._query_depth(query.from_item.query))
        for item in query.select_items:
            depths.append(1 + self._expression_depth(item.expression))
        if query.where is not None:
            depths.append(1 + self._predicate_depth(query.where))
        for predicate in query.predicates:
            depths.append(1 + self._predicate_depth(predicate))
        for item in query.date_filters:
            depths.append(1 + self._expression_depth(item.date_expression))
        for expression in query.group_by:
            depths.append(1 + self._expression_depth(expression))
        for item in query.order_by:
            depths.append(1 + self._expression_depth(item.expression))
        for join in query.joins:
            if join.on is not None:
                depths.append(1 + self._predicate_depth(join.on))
            if join.right.query is not None:
                depths.append(1 + self._query_depth(join.right.query))
        for operation in query.set_operations:
            depths.append(1 + self._query_depth(operation.query))
        return max(depths)

    def _expression_depth(self, expression: Any) -> int:
        if isinstance(expression, (ColumnExpression, LiteralExpression)):
            return 1
        if isinstance(expression, FunctionExpression):
            return 1 + max([self._expression_depth(item) for item in expression.arguments] or [0])
        if isinstance(expression, AggregationExpression):
            return 1 + (self._expression_depth(expression.argument) if expression.argument is not None else 0)
        if isinstance(expression, BinaryOperationExpression):
            return 1 + max(self._expression_depth(expression.left), self._expression_depth(expression.right))
        if isinstance(expression, UnaryOperationExpression):
            return 1 + self._expression_depth(expression.operand)
        if isinstance(expression, BooleanOperationExpression):
            return 1 + max([self._expression_depth(item) for item in expression.operands] or [0])
        if isinstance(expression, CaseExpression):
            case_depths = [
                max(self._predicate_depth(item.when), self._expression_depth(item.then))
                for item in expression.cases
            ]
            if expression.else_expression is not None:
                case_depths.append(self._expression_depth(expression.else_expression))
            return 1 + max(case_depths or [0])
        if isinstance(expression, SubqueryExpression):
            return 1 + self._query_depth(expression.query)
        if isinstance(expression, WindowExpression):
            return 1 + max(
                [self._expression_depth(expression.expression)]
                + [self._expression_depth(item) for item in expression.window.partition_by]
                + [self._expression_depth(item.expression) for item in expression.window.order_by]
            )
        return 1

    def _predicate_depth(self, predicate: Any) -> int:
        if isinstance(predicate, ComparisonPredicate):
            return 1 + max(self._expression_depth(predicate.left), self._expression_depth(predicate.right))
        if isinstance(predicate, InLiteralPredicate):
            return 1 + max(
                [self._expression_depth(predicate.expression)]
                + [self._expression_depth(item) for item in predicate.values]
            )
        if isinstance(predicate, BetweenPredicate):
            return 1 + max(
                self._expression_depth(predicate.expression),
                self._expression_depth(predicate.lower),
                self._expression_depth(predicate.upper),
            )
        if isinstance(predicate, NullPredicate):
            return 1 + self._expression_depth(predicate.expression)
        if isinstance(predicate, BooleanPredicate):
            return 1 + max([self._predicate_depth(item) for item in predicate.operands] or [0])
        if isinstance(predicate, NotPredicate):
            return 1 + self._predicate_depth(predicate.operand)
        return 1

    def _validate_predicate_limits(self, issues: list[QueryIRV2ValidationIssue], query: QueryNode) -> None:
        predicate_roots = self._predicate_roots(query)
        node_count = sum(self._predicate_node_count(predicate) for _, predicate in predicate_roots)
        if node_count > self.max_predicate_nodes:
            self._issue(
                issues,
                "error",
                "predicate_node_count_exceeded",
                f"QueryIR v2 predicate node count {node_count} exceeds max {self.max_predicate_nodes}.",
                "where",
            )
        for path, predicate in predicate_roots:
            for in_path, in_predicate in self._iter_in_predicates(predicate, path):
                if len(in_predicate.values) > self.max_in_literal_values:
                    self._issue(
                        issues,
                        "error",
                        "in_literal_list_too_large",
                        f"IN literal list has {len(in_predicate.values)} values; max is {self.max_in_literal_values}.",
                        in_path,
                    )

    def _validate_predicate_semantics(self, issues: list[QueryIRV2ValidationIssue], query: QueryNode) -> None:
        for path, predicate in self._predicate_roots(query):
            self._validate_predicate_node(issues, predicate, path)

    def _validate_predicate_node(
        self,
        issues: list[QueryIRV2ValidationIssue],
        predicate: Any,
        path: str,
    ) -> None:
        if isinstance(predicate, ComparisonPredicate):
            if not _compatible_comparison(predicate.left, predicate.operator, predicate.right):
                self._issue(
                    issues,
                    "error",
                    "incompatible_comparison_operands",
                    "Comparison predicate operands are not type-compatible.",
                    path,
                )
            return
        if isinstance(predicate, InLiteralPredicate):
            if not predicate.values:
                self._issue(issues, "error", "empty_in_literal_list", "IN literal predicate requires values.", path)
            if not _compatible_literal_list(predicate.values):
                self._issue(
                    issues,
                    "error",
                    "incompatible_in_literal_values",
                    "IN literal predicate values must be type-compatible.",
                    path,
                )
            return
        if isinstance(predicate, BetweenPredicate):
            if not _compatible_between(predicate.expression, predicate.lower, predicate.upper):
                self._issue(
                    issues,
                    "error",
                    "incompatible_between_bounds",
                    "BETWEEN predicate expression and bounds are not type-compatible.",
                    path,
                )
            return
        if isinstance(predicate, NullPredicate):
            if isinstance(predicate.expression, LiteralExpression):
                self._issue(
                    issues,
                    "error",
                    "invalid_null_predicate_operand",
                    "NULL predicate requires a non-literal expression.",
                    path,
                )
            return
        if isinstance(predicate, BooleanPredicate):
            for idx, operand in enumerate(predicate.operands):
                self._validate_predicate_node(issues, operand, f"{path}.operands.{idx}")
            return
        if isinstance(predicate, NotPredicate):
            self._validate_predicate_node(issues, predicate.operand, f"{path}.operand")

    def _predicate_roots(self, query: QueryNode) -> list[tuple[str, Any]]:
        roots: list[tuple[str, Any]] = []
        if query.where is not None:
            roots.append(("where", query.where))
        if query.having is not None:
            roots.append(("having", query.having))
        roots.extend((f"predicates.{idx}", predicate) for idx, predicate in enumerate(query.predicates))
        return roots

    def _predicate_node_count(self, predicate: Any) -> int:
        if isinstance(predicate, BooleanPredicate):
            return 1 + sum(self._predicate_node_count(item) for item in predicate.operands)
        if isinstance(predicate, NotPredicate):
            return 1 + self._predicate_node_count(predicate.operand)
        return 1

    def _iter_in_predicates(self, predicate: Any, path: str) -> list[tuple[str, InLiteralPredicate]]:
        if isinstance(predicate, InLiteralPredicate):
            return [(path, predicate)]
        if isinstance(predicate, BooleanPredicate):
            values: list[tuple[str, InLiteralPredicate]] = []
            for idx, operand in enumerate(predicate.operands):
                values.extend(self._iter_in_predicates(operand, f"{path}.operands.{idx}"))
            return values
        if isinstance(predicate, NotPredicate):
            return self._iter_in_predicates(predicate.operand, f"{path}.operand")
        return []

    def _iter_expressions(self, query: QueryNode) -> list[tuple[str, Any]]:
        expressions: list[tuple[str, Any]] = []
        for idx, item in enumerate(query.select_items):
            expressions.extend(self._walk_expression(f"select_items.{idx}.expression", item.expression))
        if query.where is not None:
            expressions.extend(self._walk_predicate("where", query.where))
        if query.having is not None:
            expressions.extend(self._walk_predicate("having", query.having))
        for idx, predicate in enumerate(query.predicates):
            expressions.extend(self._walk_predicate(f"predicates.{idx}", predicate))
        for idx, item in enumerate(query.date_filters):
            expressions.extend(self._walk_expression(f"date_filters.{idx}.date_expression", item.date_expression))
        for idx, expression in enumerate(query.group_by):
            expressions.extend(self._walk_expression(f"group_by.{idx}", expression))
        for idx, item in enumerate(query.order_by):
            expressions.extend(self._walk_expression(f"order_by.{idx}.expression", item.expression))
        for idx, join in enumerate(query.joins):
            if join.on is not None:
                expressions.extend(self._walk_predicate(f"joins.{idx}.on", join.on))
        for idx, cte in enumerate(query.ctes):
            expressions.extend(self._iter_expressions(cte.query))
        return expressions

    def _walk_expression(self, path: str, expression: Any) -> list[tuple[str, Any]]:
        values = [(path, expression)]
        if isinstance(expression, FunctionExpression):
            for idx, item in enumerate(expression.arguments):
                values.extend(self._walk_expression(f"{path}.arguments.{idx}", item))
        elif isinstance(expression, AggregationExpression) and expression.argument is not None:
            values.extend(self._walk_expression(f"{path}.argument", expression.argument))
        elif isinstance(expression, BinaryOperationExpression):
            values.extend(self._walk_expression(f"{path}.left", expression.left))
            values.extend(self._walk_expression(f"{path}.right", expression.right))
        elif isinstance(expression, UnaryOperationExpression):
            values.extend(self._walk_expression(f"{path}.operand", expression.operand))
        elif isinstance(expression, BooleanOperationExpression):
            for idx, item in enumerate(expression.operands):
                values.extend(self._walk_expression(f"{path}.operands.{idx}", item))
        elif isinstance(expression, CaseExpression):
            for idx, item in enumerate(expression.cases):
                values.extend(self._walk_predicate(f"{path}.cases.{idx}.when", item.when))
                values.extend(self._walk_expression(f"{path}.cases.{idx}.then", item.then))
            if expression.else_expression is not None:
                values.extend(self._walk_expression(f"{path}.else_expression", expression.else_expression))
        elif isinstance(expression, SubqueryExpression):
            values.extend(self._iter_expressions(expression.query))
        elif isinstance(expression, WindowExpression):
            values.extend(self._walk_expression(f"{path}.expression", expression.expression))
            for idx, item in enumerate(expression.window.partition_by):
                values.extend(self._walk_expression(f"{path}.window.partition_by.{idx}", item))
            for idx, item in enumerate(expression.window.order_by):
                values.extend(self._walk_expression(f"{path}.window.order_by.{idx}.expression", item.expression))
        return values

    def _walk_predicate(self, path: str, predicate: Any) -> list[tuple[str, Any]]:
        values: list[tuple[str, Any]] = []
        if isinstance(predicate, ComparisonPredicate):
            values.extend(self._walk_expression(f"{path}.left", predicate.left))
            values.extend(self._walk_expression(f"{path}.right", predicate.right))
        elif isinstance(predicate, InLiteralPredicate):
            values.extend(self._walk_expression(f"{path}.expression", predicate.expression))
            for idx, item in enumerate(predicate.values):
                values.extend(self._walk_expression(f"{path}.values.{idx}", item))
        elif isinstance(predicate, BetweenPredicate):
            values.extend(self._walk_expression(f"{path}.expression", predicate.expression))
            values.extend(self._walk_expression(f"{path}.lower", predicate.lower))
            values.extend(self._walk_expression(f"{path}.upper", predicate.upper))
        elif isinstance(predicate, NullPredicate):
            values.extend(self._walk_expression(f"{path}.expression", predicate.expression))
        elif isinstance(predicate, BooleanPredicate):
            for idx, item in enumerate(predicate.operands):
                values.extend(self._walk_predicate(f"{path}.operands.{idx}", item))
        elif isinstance(predicate, NotPredicate):
            values.extend(self._walk_predicate(f"{path}.operand", predicate.operand))
        elif isinstance(predicate, InSubqueryPredicate):
            values.extend(self._walk_expression(f"{path}.expression", predicate.expression))
            values.extend(self._iter_expressions(predicate.query))
        elif isinstance(predicate, ExistsPredicate):
            values.extend(self._iter_expressions(predicate.query))
        return values

    # ── Advanced construct validation ─────────────────────────────────

    def _validate_having(self, issues: list[QueryIRV2ValidationIssue], query: QueryNode) -> None:
        """HAVING requires GROUP BY and must reference at least one aggregate."""
        if query.having is None:
            return
        if not query.group_by:
            self._issue(
                issues,
                "warning",
                "having_without_group_by",
                "HAVING is present but no GROUP BY is defined.",
                "having",
            )
        # Verify HAVING references at least one aggregate function
        if not _predicate_contains_aggregate(query.having):
            self._issue(
                issues,
                "warning",
                "having_without_aggregate",
                "HAVING predicate does not reference any aggregate function. "
                "Consider moving to WHERE.",
                "having",
            )

    def _validate_ctes(self, issues: list[QueryIRV2ValidationIssue], query: QueryNode) -> None:
        """CTE validation: duplicates, undefined refs, dependency ordering, cycles."""
        if not query.ctes:
            return

        # Check for duplicate CTE names
        names = [cte.name.lower() for cte in query.ctes]
        seen: set[str] = set()
        for name in names:
            if name in seen:
                self._issue(
                    issues,
                    "error",
                    "duplicate_cte_name",
                    f"CTE name '{name}' is defined more than once.",
                    "ctes",
                )
            seen.add(name)

        # Check for recursive CTEs (not supported)
        for idx, cte in enumerate(query.ctes):
            refs = _collect_table_refs(cte.query)
            if cte.name.lower() in {r.lower() for r in refs}:
                self._issue(
                    issues,
                    "error",
                    "recursive_cte_not_supported",
                    f"CTE '{cte.name}' references itself (recursive CTEs not supported).",
                    f"ctes.{idx}",
                )

        # Recursively validate CTE bodies
        for idx, cte in enumerate(query.ctes):
            self._validate_aliases(issues, cte.query)
            self._validate_predicate_semantics(issues, cte.query)

    def _validate_set_operations(self, issues: list[QueryIRV2ValidationIssue], query: QueryNode) -> None:
        """Set operation validation: arity equality."""
        if not query.set_operations:
            return
        base_arity = len(query.select_items)
        for idx, set_op in enumerate(query.set_operations):
            branch_arity = len(set_op.query.select_items)
            if base_arity > 0 and branch_arity > 0 and base_arity != branch_arity:
                self._issue(
                    issues,
                    "error",
                    "set_operation_arity_mismatch",
                    f"Set operation branch {idx} has {branch_arity} columns, "
                    f"base query has {base_arity}.",
                    f"set_operations.{idx}",
                )

    @staticmethod
    def _issue(
        issues: list[QueryIRV2ValidationIssue],
        severity: str,
        issue_type: str,
        message: str,
        path: str | None = None,
        suggested_action: str | None = None,
    ) -> None:
        issues.append(
            QueryIRV2ValidationIssue(
                severity=severity,
                issue_type=issue_type,
                message=message,
                path=path,
                suggested_action=suggested_action,
            )
        )

    @staticmethod
    def _result(issues: list[QueryIRV2ValidationIssue]) -> QueryIRV2ValidationResult:
        errors = [issue.message for issue in issues if issue.severity == "error"]
        warnings = [issue.message for issue in issues if issue.severity == "warning"]
        return QueryIRV2ValidationResult(is_valid=not errors, issues=issues, errors=errors, warnings=warnings)


def unsupported_v2_rendering_capabilities(query: QueryNode) -> list[tuple[str, str]]:
    capabilities: list[tuple[str, str]] = []
    if query.set_operations:
        capabilities.append(("set_operation", "set_operations"))
    if query.from_item and query.from_item.from_type == "SUBQUERY":
        capabilities.append(("subquery_from_item", "from_item"))
    for idx, operation in enumerate(query.set_operations):
        capabilities.extend((cap, f"set_operations.{idx}.{path}") for cap, path in unsupported_v2_rendering_capabilities(operation.query))
    for idx, item in enumerate(query.select_items):
        capabilities.extend(_unsupported_expression_capabilities(item.expression, f"select_items.{idx}.expression"))
    if query.where is not None:
        capabilities.extend(_unsupported_predicate_capabilities(query.where, "where"))
    for idx, predicate in enumerate(query.predicates):
        capabilities.extend(_unsupported_predicate_capabilities(predicate, f"predicates.{idx}"))
    for idx, item in enumerate(query.date_filters):
        capabilities.extend(_unsupported_expression_capabilities(item.date_expression, f"date_filters.{idx}.date_expression"))
    for idx, expression in enumerate(query.group_by):
        capabilities.extend(_unsupported_expression_capabilities(expression, f"group_by.{idx}"))
    for idx, item in enumerate(query.order_by):
        capabilities.extend(_unsupported_expression_capabilities(item.expression, f"order_by.{idx}.expression"))
    for idx, join in enumerate(query.joins):
        if join.right.from_type == "SUBQUERY":
            capabilities.append(("subquery_join_item", f"joins.{idx}.right"))
        if join.on is not None:
            capabilities.extend(_unsupported_predicate_capabilities(join.on, f"joins.{idx}.on"))
    return list(dict.fromkeys(capabilities))


def _unsupported_expression_capabilities(expression: Any, path: str) -> list[tuple[str, str]]:
    capabilities: list[tuple[str, str]] = []
    if isinstance(expression, (ColumnExpression, LiteralExpression)):
        return capabilities
    if isinstance(expression, FunctionExpression):
        if expression.name.upper() != "DATE_GRAIN":
            capabilities.append(("function_expression", path))
        for idx, item in enumerate(expression.arguments):
            capabilities.extend(_unsupported_expression_capabilities(item, f"{path}.arguments.{idx}"))
    elif isinstance(expression, AggregationExpression):
        if expression.argument is not None:
            capabilities.extend(_unsupported_expression_capabilities(expression.argument, f"{path}.argument"))
    elif isinstance(expression, BinaryOperationExpression):
        if expression.operator not in {"*", "+", "-", "/"}:
            capabilities.append(("binary_operation_expression", path))
        capabilities.extend(_unsupported_expression_capabilities(expression.left, f"{path}.left"))
        capabilities.extend(_unsupported_expression_capabilities(expression.right, f"{path}.right"))
    elif isinstance(expression, UnaryOperationExpression):
        capabilities.append(("unary_operation_expression", path))
        capabilities.extend(_unsupported_expression_capabilities(expression.operand, f"{path}.operand"))
    elif isinstance(expression, BooleanOperationExpression):
        capabilities.append(("boolean_operation_expression", path))
    elif isinstance(expression, CaseExpression):
        capabilities.append(("case_expression", path))
    elif isinstance(expression, SubqueryExpression):
        capabilities.append(("subquery_expression", path))
    elif isinstance(expression, WindowExpression):
        capabilities.append(("window_expression", path))
    return capabilities


def _unsupported_predicate_capabilities(predicate: Any, path: str) -> list[tuple[str, str]]:
    capabilities: list[tuple[str, str]] = []
    if isinstance(predicate, ComparisonPredicate):
        capabilities.extend(_unsupported_expression_capabilities(predicate.left, f"{path}.left"))
        capabilities.extend(_unsupported_expression_capabilities(predicate.right, f"{path}.right"))
        if predicate.operator.upper() in {"OR"}:
            capabilities.append(("advanced_predicate_operator", path))
    elif isinstance(predicate, InLiteralPredicate):
        capabilities.extend(_unsupported_expression_capabilities(predicate.expression, f"{path}.expression"))
        for idx, item in enumerate(predicate.values):
            capabilities.extend(_unsupported_expression_capabilities(item, f"{path}.values.{idx}"))
    elif isinstance(predicate, BetweenPredicate):
        capabilities.append(("between_predicate", path))
    elif isinstance(predicate, NullPredicate):
        capabilities.append(("null_predicate", path))
    elif isinstance(predicate, BooleanPredicate):
        if predicate.operator != "AND":
            capabilities.append(("boolean_or_not_predicate", path))
        for idx, item in enumerate(predicate.operands):
            capabilities.extend(_unsupported_predicate_capabilities(item, f"{path}.operands.{idx}"))
    elif isinstance(predicate, NotPredicate):
        capabilities.append(("boolean_or_not_predicate", path))
        capabilities.extend(_unsupported_predicate_capabilities(predicate.operand, f"{path}.operand"))
    return capabilities


def _compatible_comparison(left: Any, operator: str, right: Any) -> bool:
    if not isinstance(right, LiteralExpression):
        return True
    value_type = _literal_family(right)
    if value_type == "null":
        return operator in {"=", "!=", "<>"}
    if value_type in {"number", "date"}:
        return operator in {"=", "!=", "<>", ">", ">=", "<", "<="}
    if value_type == "string":
        return operator in {"=", "!=", "<>", "LIKE", "ILIKE"}
    return True


def _compatible_between(expression: Any, lower: Any, upper: Any) -> bool:
    if not isinstance(lower, LiteralExpression) or not isinstance(upper, LiteralExpression):
        return True
    return _literal_family(lower) == _literal_family(upper)


def _compatible_literal_list(values: list[LiteralExpression]) -> bool:
    families = {_literal_family(value) for value in values}
    families.discard("null")
    return len(families) <= 1


def _literal_family(value: LiteralExpression) -> str:
    # Handle LiteralValueType enum: .value gives the raw string (e.g. "STRING")
    raw_type = value.value_type
    if hasattr(raw_type, "value"):
        value_type = str(raw_type.value).strip().lower()
    else:
        value_type = str(raw_type or "").strip().lower()
    if value.value is None or value_type in {"null", "none"}:
        return "null"
    if value_type in {"int", "integer", "float", "real", "number", "numeric", "decimal"}:
        return "number"
    if value_type in {"date", "datetime", "timestamp", "timestamp_with_timezone", "time"}:
        return "date"
    if isinstance(value.value, (int, float)) and not isinstance(value.value, bool):
        return "number"
    if isinstance(value.value, bool):
        return "boolean"
    return "string"


def _predicate_tree_contains_or(predicate: Any) -> bool:
    if predicate is None:
        return False
    if isinstance(predicate, BooleanPredicate):
        return predicate.operator == "OR" or any(_predicate_tree_contains_or(item) for item in predicate.operands)
    if isinstance(predicate, NotPredicate):
        return _predicate_tree_contains_or(predicate.operand)
    return False


def _predicate_contains_aggregate(predicate: Any) -> bool:
    """Check if a predicate tree references at least one aggregate function."""
    if isinstance(predicate, ComparisonPredicate):
        return _expression_contains_aggregate(predicate.left) or _expression_contains_aggregate(predicate.right)
    if isinstance(predicate, BooleanPredicate):
        return any(_predicate_contains_aggregate(item) for item in predicate.operands)
    if isinstance(predicate, NotPredicate):
        return _predicate_contains_aggregate(predicate.operand)
    if isinstance(predicate, InLiteralPredicate):
        return _expression_contains_aggregate(predicate.expression)
    if isinstance(predicate, BetweenPredicate):
        return (
            _expression_contains_aggregate(predicate.expression)
            or _expression_contains_aggregate(predicate.lower)
            or _expression_contains_aggregate(predicate.upper)
        )
    if isinstance(predicate, NullPredicate):
        return _expression_contains_aggregate(predicate.expression)
    return False


def _expression_contains_aggregate(expression: Any) -> bool:
    """Check if an expression tree contains an aggregate function."""
    if isinstance(expression, AggregationExpression):
        return True
    if isinstance(expression, BinaryOperationExpression):
        return _expression_contains_aggregate(expression.left) or _expression_contains_aggregate(expression.right)
    if isinstance(expression, UnaryOperationExpression):
        return _expression_contains_aggregate(expression.operand)
    if isinstance(expression, FunctionExpression):
        return any(_expression_contains_aggregate(arg) for arg in expression.arguments)
    if isinstance(expression, CaseExpression):
        for case_when in expression.cases:
            if _expression_contains_aggregate(case_when.then):
                return True
        if expression.else_expression is not None and _expression_contains_aggregate(expression.else_expression):
            return True
    return False


def _collect_table_refs(query: QueryNode) -> list[str]:
    """Collect all table names referenced in a QueryNode (for CTE cycle detection)."""
    refs: list[str] = []
    if query.from_item is not None and query.from_item.from_type == "TABLE" and query.from_item.table:
        refs.append(query.from_item.table)
    for join in query.joins:
        if join.right.from_type == "TABLE" and join.right.table:
            refs.append(join.right.table)
    return refs


__all__ = [
    "QueryIRV2ValidationIssue",
    "QueryIRV2ValidationResult",
    "QueryIRV2Validator",
    "unsupported_v2_rendering_capabilities",
]

