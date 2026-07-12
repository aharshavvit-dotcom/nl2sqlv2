"""QueryIR v2 scope analysis — derived from QueryIR structure, never persisted.

All scope-related information (correlated references, subquery depth, node count)
is computed by the analyzer and returned as a separate QueryAnalysis object.
Policy/configuration (max_subquery_depth, max_node_count) stays in the analyzer,
NOT in QueryIR data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .query_ir_v2_models import (
    AggregationExpression,
    BooleanPredicate,
    CaseExpression,
    ColumnExpression,
    ExistsPredicate,
    Expression,
    FromItem,
    InSubqueryPredicate,
    NotPredicate,
    Predicate,
    QueryNode,
    SelectItem,
    SubqueryExpression,
    WindowExpression,
)


@dataclass
class BoundColumnReference:
    """A column reference resolved to its owning scope."""
    table: str | None
    column: str
    scope_depth: int
    is_correlated: bool
    outer_scope_depth: int | None = None


@dataclass
class RelationBinding:
    """A relation (table/CTE/subquery) available in a scope."""
    alias: str
    source_type: str  # PHYSICAL_TABLE, CTE, DERIVED_TABLE, SET_RESULT
    output_columns: list[str] = field(default_factory=list)


@dataclass
class ScopeDiagnostic:
    severity: str  # error, warning, info
    code: str
    message: str
    path: str | None = None


@dataclass
class ScopeGraph:
    """Scope hierarchy for a QueryIR tree."""
    scopes: list[ScopeNode] = field(default_factory=list)


@dataclass
class ScopeNode:
    depth: int
    tables: list[str] = field(default_factory=list)
    cte_names: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    children: list[ScopeNode] = field(default_factory=list)
    relation_bindings: list[RelationBinding] = field(default_factory=list)


@dataclass
class QueryAnalysis:
    """Computed by QueryScopeAnalyzer from QueryIR structure — never persisted in QueryIR.

    Attributes:
        query_ir_fingerprint: Hash tying this analysis to the exact QueryIR that produced it.
        correlated_references: Column references that reach into an outer scope.
        unresolved_references: References that could not be resolved to any scope.
        subquery_depth: Maximum subquery nesting depth found.
        expression_depth: Maximum expression nesting depth found.
        node_count: Total AST nodes counted.
        scope_graph: Full scope hierarchy.
        diagnostics: Warnings and errors found during analysis.
        relation_bindings: All derived relation bindings (CTEs, subqueries, etc.).
    """
    query_ir_fingerprint: str = ""
    correlated_references: list[BoundColumnReference] = field(default_factory=list)
    unresolved_references: list[BoundColumnReference] = field(default_factory=list)
    subquery_depth: int = 0
    expression_depth: int = 0
    node_count: int = 0
    scope_graph: ScopeGraph = field(default_factory=ScopeGraph)
    diagnostics: list[ScopeDiagnostic] = field(default_factory=list)
    relation_bindings: list[RelationBinding] = field(default_factory=list)


class QueryScopeAnalyzer:
    """Walks QueryIR tree to compute QueryAnalysis.

    Policy parameters (max_subquery_depth, max_node_count) are analyzer config,
    not query data.
    """

    def __init__(
        self,
        *,
        max_subquery_depth: int = 8,
        max_node_count: int = 256,
        max_expression_depth: int = 64,
    ) -> None:
        self.max_subquery_depth = max_subquery_depth
        self.max_node_count = max_node_count
        self.max_expression_depth = max_expression_depth

    def analyze(self, query: QueryNode) -> QueryAnalysis:
        """Analyze a QueryIR tree and return derived scope information."""
        analysis = QueryAnalysis()
        root_scope = ScopeNode(depth=0)

        self._analyze_query(query, root_scope, analysis, depth=0)

        analysis.scope_graph = ScopeGraph(scopes=[root_scope])

        # Policy diagnostics
        if analysis.subquery_depth > self.max_subquery_depth:
            analysis.diagnostics.append(ScopeDiagnostic(
                severity="error",
                code="max_subquery_depth_exceeded",
                message=f"Subquery depth {analysis.subquery_depth} exceeds limit {self.max_subquery_depth}",
            ))
        if analysis.node_count > self.max_node_count:
            analysis.diagnostics.append(ScopeDiagnostic(
                severity="error",
                code="max_node_count_exceeded",
                message=f"Node count {analysis.node_count} exceeds limit {self.max_node_count}",
            ))
        if analysis.expression_depth > self.max_expression_depth:
            analysis.diagnostics.append(ScopeDiagnostic(
                severity="error",
                code="max_expression_depth_exceeded",
                message=f"Expression depth {analysis.expression_depth} exceeds limit {self.max_expression_depth}",
            ))

        return analysis

    def _analyze_query(
        self,
        query: QueryNode,
        scope: ScopeNode,
        analysis: QueryAnalysis,
        depth: int,
    ) -> None:
        analysis.node_count += 1

        # Register CTE bindings
        for cte in query.ctes:
            binding = RelationBinding(
                alias=cte.name,
                source_type="CTE",
                output_columns=list(cte.columns),
            )
            scope.relation_bindings.append(binding)
            scope.cte_names.append(cte.name)
            analysis.relation_bindings.append(binding)
            # Analyze CTE body in its own child scope
            cte_scope = ScopeNode(depth=depth + 1)
            scope.children.append(cte_scope)
            self._analyze_query(cte.query, cte_scope, analysis, depth + 1)
            analysis.subquery_depth = max(analysis.subquery_depth, depth + 1)

        # Register FROM tables/subqueries
        if query.from_item is not None:
            if query.from_item.from_type == "TABLE" and query.from_item.table:
                scope.tables.append(query.from_item.table)
                scope.relation_bindings.append(RelationBinding(
                    alias=query.from_item.alias or query.from_item.table,
                    source_type="PHYSICAL_TABLE",
                ))
            elif query.from_item.from_type == "SUBQUERY" and query.from_item.query is not None:
                alias = query.from_item.alias or f"_derived_{depth}"
                scope.relation_bindings.append(RelationBinding(
                    alias=alias,
                    source_type="DERIVED_TABLE",
                ))
                sub_scope = ScopeNode(depth=depth + 1)
                scope.children.append(sub_scope)
                self._analyze_query(query.from_item.query, sub_scope, analysis, depth + 1)
                analysis.subquery_depth = max(analysis.subquery_depth, depth + 1)

        # Register JOIN tables
        for join in query.joins:
            if join.right.from_type == "TABLE" and join.right.table:
                scope.tables.append(join.right.table)
            elif join.right.from_type == "SUBQUERY" and join.right.query is not None:
                sub_scope = ScopeNode(depth=depth + 1)
                scope.children.append(sub_scope)
                self._analyze_query(join.right.query, sub_scope, analysis, depth + 1)
                analysis.subquery_depth = max(analysis.subquery_depth, depth + 1)
            if join.on is not None:
                self._analyze_predicate(join.on, scope, analysis, depth, expr_depth=0)

        # Analyze SELECT expressions
        for item in query.select_items:
            self._analyze_expression(item.expression, scope, analysis, depth, expr_depth=0)

        # Analyze WHERE
        if query.where is not None:
            self._analyze_predicate(query.where, scope, analysis, depth, expr_depth=0)

        # Analyze HAVING
        if query.having is not None:
            self._analyze_predicate(query.having, scope, analysis, depth, expr_depth=0)

        # Analyze GROUP BY
        for expr in query.group_by:
            self._analyze_expression(expr, scope, analysis, depth, expr_depth=0)

        # Analyze ORDER BY
        for item in query.order_by:
            self._analyze_expression(item.expression, scope, analysis, depth, expr_depth=0)

        # Analyze set operations
        for set_op in query.set_operations:
            set_scope = ScopeNode(depth=depth + 1)
            scope.children.append(set_scope)
            self._analyze_query(set_op.query, set_scope, analysis, depth + 1)
            analysis.subquery_depth = max(analysis.subquery_depth, depth + 1)

    def _analyze_expression(
        self,
        expr: Any,
        scope: ScopeNode,
        analysis: QueryAnalysis,
        depth: int,
        expr_depth: int,
    ) -> None:
        analysis.node_count += 1
        analysis.expression_depth = max(analysis.expression_depth, expr_depth)

        if isinstance(expr, ColumnExpression):
            # Track column reference — correlated if table not in current scope
            ref = BoundColumnReference(
                table=expr.table,
                column=expr.column,
                scope_depth=depth,
                is_correlated=False,
            )
            if expr.table and expr.table not in scope.tables:
                # Check if it's in any registered binding
                bound_aliases = {b.alias for b in scope.relation_bindings}
                if expr.table not in bound_aliases:
                    ref.is_correlated = True
                    ref.outer_scope_depth = depth - 1 if depth > 0 else None
                    analysis.correlated_references.append(ref)

        elif isinstance(expr, SubqueryExpression):
            sub_scope = ScopeNode(depth=depth + 1)
            scope.children.append(sub_scope)
            self._analyze_query(expr.query, sub_scope, analysis, depth + 1)
            analysis.subquery_depth = max(analysis.subquery_depth, depth + 1)

        elif isinstance(expr, CaseExpression):
            for case_when in expr.cases:
                self._analyze_predicate(case_when.when, scope, analysis, depth, expr_depth + 1)
                self._analyze_expression(case_when.then, scope, analysis, depth, expr_depth + 1)
            if expr.else_expression is not None:
                self._analyze_expression(expr.else_expression, scope, analysis, depth, expr_depth + 1)

        elif isinstance(expr, WindowExpression):
            self._analyze_expression(expr.expression, scope, analysis, depth, expr_depth + 1)
            for part_expr in expr.window.partition_by:
                self._analyze_expression(part_expr, scope, analysis, depth, expr_depth + 1)
            for ord_item in expr.window.order_by:
                self._analyze_expression(ord_item.expression, scope, analysis, depth, expr_depth + 1)

        elif isinstance(expr, AggregationExpression):
            if expr.argument is not None:
                self._analyze_expression(expr.argument, scope, analysis, depth, expr_depth + 1)

        elif hasattr(expr, "arguments"):
            for arg in getattr(expr, "arguments", []):
                self._analyze_expression(arg, scope, analysis, depth, expr_depth + 1)

        elif hasattr(expr, "left") and hasattr(expr, "right"):
            self._analyze_expression(expr.left, scope, analysis, depth, expr_depth + 1)
            self._analyze_expression(expr.right, scope, analysis, depth, expr_depth + 1)

        elif hasattr(expr, "operand"):
            self._analyze_expression(expr.operand, scope, analysis, depth, expr_depth + 1)

        elif hasattr(expr, "operands"):
            for operand in getattr(expr, "operands", []):
                self._analyze_expression(operand, scope, analysis, depth, expr_depth + 1)

    def _analyze_predicate(
        self,
        pred: Any,
        scope: ScopeNode,
        analysis: QueryAnalysis,
        depth: int,
        expr_depth: int,
    ) -> None:
        analysis.node_count += 1

        if isinstance(pred, BooleanPredicate):
            for operand in pred.operands:
                self._analyze_predicate(operand, scope, analysis, depth, expr_depth + 1)

        elif isinstance(pred, NotPredicate):
            self._analyze_predicate(pred.operand, scope, analysis, depth, expr_depth + 1)

        elif isinstance(pred, InSubqueryPredicate):
            self._analyze_expression(pred.expression, scope, analysis, depth, expr_depth + 1)
            sub_scope = ScopeNode(depth=depth + 1)
            scope.children.append(sub_scope)
            self._analyze_query(pred.query, sub_scope, analysis, depth + 1)
            analysis.subquery_depth = max(analysis.subquery_depth, depth + 1)

        elif isinstance(pred, ExistsPredicate):
            sub_scope = ScopeNode(depth=depth + 1)
            scope.children.append(sub_scope)
            self._analyze_query(pred.query, sub_scope, analysis, depth + 1)
            analysis.subquery_depth = max(analysis.subquery_depth, depth + 1)

        elif hasattr(pred, "expression"):
            self._analyze_expression(pred.expression, scope, analysis, depth, expr_depth + 1)
            if hasattr(pred, "left"):
                self._analyze_expression(pred.left, scope, analysis, depth, expr_depth + 1)
            if hasattr(pred, "right"):
                self._analyze_expression(pred.right, scope, analysis, depth, expr_depth + 1)
            if hasattr(pred, "lower"):
                self._analyze_expression(pred.lower, scope, analysis, depth, expr_depth + 1)
            if hasattr(pred, "upper"):
                self._analyze_expression(pred.upper, scope, analysis, depth, expr_depth + 1)
            if hasattr(pred, "values"):
                for val in getattr(pred, "values", []):
                    self._analyze_expression(val, scope, analysis, depth, expr_depth + 1)

        elif hasattr(pred, "left") and hasattr(pred, "right"):
            self._analyze_expression(pred.left, scope, analysis, depth, expr_depth + 1)
            self._analyze_expression(pred.right, scope, analysis, depth, expr_depth + 1)
