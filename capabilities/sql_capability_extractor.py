from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable

import sqlglot
from sqlglot import exp

from .contracts import (
    CapabilityAnnotation,
    CorrelatedSubqueryInfo,
    JoinEdge,
    PartialSQLSupervision,
    SetOperationBranch,
    TaskMasks,
    WindowFunctionInfo,
)
from .taxonomy import Capability, SafetyLabel, SUPPORTED_QUERYIR_V1_CAPABILITIES


ANNOTATION_VERSION = "capability_taxonomy_v1"
FILTER_OPERATOR_TYPES: tuple[type[exp.Expression], ...] = (
    exp.EQ,
    exp.NEQ,
    exp.GT,
    exp.GTE,
    exp.LT,
    exp.LTE,
    exp.Like,
    exp.In,
    exp.Between,
    exp.Is,
)
AGGREGATE_TYPES: tuple[type[exp.Expression], ...] = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)
SET_OPERATION_TYPES: tuple[type[exp.Expression], ...] = (exp.Union, exp.Intersect, exp.Except)


class SQLCapabilityExtractor:
    """Deterministic SQL AST extractor for capability and auxiliary labels."""

    def __init__(
        self,
        dialect: str = "sqlite",
        supported_capabilities: Iterable[Capability] | None = None,
    ):
        self.dialect = dialect or "sqlite"
        self.supported_capabilities = frozenset(supported_capabilities or SUPPORTED_QUERYIR_V1_CAPABILITIES)
        self.parser_version = f"sqlglot:{getattr(sqlglot, '__version__', 'unknown')}"

    def extract(
        self,
        sql: str,
        *,
        example_id: str = "",
        dataset_source: str = "unknown",
        database_identifier: str = "unknown",
        schema: Any | None = None,
        sql_dialect: str | None = None,
        full_query_ir_supported: bool = False,
        unsupported_reason: str | None = None,
    ) -> CapabilityAnnotation:
        dialect = sql_dialect or self.dialect
        try:
            tree = self._parse(sql, dialect)
            partial = self._extract_partial(tree, full_query_ir_supported=full_query_ir_supported, unsupported_reason=unsupported_reason)
        except Exception as exc:
            partial = PartialSQLSupervision(
                extraction_status="parse_error",
                validation_errors=[str(exc)],
                full_query_ir_supported=full_query_ir_supported,
                unsupported_reason=unsupported_reason or "parse_error",
            )

        required = _sorted_names(partial.required_capabilities)
        safety_labels = _sorted_names(partial.safety_labels)
        supported = _sorted_names(self.supported_capabilities)
        unsupported = sorted(set(required) - set(supported))
        understood = partial.extraction_status == "ok" and (bool(required) or bool(safety_labels) or bool(partial.referenced_tables))
        currently_supported = understood and not safety_labels and not unsupported
        task_masks = self.task_masks(partial, full_query_ir_supported=full_query_ir_supported)
        partial = partial.model_copy(update={"full_query_ir_supported": full_query_ir_supported, "unsupported_reason": unsupported_reason})

        return CapabilityAnnotation(
            example_id=example_id or "unknown",
            dataset_source=dataset_source or "unknown",
            database_identifier=database_identifier or "unknown",
            sql_dialect=dialect,
            parser_version=self.parser_version,
            annotation_version=ANNOTATION_VERSION,
            schema_fingerprint=schema_fingerprint(schema),
            extraction_status=partial.extraction_status,
            validation_errors=partial.validation_errors,
            understood=understood,
            required_capabilities=required,
            supported_capabilities=supported,
            currently_supported=currently_supported,
            unsupported_required_capabilities=unsupported,
            safety_labels=safety_labels,
            partial_supervision=partial,
            task_masks=task_masks,
        )

    def with_conversion_result(self, annotation: CapabilityAnnotation, result: dict[str, Any]) -> CapabilityAnnotation:
        success = bool(result.get("success"))
        reason = None if success else (result.get("unsupported_reason") or "unsupported")
        partial = annotation.partial_supervision.model_copy(
            update={"full_query_ir_supported": success, "unsupported_reason": reason}
        )
        return annotation.model_copy(
            update={
                "partial_supervision": partial,
                "task_masks": self.task_masks(partial, full_query_ir_supported=success),
            }
        )

    @staticmethod
    def task_masks(partial: PartialSQLSupervision, *, full_query_ir_supported: bool | None = None) -> TaskMasks:
        parseable = partial.extraction_status == "ok"
        full_ir = partial.full_query_ir_supported if full_query_ir_supported is None else full_query_ir_supported
        has_columns = bool(partial.referenced_columns or partial.selected_columns)
        has_tables = bool(partial.referenced_tables)
        return TaskMasks(
            capability=1 if parseable and (partial.required_capabilities or partial.safety_labels) else 0,
            safety=1 if parseable and partial.safety_labels else 0,
            table=1 if parseable and has_tables else 0,
            column=1 if parseable and has_columns else 0,
            aggregation=1 if parseable and partial.aggregation_functions else 0,
            filter=1 if parseable and (partial.filter_columns or partial.filter_operators) else 0,
            join_edge=1 if parseable and partial.join_edges else 0,
            complexity=1 if parseable else 0,
            contrastive_schema_linking=1 if parseable and (has_tables or has_columns) else 0,
            subquery=1 if parseable and partial.subquery_depth > 0 else 0,
            window=1 if parseable and partial.window_functions else 0,
            set_operation=1 if parseable and partial.set_operation else 0,
            full_query_ir=1 if full_ir else 0,
        )

    def _parse(self, sql: str, dialect: str) -> exp.Expression:
        last_error: Exception | None = None
        for candidate in [dialect, None]:
            try:
                return sqlglot.parse_one(sql, read=candidate)
            except Exception as exc:
                last_error = exc
        raise ValueError(f"SQL parse failed: {last_error}")

    def _extract_partial(
        self,
        tree: exp.Expression,
        *,
        full_query_ir_supported: bool,
        unsupported_reason: str | None,
    ) -> PartialSQLSupervision:
        safety_labels = self._safety_labels(tree)
        capabilities = set[Capability]()
        capabilities.update(self._select_capabilities(tree))
        capabilities.update(self._join_capabilities(tree))
        capabilities.update(self._filter_capabilities(tree))
        capabilities.update(self._aggregation_capabilities(tree))
        capabilities.update(self._subquery_capabilities(tree))
        capabilities.update(self._window_capabilities(tree))
        capabilities.update(self._set_operation_capabilities(tree))
        capabilities.update(self._cte_capabilities(tree))

        referenced_tables = sorted(set(self._table_names(tree)))
        referenced_columns = sorted(set(self._column_sql(column) for column in tree.find_all(exp.Column)))
        selected_columns = sorted(set(self._selected_columns(tree)))
        group_by_columns = sorted(set(self._group_by_columns(tree)))
        filter_columns, filter_operators = self._filter_labels(tree)
        join_edges = self._join_edges(tree)
        window_functions = self._window_functions(tree)
        subquery_types = self._subquery_types(tree)
        set_operation = self._set_operation(tree)

        return PartialSQLSupervision(
            required_capabilities=sorted(capabilities, key=lambda item: item.value),
            safety_labels=sorted(safety_labels, key=lambda item: item.value),
            referenced_tables=referenced_tables,
            referenced_columns=referenced_columns,
            selected_columns=selected_columns,
            aggregation_functions=sorted(set(self._aggregation_functions(tree))),
            group_by_columns=group_by_columns,
            filter_columns=sorted(set(filter_columns)),
            filter_operators=sorted(set(filter_operators)),
            join_edges=join_edges,
            join_path_length=len(join_edges) if join_edges else 0,
            subquery_types=sorted(set(subquery_types)),
            subquery_depth=self._subquery_depth(tree),
            correlated_subqueries=self._correlated_subqueries(tree),
            window_functions=window_functions,
            window_partition_columns=sorted({col for item in window_functions for col in item.partition_columns}),
            window_order_columns=sorted({col for item in window_functions for col in item.order_columns}),
            set_operation=set_operation,
            set_operation_branches=self._set_operation_branches(tree),
            has_case=tree.find(exp.Case) is not None,
            has_having=tree.find(exp.Having) is not None,
            full_query_ir_supported=full_query_ir_supported,
            unsupported_reason=unsupported_reason,
            extraction_status="ok",
            validation_errors=[],
        )

    def _select_capabilities(self, tree: exp.Expression) -> set[Capability]:
        capabilities: set[Capability] = set()
        if isinstance(tree, exp.Select) or any(True for _ in tree.find_all(exp.Select)) or isinstance(tree, SET_OPERATION_TYPES):
            capabilities.add(Capability.SIMPLE_SELECT)
        outer_select = self._outer_select(tree)
        if outer_select is not None and len(getattr(outer_select, "expressions", []) or []) > 1:
            capabilities.add(Capability.MULTI_COLUMN_SELECT)
        if any(True for _ in tree.find_all(exp.Case)):
            capabilities.add(Capability.CASE_EXPRESSION)
        if any(True for _ in tree.find_all(exp.Having)):
            capabilities.add(Capability.HAVING)
        if any(self._order_expressions(select) for select in tree.find_all(exp.Select)):
            capabilities.add(Capability.ORDER_BY)
        if any(select.args.get("limit") is not None for select in tree.find_all(exp.Select)):
            capabilities.add(Capability.LIMIT)
        return capabilities

    def _join_capabilities(self, tree: exp.Expression) -> set[Capability]:
        join_count = sum(1 for _ in tree.find_all(exp.Join))
        if join_count == 1:
            return {Capability.ONE_HOP_JOIN}
        if join_count > 1:
            return {Capability.MULTI_HOP_JOIN}
        return set()

    def _filter_capabilities(self, tree: exp.Expression) -> set[Capability]:
        capabilities: set[Capability] = set()
        filters = [node for node in tree.find_all(*FILTER_OPERATOR_TYPES)]
        if filters:
            capabilities.add(Capability.FILTER)
        if len(filters) > 1 or any(True for _ in tree.find_all(exp.And)):
            capabilities.add(Capability.MULTIPLE_FILTERS)
        if any(True for _ in tree.find_all(exp.Or)):
            capabilities.add(Capability.OR_FILTER)
        return capabilities

    def _aggregation_capabilities(self, tree: exp.Expression) -> set[Capability]:
        capabilities: set[Capability] = set()
        if any(True for _ in tree.find_all(*AGGREGATE_TYPES)):
            capabilities.add(Capability.AGGREGATION)
        group_columns = self._group_by_columns(tree)
        if group_columns:
            capabilities.add(Capability.GROUP_BY)
        if len(group_columns) > 1:
            capabilities.add(Capability.MULTI_GROUP_BY)
        return capabilities

    def _subquery_capabilities(self, tree: exp.Expression) -> set[Capability]:
        capabilities: set[Capability] = set()
        for subquery in tree.find_all(exp.Subquery):
            if self._is_derived_table(subquery):
                capabilities.add(Capability.DERIVED_TABLE)
            else:
                capabilities.add(Capability.SCALAR_SUBQUERY)
        for node in tree.find_all(exp.In):
            if self._contains_select(node):
                if self._is_not_parent(node):
                    capabilities.add(Capability.NOT_IN_SUBQUERY)
                else:
                    capabilities.add(Capability.IN_SUBQUERY)
        for node in tree.find_all(exp.Exists):
            if self._is_not_parent(node):
                capabilities.add(Capability.NOT_EXISTS_SUBQUERY)
            else:
                capabilities.add(Capability.EXISTS_SUBQUERY)
        if self._correlated_subqueries(tree):
            capabilities.add(Capability.CORRELATED_SUBQUERY)
        return capabilities

    def _window_capabilities(self, tree: exp.Expression) -> set[Capability]:
        capabilities: set[Capability] = set()
        for window in tree.find_all(exp.Window):
            name = self._window_function_name(window)
            if name == "ROW_NUMBER":
                capabilities.add(Capability.WINDOW_ROW_NUMBER)
            elif name == "RANK":
                capabilities.add(Capability.WINDOW_RANK)
            elif name == "DENSE_RANK":
                capabilities.add(Capability.WINDOW_DENSE_RANK)
            elif name == "LAG":
                capabilities.add(Capability.WINDOW_LAG)
            elif name == "LEAD":
                capabilities.add(Capability.WINDOW_LEAD)
            elif name in {"COUNT", "SUM", "AVG", "MIN", "MAX"}:
                capabilities.add(Capability.WINDOW_AGGREGATE)
            if window.args.get("spec") is not None:
                capabilities.add(Capability.WINDOW_FRAME)
        return capabilities

    def _set_operation_capabilities(self, tree: exp.Expression) -> set[Capability]:
        capabilities: set[Capability] = set()
        for node in self._set_operation_nodes(tree):
            if isinstance(node, exp.Union):
                capabilities.add(Capability.UNION if bool(node.args.get("distinct", True)) else Capability.UNION_ALL)
            elif isinstance(node, exp.Intersect):
                capabilities.add(Capability.INTERSECT)
            elif isinstance(node, exp.Except):
                capabilities.add(Capability.EXCEPT)
        return capabilities

    def _cte_capabilities(self, tree: exp.Expression) -> set[Capability]:
        with_expr = tree.args.get("with") or tree.args.get("with_")
        if with_expr is None and not any(True for _ in tree.find_all(exp.CTE)):
            return set()
        capabilities = {Capability.CTE}
        if bool(getattr(with_expr, "args", {}).get("recursive")):
            capabilities.add(Capability.RECURSIVE_CTE)
        return capabilities

    def _safety_labels(self, tree: exp.Expression) -> set[SafetyLabel]:
        labels: set[SafetyLabel] = set()
        if isinstance(tree, exp.Insert):
            labels.add(SafetyLabel.MUTATION_INSERT)
        elif isinstance(tree, exp.Update):
            labels.add(SafetyLabel.MUTATION_UPDATE)
        elif isinstance(tree, exp.Delete):
            labels.add(SafetyLabel.MUTATION_DELETE)
        elif _isinstance_name(tree, "Merge"):
            labels.add(SafetyLabel.MUTATION_MERGE)
        elif isinstance(tree, exp.Create):
            labels.add(SafetyLabel.DDL_CREATE)
        elif isinstance(tree, exp.Alter):
            labels.add(SafetyLabel.DDL_ALTER)
        elif isinstance(tree, exp.Drop):
            labels.add(SafetyLabel.DDL_DROP)
        elif _isinstance_name(tree, "Command"):
            labels.add(SafetyLabel.ADMINISTRATIVE)
        return labels

    def _selected_columns(self, tree: exp.Expression) -> list[str]:
        columns: list[str] = []
        outer_select = self._outer_select(tree)
        if outer_select is None:
            return columns
        for item in getattr(outer_select, "expressions", []) or []:
            if isinstance(item, exp.Star):
                columns.append("*")
                continue
            inner = item.this if isinstance(item, exp.Alias) and item.this is not None else item
            if isinstance(inner, exp.Column):
                columns.append(self._column_sql(inner))
            else:
                columns.extend(self._column_sql(column) for column in inner.find_all(exp.Column))
        return columns

    def _group_by_columns(self, tree: exp.Expression) -> list[str]:
        columns: list[str] = []
        for select in tree.find_all(exp.Select):
            group = select.args.get("group")
            if group is None:
                continue
            for item in getattr(group, "expressions", []) or []:
                if isinstance(item, exp.Column):
                    columns.append(self._column_sql(item))
                else:
                    columns.extend(self._column_sql(column) for column in item.find_all(exp.Column))
        return columns

    def _filter_labels(self, tree: exp.Expression) -> tuple[list[str], list[str]]:
        columns: list[str] = []
        operators: list[str] = []
        for node in tree.find_all(*FILTER_OPERATOR_TYPES):
            operators.append(_operator_name(node))
            left = getattr(node, "this", None)
            if isinstance(left, exp.Column):
                columns.append(self._column_sql(left))
            else:
                columns.extend(self._column_sql(column) for column in node.find_all(exp.Column))
        return columns, operators

    def _join_edges(self, tree: exp.Expression) -> list[JoinEdge]:
        edges: list[JoinEdge] = []
        for join in tree.find_all(exp.Join):
            condition = join.args.get("on")
            join_type = str(join.args.get("kind") or "INNER").upper()
            eq_conditions = list(condition.find_all(exp.EQ)) if condition is not None else []
            if isinstance(condition, exp.EQ):
                eq_conditions.insert(0, condition)
            if not eq_conditions:
                edges.append(JoinEdge(condition=_sql(condition), join_type=join_type))
                continue
            for eq_condition in _dedupe_nodes(eq_conditions):
                left = eq_condition.this if isinstance(eq_condition.this, exp.Column) else None
                right = eq_condition.expression if isinstance(eq_condition.expression, exp.Column) else None
                edges.append(
                    JoinEdge(
                        left_table=str(left.table) if left is not None and left.table else None,
                        left_column=str(left.name) if left is not None and left.name else None,
                        right_table=str(right.table) if right is not None and right.table else None,
                        right_column=str(right.name) if right is not None and right.name else None,
                        condition=_sql(eq_condition),
                        join_type=join_type,
                    )
                )
        return edges

    def _aggregation_functions(self, tree: exp.Expression) -> list[str]:
        return [node.key.upper() for node in tree.find_all(*AGGREGATE_TYPES)]

    def _window_functions(self, tree: exp.Expression) -> list[WindowFunctionInfo]:
        windows: list[WindowFunctionInfo] = []
        for window in tree.find_all(exp.Window):
            order_columns, order_directions = self._order_columns_and_directions(window.args.get("order"))
            windows.append(
                WindowFunctionInfo(
                    function=self._window_function_name(window),
                    arguments=[self._column_sql(column) for column in (window.this.find_all(exp.Column) if window.this is not None else [])],
                    partition_columns=[self._column_sql(column) for column in self._window_partition_columns(window)],
                    order_columns=order_columns,
                    order_directions=order_directions,
                    frame_definition=_sql(window.args.get("spec")) if window.args.get("spec") is not None else None,
                )
            )
        return windows

    def _window_partition_columns(self, window: exp.Window) -> list[exp.Column]:
        partition = window.args.get("partition_by") or []
        items = partition if isinstance(partition, list) else [partition]
        columns: list[exp.Column] = []
        for item in items:
            if isinstance(item, exp.Column):
                columns.append(item)
            elif isinstance(item, exp.Expression):
                columns.extend(item.find_all(exp.Column))
        return columns

    def _order_columns_and_directions(self, order: exp.Expression | None) -> tuple[list[str], list[str]]:
        if order is None:
            return [], []
        columns: list[str] = []
        directions: list[str] = []
        for ordered in getattr(order, "expressions", []) or []:
            target = ordered.this if getattr(ordered, "this", None) is not None else ordered
            if isinstance(target, exp.Column):
                columns.append(self._column_sql(target))
            elif isinstance(target, exp.Expression):
                columns.extend(self._column_sql(column) for column in target.find_all(exp.Column))
            directions.append("DESC" if bool(ordered.args.get("desc")) else "ASC")
        return columns, directions

    def _subquery_types(self, tree: exp.Expression) -> list[str]:
        values: list[str] = []
        for subquery in tree.find_all(exp.Subquery):
            values.append("DERIVED_TABLE" if self._is_derived_table(subquery) else "SCALAR_SUBQUERY")
        for node in tree.find_all(exp.In):
            if self._contains_select(node):
                values.append("NOT_IN_SUBQUERY" if self._is_not_parent(node) else "IN_SUBQUERY")
        for node in tree.find_all(exp.Exists):
            values.append("NOT_EXISTS_SUBQUERY" if self._is_not_parent(node) else "EXISTS_SUBQUERY")
        if self._correlated_subqueries(tree):
            values.append("CORRELATED_SUBQUERY")
        return values

    def _subquery_depth(self, tree: exp.Expression) -> int:
        def depth(node: exp.Expression, current: int) -> int:
            child_depths = [
                depth(child, current + (1 if isinstance(child, exp.Subquery) else 0))
                for child in node.iter_expressions()
                if isinstance(child, exp.Expression)
            ]
            return max([current, *child_depths])

        explicit_depth = depth(tree, 0)
        implicit_depth = 1 if any(True for _ in tree.find_all(exp.Exists)) or any(self._contains_select(node) for node in tree.find_all(exp.In)) else 0
        return max(explicit_depth, implicit_depth)

    def _correlated_subqueries(self, tree: exp.Expression) -> list[CorrelatedSubqueryInfo]:
        outer_aliases = self._table_aliases(tree)
        results: list[CorrelatedSubqueryInfo] = []
        for subquery in self._subquery_scopes(tree):
            inner_aliases = self._table_aliases(subquery)
            candidate_outer = set(outer_aliases) - set(inner_aliases)
            correlated_columns = []
            operators = []
            for column in subquery.find_all(exp.Column):
                table = str(column.table) if column.table else ""
                if table and table in candidate_outer:
                    correlated_columns.append(self._column_sql(column))
                    parent = getattr(column, "parent", None)
                    if isinstance(parent, FILTER_OPERATOR_TYPES):
                        operators.append(_operator_name(parent))
            if correlated_columns:
                results.append(
                    CorrelatedSubqueryInfo(
                        outer_scope_tables=sorted(candidate_outer),
                        inner_scope_tables=sorted(inner_aliases),
                        correlated_columns=sorted(set(correlated_columns)),
                        correlation_operators=sorted(set(operators)),
                    )
                )
        return results

    def _set_operation(self, tree: exp.Expression) -> str | None:
        node = next(iter(self._set_operation_nodes(tree)), None)
        if node is None:
            return None
        if isinstance(node, exp.Union):
            return "UNION" if bool(node.args.get("distinct", True)) else "UNION_ALL"
        if isinstance(node, exp.Intersect):
            return "INTERSECT"
        if isinstance(node, exp.Except):
            return "EXCEPT"
        return None

    def _set_operation_branches(self, tree: exp.Expression) -> list[SetOperationBranch]:
        node = next(iter(self._set_operation_nodes(tree)), None)
        if node is None:
            return []
        branches = [node.this, node.expression]
        records: list[SetOperationBranch] = []
        for index, branch in enumerate(branches, start=1):
            if not isinstance(branch, exp.Expression):
                continue
            branch_caps = sorted(
                self._select_capabilities(branch)
                | self._filter_capabilities(branch)
                | self._aggregation_capabilities(branch)
                | self._join_capabilities(branch),
                key=lambda item: item.value,
            )
            records.append(SetOperationBranch(branch_index=index, sql=_sql(branch), required_capabilities=branch_caps))
        return records

    def _set_operation_nodes(self, tree: exp.Expression) -> list[exp.Expression]:
        nodes = []
        if isinstance(tree, SET_OPERATION_TYPES):
            nodes.append(tree)
        nodes.extend(tree.find_all(*SET_OPERATION_TYPES))
        return _dedupe_nodes(nodes)

    def _table_names(self, tree: exp.Expression) -> list[str]:
        return [str(table.name) for table in tree.find_all(exp.Table) if table.name]

    def _table_aliases(self, tree: exp.Expression) -> set[str]:
        aliases: set[str] = set()
        for table in tree.find_all(exp.Table):
            if table.name:
                aliases.add(str(table.name))
            if table.alias:
                aliases.add(str(table.alias))
        return aliases

    def _outer_select(self, tree: exp.Expression) -> exp.Select | None:
        if isinstance(tree, exp.Select):
            return tree
        if isinstance(tree, SET_OPERATION_TYPES):
            return tree.this if isinstance(tree.this, exp.Select) else tree.find(exp.Select)
        return tree.find(exp.Select)

    def _order_expressions(self, select: exp.Select) -> list[Any]:
        order = select.args.get("order")
        return list(getattr(order, "expressions", []) or []) if order is not None else []

    def _is_derived_table(self, subquery: exp.Subquery) -> bool:
        parent = getattr(subquery, "parent", None)
        return isinstance(parent, (exp.From, exp.Join))

    def _contains_select(self, node: exp.Expression) -> bool:
        return any(True for _ in node.find_all(exp.Select))

    def _is_not_parent(self, node: exp.Expression) -> bool:
        return isinstance(getattr(node, "parent", None), exp.Not)

    def _window_function_name(self, window: exp.Window) -> str:
        target = window.this
        if target is None:
            return "UNKNOWN"
        name = getattr(target, "key", "") or type(target).__name__
        normalized = str(name).upper()
        return {
            "ROWNUMBER": "ROW_NUMBER",
            "DENSERANK": "DENSE_RANK",
        }.get(normalized, normalized)

    def _column_sql(self, column: exp.Column) -> str:
        table = str(column.table) if column.table else ""
        name = str(column.name) if column.name else _sql(column)
        return f"{table}.{name}" if table else name

    def _subquery_scopes(self, tree: exp.Expression) -> list[exp.Expression]:
        scopes: list[exp.Expression] = []
        scopes.extend(tree.find_all(exp.Subquery))
        for exists in tree.find_all(exp.Exists):
            if isinstance(exists.this, exp.Expression):
                scopes.append(exists.this)
        for node in tree.find_all(exp.In):
            for select in node.find_all(exp.Select):
                scopes.append(select)
        return _dedupe_nodes(scopes)


def schema_fingerprint(schema: Any | None) -> str:
    if schema is None:
        return "unknown"
    if hasattr(schema, "model_dump"):
        payload = schema.model_dump()
    elif hasattr(schema, "to_dict"):
        payload = schema.to_dict()
    elif isinstance(schema, dict):
        payload = schema
    else:
        payload = str(schema)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capability_vector(capabilities: Iterable[str | Capability]) -> list[int]:
    values = {item.value if isinstance(item, Capability) else str(item) for item in capabilities}
    return [1 if capability.value in values else 0 for capability in Capability]


def capability_frequencies(rows: Iterable[CapabilityAnnotation]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(str(item) for item in row.required_capabilities)
    return dict(sorted(counts.items()))


def _operator_name(node: exp.Expression) -> str:
    if isinstance(node, exp.EQ):
        return "EQ"
    if isinstance(node, exp.NEQ):
        return "NEQ"
    if isinstance(node, exp.GT):
        return "GT"
    if isinstance(node, exp.GTE):
        return "GTE"
    if isinstance(node, exp.LT):
        return "LT"
    if isinstance(node, exp.LTE):
        return "LTE"
    if isinstance(node, exp.Like):
        return "LIKE"
    if isinstance(node, exp.In):
        return "IN"
    if isinstance(node, exp.Between):
        return "BETWEEN"
    if isinstance(node, exp.Is):
        return "IS"
    return type(node).__name__.upper()


def _sql(node: exp.Expression | None) -> str | None:
    return None if node is None else node.sql(dialect="sqlite")


def _isinstance_name(node: exp.Expression, class_name: str) -> bool:
    klass = getattr(exp, class_name, None)
    return bool(klass is not None and isinstance(node, klass))


def _sorted_names(values: Iterable[Any]) -> list[str]:
    return sorted(item.value if hasattr(item, "value") else str(item) for item in values)


def _dedupe_nodes(nodes: Iterable[exp.Expression]) -> list[exp.Expression]:
    seen: set[int] = set()
    result: list[exp.Expression] = []
    for node in nodes:
        marker = id(node)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(node)
    return result
