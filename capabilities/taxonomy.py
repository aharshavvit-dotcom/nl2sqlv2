from __future__ import annotations

from enum import Enum


class Capability(str, Enum):
    SIMPLE_SELECT = "SIMPLE_SELECT"
    MULTI_COLUMN_SELECT = "MULTI_COLUMN_SELECT"
    FILTER = "FILTER"
    MULTIPLE_FILTERS = "MULTIPLE_FILTERS"
    OR_FILTER = "OR_FILTER"
    AGGREGATION = "AGGREGATION"
    GROUP_BY = "GROUP_BY"
    MULTI_GROUP_BY = "MULTI_GROUP_BY"
    HAVING = "HAVING"
    ORDER_BY = "ORDER_BY"
    LIMIT = "LIMIT"
    CASE_EXPRESSION = "CASE_EXPRESSION"
    ONE_HOP_JOIN = "ONE_HOP_JOIN"
    MULTI_HOP_JOIN = "MULTI_HOP_JOIN"
    SCALAR_SUBQUERY = "SCALAR_SUBQUERY"
    IN_SUBQUERY = "IN_SUBQUERY"
    NOT_IN_SUBQUERY = "NOT_IN_SUBQUERY"
    EXISTS_SUBQUERY = "EXISTS_SUBQUERY"
    NOT_EXISTS_SUBQUERY = "NOT_EXISTS_SUBQUERY"
    DERIVED_TABLE = "DERIVED_TABLE"
    CORRELATED_SUBQUERY = "CORRELATED_SUBQUERY"
    WINDOW_ROW_NUMBER = "WINDOW_ROW_NUMBER"
    WINDOW_RANK = "WINDOW_RANK"
    WINDOW_DENSE_RANK = "WINDOW_DENSE_RANK"
    WINDOW_LAG = "WINDOW_LAG"
    WINDOW_LEAD = "WINDOW_LEAD"
    WINDOW_AGGREGATE = "WINDOW_AGGREGATE"
    WINDOW_FRAME = "WINDOW_FRAME"
    UNION_ALL = "UNION_ALL"
    UNION = "UNION"
    INTERSECT = "INTERSECT"
    EXCEPT = "EXCEPT"
    CTE = "CTE"
    RECURSIVE_CTE = "RECURSIVE_CTE"


class SafetyLabel(str, Enum):
    MUTATION_INSERT = "MUTATION_INSERT"
    MUTATION_UPDATE = "MUTATION_UPDATE"
    MUTATION_DELETE = "MUTATION_DELETE"
    MUTATION_MERGE = "MUTATION_MERGE"
    DDL_CREATE = "DDL_CREATE"
    DDL_ALTER = "DDL_ALTER"
    DDL_DROP = "DDL_DROP"
    ADMINISTRATIVE = "ADMINISTRATIVE"
    UNSUPPORTED_DIALECT = "UNSUPPORTED_DIALECT"
    MISSING_SCHEMA_ELEMENT = "MISSING_SCHEMA_ELEMENT"
    AMBIGUOUS_REQUEST = "AMBIGUOUS_REQUEST"
    NON_DATABASE_REQUEST = "NON_DATABASE_REQUEST"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    UNSAFE_REQUEST = "UNSAFE_REQUEST"


ALL_CAPABILITIES: tuple[Capability, ...] = tuple(Capability)
ALL_SAFETY_LABELS: tuple[SafetyLabel, ...] = tuple(SafetyLabel)

# Conservative installed-support policy for the current QueryIR v1 renderer.
# This is intentionally independent from what a SQL example requires.
SUPPORTED_QUERYIR_V1_CAPABILITIES: frozenset[Capability] = frozenset(
    {
        Capability.SIMPLE_SELECT,
        Capability.MULTI_COLUMN_SELECT,
        Capability.FILTER,
        Capability.MULTIPLE_FILTERS,
        Capability.AGGREGATION,
        Capability.GROUP_BY,
        Capability.ORDER_BY,
        Capability.LIMIT,
        Capability.ONE_HOP_JOIN,
    }
)


def capability_names(values: list[Capability] | set[Capability] | tuple[Capability, ...]) -> list[str]:
    return sorted(item.value if isinstance(item, Capability) else str(item) for item in values)


def safety_label_names(values: list[SafetyLabel] | set[SafetyLabel] | tuple[SafetyLabel, ...]) -> list[str]:
    return sorted(item.value if isinstance(item, SafetyLabel) else str(item) for item in values)
