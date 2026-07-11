# SQL Capability Taxonomy

Phase 1 introduces a first-class, multi-label SQL capability taxonomy. These labels describe what a query requires, not whether the current QueryIR v1 renderer can execute it.

## Query Capabilities

- SIMPLE_SELECT
- MULTI_COLUMN_SELECT
- FILTER
- MULTIPLE_FILTERS
- OR_FILTER
- AGGREGATION
- GROUP_BY
- MULTI_GROUP_BY
- HAVING
- ORDER_BY
- LIMIT
- CASE_EXPRESSION
- ONE_HOP_JOIN
- MULTI_HOP_JOIN
- SCALAR_SUBQUERY
- IN_SUBQUERY
- NOT_IN_SUBQUERY
- EXISTS_SUBQUERY
- NOT_EXISTS_SUBQUERY
- DERIVED_TABLE
- CORRELATED_SUBQUERY
- WINDOW_ROW_NUMBER
- WINDOW_RANK
- WINDOW_DENSE_RANK
- WINDOW_LAG
- WINDOW_LEAD
- WINDOW_AGGREGATE
- WINDOW_FRAME
- UNION_ALL
- UNION
- INTERSECT
- EXCEPT
- CTE
- RECURSIVE_CTE

## Safety Labels

Safety labels are separate from query capabilities and are not mutually exclusive with one another.

- MUTATION_INSERT
- MUTATION_UPDATE
- MUTATION_DELETE
- MUTATION_MERGE
- DDL_CREATE
- DDL_ALTER
- DDL_DROP
- ADMINISTRATIVE
- UNSUPPORTED_DIALECT
- MISSING_SCHEMA_ELEMENT
- AMBIGUOUS_REQUEST
- NON_DATABASE_REQUEST
- INSUFFICIENT_CONTEXT
- UNSAFE_REQUEST

## Installed Support Policy

The current QueryIR v1 support policy is conservative and additive:

- SIMPLE_SELECT
- MULTI_COLUMN_SELECT
- FILTER
- MULTIPLE_FILTERS
- AGGREGATION
- GROUP_BY
- ORDER_BY
- LIMIT
- ONE_HOP_JOIN

Unsupported examples are retained only for masked auxiliary supervision. They do not contribute to full QueryIR or SQL-generation losses.
