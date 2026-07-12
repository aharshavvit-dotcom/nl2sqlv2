# QueryIR v2 Boolean Predicate Specification

Phase: 2B

Scope: recursive `WHERE` predicate trees and diagnostic/test-only native
rendering for AND, OR, NOT, null, range, comparison, and literal membership
predicates.

## Predicate Nodes

QueryIR v2 predicates use the discriminator `predicate_type`.

Supported Phase 2B predicate node types:

- `COMPARISON_PREDICATE`
- `BOOLEAN_PREDICATE`
- `NOT_PREDICATE`
- `NULL_PREDICATE`
- `BETWEEN_PREDICATE`
- `IN_LITERAL_PREDICATE`

`BOOLEAN_PREDICATE` has `operator` of `AND` or `OR` and an ordered `operands`
list. Mixed AND/OR grouping is represented by nesting, not flattening.

`NOT_PREDICATE` has exactly one `operand`.

`IN_LITERAL_PREDICATE` supports literal lists only. Subquery-backed `IN` remains
a non-goal.

## Query Root

`QueryNode.where` is the canonical Phase 2B predicate-tree root. The older
`QueryNode.predicates` list remains for compatibility with Phase 2A payloads
and v1 migration diagnostics, but native Boolean rendering and SQL-to-v2
conversion use `where`.

## Validation

`QueryIRV2Validator` enforces:

- valid discriminated predicate nodes;
- Boolean operators limited to `AND` and `OR`;
- at least two operands for Boolean predicates;
- one operand for NOT predicates;
- maximum recursive depth;
- maximum predicate node count;
- maximum IN literal list size;
- literal type compatibility for comparison, IN, and BETWEEN predicates;
- `IS NULL` / `IS NOT NULL` via `NULL_PREDICATE`, not `column = NULL`;
- `OR_FILTER` capability-label consistency with the v2 predicate tree.

## Canonicalization

`canonicalize_predicate` is deterministic and idempotent.

It may:

- flatten nested `AND` under `AND`;
- flatten nested `OR` under `OR`;
- remove one-child Boolean groups if encountered;
- preserve mixed AND/OR grouping;
- preserve operand order;
- preserve explicit NOT.

## SQL-to-v2 Conversion

`SQLToQueryIRV2Converter` parses SQLGlot ASTs into typed predicate trees for:

- AND;
- OR;
- NOT;
- comparisons;
- `IS NULL` / `IS NOT NULL`;
- literal `IN` / `NOT IN`;
- `BETWEEN` / `NOT BETWEEN`.

It rejects non-SELECT statements, HAVING, CASE, subqueries, windows, set
operations, and subquery-backed IN.

## Native Rendering

`QueryIRV2NativeRenderer` renders v2 predicate trees recursively and uses the
existing safe identifier/literal helpers. It parenthesizes Boolean children to
preserve the explicit tree, including AND under OR and OR under AND.

Runtime policy for Phase 2B:

- model output QueryIR version remains `1`;
- production preferred QueryIR version remains `1`;
- OR rendering is disabled by default;
- tests and diagnostics may enable OR rendering with `enable_or_rendering=True`.

## v1 Compatibility

v1-to-v2 migration maps:

- no filters -> no `where`;
- one filter -> that predicate as `where`;
- multiple filters -> `BOOLEAN_PREDICATE(operator="AND")`.

v2-to-v1 compatibility accepts only the v1-safe subset:

- a single comparison predicate;
- literal IN predicates supported by v1;
- an AND tree of v1-compatible predicates.

It rejects OR, NOT, NULL, BETWEEN, and other non-v1 predicates with
`v2_predicate_not_representable_in_v1`.

## Non-goals

No HAVING, CASE, scalar subqueries, IN subqueries, EXISTS, windows, set
operations, model retraining, relation-aware attention, learned loss weighting,
semantic RAG, or production routing changes were added.
