# QueryIR v2 Boolean Predicate Evaluation Report

Phase: 2B

## Scope

This report covers recursive QueryIR v2 `WHERE` predicate trees and
diagnostic/test-only native rendering for AND, OR, NOT, null predicates,
literal IN predicates, and BETWEEN predicates.

## Evaluation Dataset

Curated dataset: `evaluation/query_ir_v2_boolean_eval_cases.jsonl`

Coverage:

- simple OR;
- OR with AND;
- nested OR;
- nested AND;
- NOT over OR;
- OR with null predicates;
- OR across two tables;
- OR involving dates;
- equivalent reordered expressions;
- deliberately non-equivalent precedence cases.

All examples are parsed through SQLGlot, converted to QueryIR v2, rendered with
the feature-flagged native v2 renderer, reparsed to QueryIR v2, and executed
against an in-memory SQLite fixture in tests.

## Results

| Evaluation slice | Cases | Passed |
| --- | ---: | ---: |
| Curated Boolean evaluation set | 12 | 12 |
| Simple OR | 1 | 1 |
| Mixed AND/OR | 2 | 2 |
| NOT | 1 | 1 |
| Null predicate OR | 1 | 1 |
| Cross-table OR | 1 | 1 |
| Date OR | 1 | 1 |
| Precedence pair | 2 | 2 |
| Reordered-equivalent pair | 2 | 2 |

Additional compatibility gate:

| Gate | Count | Passed |
| --- | ---: | ---: |
| Phase 2A v1 generated corpus parity | 3075 | 3075 |

## Structural Equivalence

For each curated Boolean case:

1. Parse SQL to QueryIR v2.
2. Render with `QueryIRV2NativeRenderer(enable_or_rendering=True)`.
3. Reparse rendered SQL to QueryIR v2.
4. Assert exact predicate-tree equality.

Result: `12 / 12` exact predicate-tree structural matches.

## Execution Equivalence

For each curated Boolean case:

1. Execute the source SQL against the SQLite fixture.
2. Execute the rendered QueryIR v2 SQL against the same fixture.
3. Compare sorted result rows.

Result: `12 / 12` execution-equivalent.

## Precedence Preservation

The required precedence pair is included:

```sql
region = 'US' OR region = 'CA' AND status = 'ACTIVE'
```

and:

```sql
(region = 'US' OR region = 'CA') AND status = 'ACTIVE'
```

The converted predicate trees differ and the SQLite result sets differ, proving
that grouping is preserved rather than flattened.

## Test Commands

```powershell
python -m pytest tests\test_query_ir_v2_boolean_models.py tests\test_query_ir_v2_boolean_validation.py tests\test_query_ir_v2_boolean_canonicalization.py tests\test_sql_to_query_ir_v2_boolean_conversion.py tests\test_query_ir_v2_boolean_renderer.py tests\test_query_ir_v2_boolean_precedence.py tests\test_query_ir_v2_boolean_execution_equivalence.py tests\test_query_ir_v1_boolean_migration.py tests\test_query_ir_v2_boolean_v1_compatibility.py tests\test_capability_or_filter_consistency.py tests\test_query_ir_v2_boolean_depth_limits.py
```

```powershell
python -m ir.query_ir_v2_parity training_data\ir_training_examples.jsonl training_data\ir_validation_examples.jsonl training_data\ir_test_examples.jsonl
```

## Known Limitations

- OR rendering is disabled by default and is not production-routed.
- No subquery-backed IN, EXISTS, HAVING, CASE, windows, or set operations.
- v2-to-v1 compatibility rejects OR and NOT rather than approximating them.
- The current trained model still emits QueryIR v1.

## Phase 2C Recommendation

Add one production-gated advanced predicate capability next, preferably
BETWEEN/date-range normalization or null predicate support in the broader
runtime validation path, before attempting HAVING, CASE, subqueries, or model
architecture changes.
