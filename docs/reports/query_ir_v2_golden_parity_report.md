# QueryIR v2 Golden Parity Report

Phase: 2A

Scope: QueryIR v1 to QueryIR v2 migration, v2-to-v1 compatibility conversion,
and rendering through the existing v1 renderer.

## Corpus

The Phase 2A parity corpus is the local supported QueryIR v1 generated corpus:

- `training_data/ir_training_examples.jsonl`
- `training_data/ir_validation_examples.jsonl`
- `training_data/ir_test_examples.jsonl`

The unit-level parity test also keeps a compact representative corpus in
`tests/query_ir_v2_test_helpers.py` covering metric summary, metric by
dimension, count by dimension, simple filters, date-grain trends, joins,
binary metric expressions, ordering, and limits.

## Method

For each supported v1 example:

1. Render SQL through `IRToSQLRenderer`.
2. Migrate v1 to QueryIR v2 with `migrate_v1_to_v2`.
3. Convert v2 through `QueryIRV2RendererAdapter`.
4. Render SQL through the existing v1 renderer.
5. Normalize/compare SQL structure with `SQLStructureComparator`.
6. Assert structural semantic equivalence.

## Results

| Metric | Count |
| --- | ---: |
| Total migrated | 3075 |
| Total parity passed | 3075 |
| Total migration failures | 0 |
| Total SQL normalization differences | 0 |
| Unsupported conversion count | 0 |

The full corpus command was:

```powershell
python -m ir.query_ir_v2_parity training_data\ir_training_examples.jsonl training_data\ir_validation_examples.jsonl training_data\ir_test_examples.jsonl
```

The focused executable assertion is
`tests/test_query_ir_v2_renderer_parity.py::test_v2_compatibility_adapter_preserves_supported_v1_renderer_semantics`.

## Acceptance status

The v2 compatibility path preserves every supported v1 behavior in the Phase
2A corpus by converting v2 back to the v1-compatible subset before rendering.

Advanced v2 nodes intentionally fail before rendering with
`unsupported_v2_rendering_capability`.

## Known limitations

This report does not claim support for HAVING, OR rendering, CASE rendering,
subqueries, windows, set operations, relation-aware attention, model retraining,
or production routing changes.
