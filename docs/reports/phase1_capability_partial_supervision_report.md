# Phase 1 Capability and Partial Supervision Report

Date: 2026-07-11

## Frozen Baseline

- Git commit: `fbffe6b18fb2bcfea6cf8945b825f9b8b00d7299`
- Canonical training config: `configs/neural_training_default.yaml`
- Canonical model version: `schema_aware_queryir_v1`
- Explicit QueryIR schema version: absent from `ir/query_ir_models.py`
- Relation-aware attention in canonical config: disabled (`relation_aware_attention.enabled: false`)
- Baseline unsupported report: `artifacts/generic_training/unsupported_sql_report.json`
- Baseline corpus quality report: `artifacts/generic_training/corpus_quality_report.json`
- Baseline model/evaluation artifact areas: `artifacts/model_bundle`, `artifacts/pipeline`, `artifacts/evaluation`, `models/tfidf_retriever.joblib`

## Baseline Readiness Audits

- `python scripts/audit_integration_readiness.py`: PASS, 24 passed, 0 failed
- `python scripts/audit_execution_pipeline_readiness.py`: PASS, 5 passed, 0 failed
- `python scripts/audit_generic_nl2sql_readiness.py`: PASS, 13 passed, 0 failed
- `python scripts/audit_self_training_readiness.py`: PASS, 6 passed, 0 failed

## Baseline Unsupported Counts

From `artifacts/generic_training/unsupported_sql_report.json`:

- Unsupported examples: 653
- Unsupported by dataset: `wikisql=185`, `bird-mini=468`
- Top unsupported features: `unsupported_join=184`, `validator_failed=143`, `unsupported_expression=90`, `case_expression=75`, `nested_query=63`
- Advanced features previously collapsed into unsupported buckets: `set_operation=3`, `or_filter=17`, `having_clause=5`, `window_function=3`

## Phase 1 Implementation

Added:

- Multi-label capability enum and separate safety labels in `capabilities/taxonomy.py`
- Strict typed contracts in `capabilities/contracts.py`
- SQLGlot AST capability and partial-supervision extractor in `capabilities/sql_capability_extractor.py`
- Capability distribution and unsupported-retention reporting in `capabilities/reporting.py`
- Additive artifact builder in `training/build_capability_annotations.py`
- Corpus-builder integration that extracts capability labels before QueryIR conversion and masks unsupported full-IR loss
- Neural batch contract fields: `capability_labels` and `task_masks`

No recursive QueryIR v2, advanced SQL rendering, relation-aware activation, learned loss weighting, semantic RAG, or hierarchical decoder work was implemented.

## Generated Artifacts

The Phase 1 artifact builder reads the existing frozen split files and writes additive outputs only:

- `data/processed/generic_ir_capability_annotations.jsonl`
- `data/processed/generic_ir_partial_supervision.jsonl`
- `artifacts/generic_training/capability_distribution_report.json`
- `artifacts/generic_training/capability_distribution_report.md`
- `artifacts/generic_training/unsupported_example_retention_report.json`
- `artifacts/generic_training/unsupported_example_retention_report.md`

The capability report includes unseen DB examples, so it reports 1000 total rows. The existing corpus-quality report excludes unseen DB from its supported-count denominator and reports 950 rows.

## Capability Statistics After

From `artifacts/generic_training/capability_distribution_report.json`:

- Total examples: 1000
- Parseable SQL: 1000
- Full QueryIR-supported: 347
- Partial-supervision-only: 653
- Auxiliary-training eligible: 1000
- Capabilities observed: 25
- Zero-coverage capabilities: 9
- Partial-supervision extraction coverage: 1.0
- Conservative support-policy accuracy against current full-IR support: 0.57

High-frequency capabilities:

- SIMPLE_SELECT: 1000
- FILTER: 977
- MULTIPLE_FILTERS: 501
- AGGREGATION: 409
- ONE_HOP_JOIN: 304
- MULTI_HOP_JOIN: 103
- ORDER_BY: 101
- LIMIT: 100

Rare or zero-coverage warnings are preserved in the capability report and should guide targeted data collection before model-head training.

## Unsupported Retention

From `artifacts/generic_training/unsupported_example_retention_report.json`:

- Unsupported examples: 653
- Retained for auxiliary supervision: 653
- Not retained: 0
- Full QueryIR loss masked: 653

Auxiliary labels retained include capability, table, column, filter, aggregation, join-edge, subquery, window, set-operation, complexity, and contrastive schema-linking masks where reliable.

## Regression Results

- Phase 1 tests: `29 passed`
- Existing generic corpus builder tests: `3 passed`
- Combined Phase 1 plus existing corpus-builder tests: `32 passed`
- Supported runtime/core regression tests: `44 passed`
- Final readiness audits: integration `24/0`, execution pipeline `5/0`, generic NL2SQL `13/0`, self-training `6/0`

Existing supported-query behavior remains backward compatible: the active split files, runtime routing, renderer behavior, model architecture, and production bundle policy were not changed.

`python scripts/run_golden_tests.py` did not reach its golden cases because the existing retrieval artifact is missing `artifacts/option_c_model/sklearn_artifact_metadata.json` and the runtime correctly fails closed. No model artifact was rebuilt in this phase.

## Known Limitations

- QueryIR v1 still cannot render subqueries, window functions, set operations, HAVING, CASE, or OR filters.
- Capability labels are deterministic AST labels, not learned predictions.
- The neural capability head is not enabled in this phase.
- Class thresholds, average precision, and promotion integration require a future trained capability head.
- Some capability families have rare or zero coverage in the current frozen corpus.

## Phase 2 Recommendation

Phase 2 QueryIR v2 can begin after this Phase 1 branch is reviewed and accepted. The next phase should use the new capability report to prioritize recursive QueryIR support for subqueries, set operations, windows, HAVING, CASE, and OR filter rendering.
