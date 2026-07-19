# Test Suite Cleanup Report

Generated: 2026-07-15T17:11:11+00:00

| Metric | Before | After |
| --- | ---: | ---: |
| Test files | 202 | 151 |
| Active test files | 137 | 86 |
| Collected tests | 974 | 958 |
| Unit files | 120 | 73 |
| Integration files | 5 | 3 |
| Regression files | 2 | 2 |
| Legacy files | 65 | 65 |
| Duplicate files | 0 | 0 |
| Excluded files | 65 | 65 |
| Runtime | 148.57s | 45.55s |
| Line coverage | n/a | n/a |
| Branch coverage | n/a | n/a |
| Flaky tests | n/a | n/a |
| Requirements mapped | 19 | 10 |

## Tests Merged

- `tests/unit/ir/test_query_ir_v2_models.py` <= `tests/test_query_ir_v2_models.py`, `tests/test_query_ir_v2_boolean_models.py`, `tests/test_query_ir_v2_serialization.py`, `tests/test_query_ir_v2_fingerprint.py`
- `tests/unit/ir/test_query_ir_v2_validation.py` <= `tests/test_query_ir_v2_validation.py`, `tests/test_query_ir_v2_depth_limits.py`, `tests/test_query_ir_v2_boolean_validation.py`, `tests/test_query_ir_v2_boolean_depth_limits.py`, `tests/test_query_ir_v2_boolean_canonicalization.py`
- `tests/unit/ir/test_query_ir_v2_conversion.py` <= `tests/test_sql_to_query_ir_v2_boolean_conversion.py`
- `tests/unit/ir/test_query_ir_v2_rendering.py` <= `tests/test_query_ir_v2_boolean_renderer.py`
- `tests/unit/ir/test_query_ir_migration.py` <= `tests/test_query_ir_v1_to_v2_migration.py`, `tests/test_query_ir_v1_boolean_migration.py`, `tests/test_query_ir_v2_to_v1_compatibility.py`, `tests/test_query_ir_v2_boolean_v1_compatibility.py`, `tests/test_query_ir_version_loader.py`, `tests/test_query_ir_v2_renderer_parity.py`
- `tests/integration/test_query_ir_v2_execution.py` <= `tests/test_query_ir_v2_boolean_precedence.py`, `tests/test_query_ir_v2_boolean_execution_equivalence.py`
- `tests/unit/capabilities/test_capability_pipeline.py` <= `tests/test_capability_artifact_schema.py`, `tests/test_capability_dataset_reporting.py`, `tests/test_capability_or_filter_consistency.py`, `tests/test_capability_taxonomy.py`, `tests/test_capability_training_batch.py`, `tests/test_sql_capability_extractor.py`, `tests/test_training_inference_capability_parity.py`, `tests/test_unsupported_example_task_masks.py`
- `tests/unit/data/test_dataset_pipeline.py` <= `tests/test_20_dataset_split_manager.py`, `tests/test_21_dataset_leakage_checker.py`, `tests/test_22_generic_ir_corpus_builder.py`, `tests/test_24_dataset_scale_evaluator.py`, `tests/test_dataset_leakage_domain.py`, `tests/test_dataset_split_integrity.py`, `tests/test_verify_datasets.py`, `tests/test_sql_partial_supervision.py`
- `tests/unit/retrieval/test_retrieval_pipeline.py` <= `tests/test_04_retrieval_runtime.py`, `tests/test_23_retrieval_rag_index.py`, `tests/test_train_retriever_from_datasets.py`
- `tests/unit/execution/test_execution_evaluation.py` <= `tests/test_50_sql_canonicalizer.py`, `tests/test_51_sql_structure_comparator.py`, `tests/test_52_result_comparator.py`, `tests/test_53_execution_aware_evaluation.py`
- `tests/unit/runtime/test_generic_planner_and_grounding.py` <= `tests/test_10_generic_table_intent.py`, `tests/test_11_generic_join_policy.py`, `tests/test_60_schema_profiler.py`, `tests/test_61_glossary_generator.py`, `tests/test_62_semantic_mapper.py`, `tests/test_63_ambiguity_detector.py`, `tests/test_64_clarification_runtime.py`, `tests/test_120_schema_value_index.py`, `tests/test_121_filter_value_extractor.py`, `tests/test_122_filter_grounding.py`, `tests/test_123_projection_resolution.py`, `tests/test_124_dimension_resolution.py`
- `tests/integration/test_database_and_connected_regression.py` <= `tests/test_03_database_connectors.py`, `tests/test_12_generic_postgres_schema_runtime.py`, `tests/test_65_connected_db_regression_generator.py`, `tests/test_66_connected_db_regression_runner.py`, `tests/test_134_database_integration.py`
- `tests/unit/execution/test_sql_validation_and_safety.py` <= `tests/test_02_sql_validation.py`, `tests/test_105_sql_validation_policy.py`, `tests/test_119_renderer_attribution.py`, `tests/test_132_sqlite_prediction_cache.py`, `tests/test_133_telemetry_privacy.py`

## Consolidation Skipped

- `tests/unit/feedback/test_feedback_pipeline.py`: top-level namespace collision
- `tests/integration/test_model_lifecycle_pipeline.py`: top-level namespace collision
- `tests/unit/model/test_neural_training_components.py`: top-level namespace collision
- `tests/e2e/test_application_and_training_smoke.py`: top-level namespace collision

## Gates

- T1_inventory: PASS
- T2_requirement_mapping: PASS
- T3_consolidation: PASS
- T4_legacy_resolution: PASS
- T5_coverage_preservation: PASS
- T6_final_execution: REVIEW_REQUIRED

## Notes

- Full-suite and lane validation passed after consolidation. Line and branch coverage were not measured in this pass.
- Standalone runtime smoke and production bundle smoke were not run; the local `artifacts/model_bundle/current` production bundle is still absent.
- Merged source and target AST test-body counts match exactly for all 13 consolidated clusters.
- Merged source files are recorded in `artifacts/repository_cleanup/test_deletion_manifest.json` with coverage preserved by the target module.
