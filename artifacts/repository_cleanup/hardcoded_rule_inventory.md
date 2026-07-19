# Hardcoded Rule Inventory

| Path | Patterns | Classification |
| --- | --- | --- |
| README.md | customers, orders | review_required |
| app/streamlit_app.py | customers | review_required |
| artifacts/repository_cleanup/archive_manifest.json | revenue | review_required |
| artifacts/repository_cleanup/before_after_stats.json | customers, orders, revenue | review_required |
| artifacts/repository_cleanup/cleanup_report.md | revenue | review_required |
| artifacts/repository_cleanup/configuration_usage_report.json | customers, orders | review_required |
| artifacts/repository_cleanup/hardcoded_rule_inventory.json | customers, orders, order_items, revenue, quantity * unit_price, quantity * price, APPROVED_REVENUE_EXPR, sample schema, normalize_runtime_result | review_required |
| artifacts/repository_cleanup/hardcoded_rule_inventory.md | customers, orders, order_items, revenue, quantity * unit_price, quantity * price, APPROVED_REVENUE_EXPR, sample schema, normalize_runtime_result | review_required |
| artifacts/repository_cleanup/repository_inventory.json | customers, orders, revenue | review_required |
| artifacts/repository_cleanup/test_cleanup_report.json | revenue | review_required |
| artifacts/repository_cleanup/test_inventory.json | customers, orders, revenue | review_required |
| artifacts/repository_cleanup/test_suite_cleanup_report.json | revenue | review_required |
| data/splits/semantic_v2/development_validation_ids.json | customers | review_required |
| data/splits/semantic_v2/split_manifest.json | customers | review_required |
| data/splits/semantic_v2/train_ids.json | customers | review_required |
| data/synonyms.yaml | customers, orders, order_items, revenue | configuration_driven_semantic_defaults |
| dataset_training/ir_corpus_builder.py | customers, orders, order_items | review_required |
| docs/architecture/training_data_lineage.md | customers | documentation_example_or_report |
| docs/reports/enterprise_nl2sql_system_audit.md | orders, revenue, sample schema, normalize_runtime_result | documentation_example_or_report |
| docs/reports/remaining_program_owner_matrix.md | order_items, APPROVED_REVENUE_EXPR | documentation_example_or_report |
| docs/specs/query_ir_v1_frozen_spec.md | revenue | documentation_example_or_report |
| evaluation/adaptive_router_benchmark_cases.jsonl | customers, orders, order_items, revenue | test_or_benchmark_fixture |
| evaluation/fixtures/controlled_evaluation.sql | orders | test_or_benchmark_fixture |
| evaluation/fixtures/controlled_evaluation_cases.jsonl | orders | test_or_benchmark_fixture |
| evaluation/generic_benchmark_cases.jsonl | customers, orders, revenue | test_or_benchmark_fixture |
| evaluation/golden_runtime_tests.jsonl | customers, orders, order_items, revenue | test_or_benchmark_fixture |
| evaluation/golden_tests.jsonl | customers, orders, revenue | test_or_benchmark_fixture |
| evaluation/hard_negative_cases.jsonl | customers, orders, order_items, revenue | test_or_benchmark_fixture |
| evaluation/hybrid_benchmark_cases.jsonl | customers, orders, order_items, revenue | test_or_benchmark_fixture |
| evaluation/hybrid_router_eval_cases.jsonl | customers, orders | test_or_benchmark_fixture |
| evaluation/ir_conversion_golden.jsonl | customers, orders, order_items, revenue | test_or_benchmark_fixture |
| evaluation/neural_ir_eval_cases.jsonl | customers, orders, revenue, quantity * price | test_or_benchmark_fixture |
| evaluation/neural_ir_v2_eval_cases.jsonl | customers, orders, order_items, revenue | test_or_benchmark_fixture |
| evaluation/option_a_eval_cases.jsonl | customers, orders, revenue, quantity * price | test_or_benchmark_fixture |
| evaluation/option_a_v2_eval_cases.jsonl | customers, orders, order_items, revenue | test_or_benchmark_fixture |
| evaluation/query_ir_v2_boolean_eval_cases.jsonl | customers, orders | test_or_benchmark_fixture |
| generic_planner/generic_slot_resolver.py | customers, orders, order_items | runtime_or_gate_review_required |
| generic_planner/table_intent_resolver.py | revenue | runtime_or_gate_review_required |
| inference/candidate_reranker.py | orders, revenue | runtime_or_gate_review_required |
| inference/prediction_confidence.py | revenue | runtime_or_gate_review_required |
| inference/prediction_orchestrator.py | revenue | runtime_or_gate_review_required |
| inference/schema_aware_mapper.py | customers, orders, order_items, revenue | runtime_or_gate_review_required |
| inference/slot_resolver.py | customers, orders | runtime_or_gate_review_required |
| inference/template_selector.py | revenue | runtime_or_gate_review_required |
| ir/ir_validator.py | orders, order_items, revenue | review_required |
| ir/option_c_to_ir.py | revenue | review_required |
| ir/query_ir_v2_rendering/expressions.py | order_items | review_required |
| ir/query_ir_v2_roundtrip.py | orders | review_required |
| ir/semantic_metric_resolver.py | order_items, revenue | review_required |
| ir/sql_to_ir_converter.py | order_items, APPROVED_REVENUE_EXPR | review_required |
| model_bundle/bundle_validator.py | orders | review_required |
| neural_ir/candidate_builder.py | customers, orders, revenue | review_required |
| neural_ir/confidence_calibrator.py | revenue | review_required |
| neural_ir/dataset_quality.py | revenue | review_required |
| neural_ir/error_analysis.py | revenue | review_required |
| neural_ir/hard_negative_builder.py | orders, order_items, revenue | review_required |
| neural_ir/ir_label_encoder.py | order_items, revenue | review_required |
| neural_ir/ir_repair.py | order_items, revenue | review_required |
| neural_ir/option_a_to_ir.py | order_items, revenue | review_required |
| neural_ir/schema_linearizer.py | revenue | review_required |
| neural_ir/schema_linker.py | customers, revenue | review_required |
| neural_ir/training_curriculum.py | order_items | review_required |
| nl2sql_v1/slot_extractor.py | customers, orders, revenue | review_required |
| orchestration/step_runner.py | orders | review_required |
| quality_gates/regression_suite.py | customers, orders, order_items, revenue | runtime_or_gate_review_required |
| retriever/retrieval_nl2sql_model.py | normalize_runtime_result | runtime_or_gate_review_required |
| reward/reward_features.py | revenue | review_required |
| scripts/audit_generic_nl2sql_readiness.py | customers, orders | review_required |
| scripts/create_sample_db.py | customers, orders, order_items | review_required |
| scripts/generate_repository_cleanup_inventory.py | customers, orders, order_items, revenue, quantity * unit_price, quantity * price, APPROVED_REVENUE_EXPR, sample schema, normalize_runtime_result | review_required |
| scripts/run_golden_tests.py | orders | review_required |
| semantic_layer/schema_profiler.py | customers | review_required |
| tests/integration/test_database_and_connected_regression.py | orders, revenue | test_or_benchmark_fixture |
| tests/integration/test_query_ir_v2_execution.py | customers | test_or_benchmark_fixture |
| tests/legacy/test_active_pipeline.py | customers, orders, revenue | test_or_benchmark_fixture |
| tests/legacy/test_bird_adapter.py | orders | test_or_benchmark_fixture |
| tests/legacy/test_build_ir_training_data.py | orders, revenue | test_or_benchmark_fixture |
| tests/legacy/test_candidate_builder.py | customers, orders | test_or_benchmark_fixture |
| tests/legacy/test_central_sql_validator.py | customers, orders | test_or_benchmark_fixture |
| tests/legacy/test_confidence_breakdown.py | customers, orders, revenue | test_or_benchmark_fixture |
| tests/legacy/test_confidence_caps.py | customers, orders, revenue | test_or_benchmark_fixture |
| tests/legacy/test_corpus_builder.py | orders | test_or_benchmark_fixture |
| tests/legacy/test_dataset_models.py | orders | test_or_benchmark_fixture |
| tests/legacy/test_date_filter_runtime.py | orders | test_or_benchmark_fixture |
| tests/legacy/test_evaluate_runtime.py | customers, orders, revenue | test_or_benchmark_fixture |
| tests/legacy/test_hard_negative_builder.py | orders, order_items, revenue | test_or_benchmark_fixture |
| tests/legacy/test_hybrid_benchmark.py | orders | test_or_benchmark_fixture |
| tests/legacy/test_hybrid_router.py | customers, orders | test_or_benchmark_fixture |
| tests/legacy/test_hybrid_router_calibration.py | orders | test_or_benchmark_fixture |
| tests/legacy/test_inference_runtime.py | customers, orders, order_items, revenue | test_or_benchmark_fixture |
| tests/legacy/test_ir_label_encoder.py | customers, orders | test_or_benchmark_fixture |
| tests/legacy/test_ir_repair.py | customers, orders, order_items, revenue | test_or_benchmark_fixture |
| tests/legacy/test_ir_roundtrip_validator.py | customers, orders, revenue | test_or_benchmark_fixture |
| tests/legacy/test_ir_to_sql_renderer.py | customers, orders, revenue | test_or_benchmark_fixture |
| tests/legacy/test_ir_validator.py | orders, revenue | test_or_benchmark_fixture |
| tests/legacy/test_main_flow.py | customers, orders, order_items, revenue | test_or_benchmark_fixture |
| tests/legacy/test_neural_ir_dataset.py | customers, orders | test_or_benchmark_fixture |
| tests/legacy/test_neural_ir_tokenizer.py | customers | test_or_benchmark_fixture |
| tests/legacy/test_neural_ir_vocab.py | customers | test_or_benchmark_fixture |
| tests/legacy/test_option_a_candidate_masks.py | customers, orders | test_or_benchmark_fixture |
| tests/legacy/test_option_a_curriculum_training.py | orders | test_or_benchmark_fixture |
| tests/legacy/test_option_a_error_analysis.py | customers, orders | test_or_benchmark_fixture |
| tests/legacy/test_option_a_predictor.py | customers, orders | test_or_benchmark_fixture |
| tests/legacy/test_option_a_to_ir.py | customers, orders | test_or_benchmark_fixture |
| tests/legacy/test_option_a_training_smoke.py | orders | test_or_benchmark_fixture |
| tests/legacy/test_option_a_v2_evaluator.py | orders | test_or_benchmark_fixture |
| tests/legacy/test_option_a_v2_training_smoke.py | orders | test_or_benchmark_fixture |
| tests/legacy/test_option_c_to_ir.py | customers, orders | test_or_benchmark_fixture |
| tests/legacy/test_prediction_orchestrator.py | customers, orders, revenue | test_or_benchmark_fixture |
| tests/legacy/test_product_revenue_semantics.py | orders, order_items, revenue | test_or_benchmark_fixture |
| tests/legacy/test_query_ir_models.py | customers, orders, revenue | test_or_benchmark_fixture |
| tests/legacy/test_retrieval_nl2sql_model.py | customers, orders, revenue | test_or_benchmark_fixture |
| tests/legacy/test_runtime_end_to_end.py | customers | test_or_benchmark_fixture |
| tests/legacy/test_runtime_golden_cases.py | customers, orders, order_items, revenue | test_or_benchmark_fixture |
| tests/legacy/test_safe_preview_sql.py | customers | test_or_benchmark_fixture |
| tests/legacy/test_schema_linearizer.py | customers, orders | test_or_benchmark_fixture |
| tests/legacy/test_schema_linker.py | customers, orders | test_or_benchmark_fixture |
| tests/legacy/test_simple_filter_runtime.py | customers, orders | test_or_benchmark_fixture |
| tests/legacy/test_spider_adapter.py | orders | test_or_benchmark_fixture |
| tests/legacy/test_sql_feature_extractor.py | orders | test_or_benchmark_fixture |
| tests/legacy/test_sql_pattern_classifier.py | customers, orders | test_or_benchmark_fixture |
| tests/legacy/test_sql_to_ir_converter.py | customers, orders, order_items, revenue | test_or_benchmark_fixture |
| tests/legacy/test_sql_to_ir_rules.py | customers, orders, revenue | test_or_benchmark_fixture |
| tests/legacy/test_sql_validator.py | customers | test_or_benchmark_fixture |
| tests/legacy/test_template_selector_filters.py | revenue | test_or_benchmark_fixture |
| tests/legacy/test_train_retriever_from_datasets.py | orders | test_or_benchmark_fixture |
| tests/legacy/test_validate_ir_corpus.py | orders, revenue | test_or_benchmark_fixture |
| tests/legacy/test_wikisql_adapter.py | revenue | test_or_benchmark_fixture |
| tests/query_ir_v2_boolean_helpers.py | customers, orders | test_or_benchmark_fixture |
| tests/query_ir_v2_test_helpers.py | customers, orders, order_items, revenue | test_or_benchmark_fixture |
| tests/test_01_core_ir.py | customers, orders, order_items, revenue | test_or_benchmark_fixture |
| tests/test_05_neural_runtime.py | customers, orders | test_or_benchmark_fixture |
| tests/test_06_adaptive_router.py | customers, orders | test_or_benchmark_fixture |
| tests/test_07_training_data_pipeline.py | orders | test_or_benchmark_fixture |
| tests/test_08_streamlit_app_helpers.py | orders | test_or_benchmark_fixture |
| tests/test_09_end_to_end_smoke.py | customers, orders | test_or_benchmark_fixture |
| tests/test_102_candidate_relation_graph.py | orders | test_or_benchmark_fixture |
| tests/test_106_semantic_pass.py | orders | test_or_benchmark_fixture |
| tests/test_108_query_ir_diff_enhanced.py | orders | test_or_benchmark_fixture |
| tests/test_118_route_diagnostics.py | orders, revenue | test_or_benchmark_fixture |
| tests/test_125_grounding_e2e_regression.py | customers, orders | test_or_benchmark_fixture |
| tests/test_33_feedback_index.py | orders | test_or_benchmark_fixture |
| tests/test_40_gold_comparator.py | orders | test_or_benchmark_fixture |
| tests/test_41_error_classifier.py | orders | test_or_benchmark_fixture |
| tests/test_42_hard_negative_generator.py | orders | test_or_benchmark_fixture |
| tests/test_43_correction_generator.py | orders, revenue | test_or_benchmark_fixture |
| tests/test_45_self_training_loop.py | orders | test_or_benchmark_fixture |
| tests/test_46_prediction_runner.py | orders | test_or_benchmark_fixture |
| tests/test_81_diagnostics_and_telemetry.py | orders | test_or_benchmark_fixture |
| tests/test_cache_identity_domain.py | orders | test_or_benchmark_fixture |
| tests/test_gate1_queryir_v2_advanced.py | customers, orders | test_or_benchmark_fixture |
| tests/test_gate1_validation_migration.py | orders | test_or_benchmark_fixture |
| tests/test_gate2_data_readiness.py | customers, orders | test_or_benchmark_fixture |
| tests/test_gate3_architecture.py | customers, orders | test_or_benchmark_fixture |
| tests/test_gate5_evaluation.py | customers, orders | test_or_benchmark_fixture |
| tests/test_gate6_production.py | orders | test_or_benchmark_fixture |
| tests/test_training_audit_fixes.py | customers, orders, normalize_runtime_result | test_or_benchmark_fixture |
| tests/unit/capabilities/test_capability_pipeline.py | customers, orders, revenue | test_or_benchmark_fixture |
| tests/unit/data/test_dataset_pipeline.py | customers, orders | test_or_benchmark_fixture |
| tests/unit/execution/test_sql_validation_and_safety.py | customers, orders | test_or_benchmark_fixture |
| tests/unit/ir/test_query_ir_migration.py | customers, orders, order_items, revenue | test_or_benchmark_fixture |
| tests/unit/ir/test_query_ir_v2_conversion.py | customers | test_or_benchmark_fixture |
| tests/unit/ir/test_query_ir_v2_models.py | orders, order_items, revenue | test_or_benchmark_fixture |
| tests/unit/ir/test_query_ir_v2_rendering.py | customers | test_or_benchmark_fixture |
| tests/unit/ir/test_query_ir_v2_validation.py | customers, orders | test_or_benchmark_fixture |
| tests/unit/retrieval/test_retrieval_pipeline.py | customers, orders, revenue | test_or_benchmark_fixture |
| tests/unit/runtime/test_generic_planner_and_grounding.py | customers, orders, revenue | test_or_benchmark_fixture |
| training/schema_graph.py | orders | review_required |
| training_data/examples.jsonl | customers, orders, order_items, revenue | review_required |
