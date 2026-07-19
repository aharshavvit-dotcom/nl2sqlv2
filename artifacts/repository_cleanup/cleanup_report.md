# Repository Cleanup Report

Generated: 2026-07-15T17:19:37+00:00

## Executive Summary

This pass created the cleanup inventory, repository map, manifests, artifact governance reports, and low-risk deletion evidence. It did not delete production-critical bundles, frozen splits, model checkpoints, raw datasets, or run-scoped reports.

## Safety Statement

- Current production bundle exists locally: False. Missing current bundle is a release risk, not a cleanup target.
- Singleton candidate bundle exists locally: True. It is retained pending artifact review.
- Run-scoped candidate bundles found: 11. They are retained pending retention policy application.
- Active split version: semantic_v2. Frozen split manifests are retained.

## Low-Risk Deletions

| Path | Reason | Risk |
| --- | --- | --- |
| .pytest_cache | Low-risk generated Python/test cache or local smoke log. | low |
| app/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| capabilities/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| clarification/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| connected_db_testing/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| dataset_training/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| datasets/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| db/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| deployment/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| evaluation/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| execution/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| execution_eval/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| feedback/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| generic_planner/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| inference/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| inference/grounding/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| ir/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| ir/query_ir_v2_rendering/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| model_bundle/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| model_registry/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| model_selection/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| neural_ir/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| neural_optimization/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| nl2sql_v1/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| orchestration/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| quality_gates/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| retrieval/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| retriever/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| reward/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| scripts/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| self_training/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| semantic_layer/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| tests/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| tests/fixtures/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| tests/integration/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| tests/legacy/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| tests/unit/capabilities/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| tests/unit/data/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| tests/unit/execution/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| tests/unit/ir/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| tests/unit/retrieval/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| tests/unit/runtime/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| training/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| training_ir/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |
| validation/__pycache__ | Low-risk generated Python/test cache or local smoke log. | low |

## Archive / Review Candidates

| Path | Reason | Action |
| --- | --- | --- |
| .github/workflows/ci.yml | Runs automated validation outside the local development machine. | review_before_commit |
| docs/reports/enterprise_nl2sql_system_audit.md | Some reports preserve decision history, but run-specific reports should not be primary docs. | archive_or_move_after_review |
| docs/reports/phase1_capability_partial_supervision_report.md | Some reports preserve decision history, but run-specific reports should not be primary docs. | archive_or_move_after_review |
| docs/reports/query_ir_v2_boolean_evaluation_report.md | Some reports preserve decision history, but run-specific reports should not be primary docs. | archive_or_move_after_review |
| docs/reports/query_ir_v2_golden_parity_report.md | Some reports preserve decision history, but run-specific reports should not be primary docs. | archive_or_move_after_review |
| docs/reports/remaining_program_baseline.json | Some reports preserve decision history, but run-specific reports should not be primary docs. | archive_or_move_after_review |
| docs/reports/remaining_program_owner_matrix.md | Some reports preserve decision history, but run-specific reports should not be primary docs. | archive_or_move_after_review |
| docs/reports/run_20260712T044706_01b12c98_quality_gate_diagnosis.md | Some reports preserve decision history, but run-specific reports should not be primary docs. | archive_or_move_after_review |
| docs/reports/test_suite_cleanup_report.md | Some reports preserve decision history, but run-specific reports should not be primary docs. | archive_or_move_after_review |
| nl2sql_v1/README_LEGACY.md | Migration and legacy tests still compare or validate older behavior. | retain_pending_consolidation |
| nl2sql_v1/__init__.py | Migration and legacy tests still compare or validate older behavior. | retain_pending_consolidation |
| nl2sql_v1/engine.py | Migration and legacy tests still compare or validate older behavior. | retain_pending_consolidation |
| nl2sql_v1/executor.py | Migration and legacy tests still compare or validate older behavior. | retain_pending_consolidation |
| nl2sql_v1/feedback.py | Migration and legacy tests still compare or validate older behavior. | retain_pending_consolidation |
| nl2sql_v1/join_resolver.py | Migration and legacy tests still compare or validate older behavior. | retain_pending_consolidation |
| nl2sql_v1/renderer.py | Migration and legacy tests still compare or validate older behavior. | retain_pending_consolidation |
| nl2sql_v1/retriever.py | Migration and legacy tests still compare or validate older behavior. | retain_pending_consolidation |
| nl2sql_v1/schema.py | Migration and legacy tests still compare or validate older behavior. | retain_pending_consolidation |
| nl2sql_v1/schema_matcher.py | Migration and legacy tests still compare or validate older behavior. | retain_pending_consolidation |
| nl2sql_v1/slot_extractor.py | Migration and legacy tests still compare or validate older behavior. | retain_pending_consolidation |
| nl2sql_v1/template_adapter.py | Migration and legacy tests still compare or validate older behavior. | retain_pending_consolidation |
| nl2sql_v1/validator.py | Migration and legacy tests still compare or validate older behavior. | retain_pending_consolidation |
| tests/legacy/README.md | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_active_pipeline.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_attention_model.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_bird_adapter.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_build_ir_training_data.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_candidate_builder.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_canonical_runtime_only.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_central_sql_validator.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_confidence_breakdown.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_confidence_caps.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_corpus_builder.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_dataset_models.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_date_filter_runtime.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_download_paths.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_evaluate_retriever.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_evaluate_runtime.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_hard_negative_builder.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_hybrid_benchmark.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_hybrid_router.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_hybrid_router_calibration.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_inference_runtime.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_ir_conversion_golden.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_ir_label_encoder.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_ir_repair.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_ir_roundtrip_validator.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_ir_to_sql_renderer.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_ir_training_dataset.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_ir_validator.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_main_flow.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_neural_ir_dataset.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_neural_ir_tokenizer.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_neural_ir_vocab.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_no_active_legacy_runtime_usage.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_option_a_candidate_masks.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_option_a_confidence_calibrator.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_option_a_curriculum_training.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_option_a_error_analysis.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_option_a_model_forward.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_option_a_predictor.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_option_a_to_ir.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_option_a_training_smoke.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_option_a_v2_evaluator.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_option_a_v2_training_smoke.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_option_c_to_ir.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_pointer_network.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_prediction_orchestrator.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_product_revenue_semantics.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_query_ir_models.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_retrieval_nl2sql_model.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_runtime_end_to_end.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_runtime_golden_cases.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_safe_preview_sql.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_schema_linearizer.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_schema_linker.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_simple_filter_runtime.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_spider_adapter.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_sql_feature_extractor.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_sql_pattern_classifier.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_sql_to_ir_converter.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_sql_to_ir_rules.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_sql_validator.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_template_selector_filters.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_train_retriever_from_datasets.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_validate_ir_corpus.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_wikisql_adapter.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/legacy/test_yaml_synonym_usage.py | They guard migrations and older APIs that may still be referenced. | retain_pending_consolidation |
| tests/test_catalog.yaml | Cleanup is safe only when behavior remains covered. | review_before_commit |
| training_ir/__init__.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/analyze_ir_dataset_quality.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/analyze_option_a_errors.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/benchmark_hybrid_system.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/build_hard_negative_data.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/build_ir_training_data.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/calibrate_hybrid_router.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/calibrate_option_a_confidence.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/evaluate_ir_conversion.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/evaluate_option_a_model.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/evaluate_option_a_v2_model.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/predict_with_option_a.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/repair_option_a_predictions.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/run_option_a_ablation.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/train_option_a_curriculum.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/train_option_a_model.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/train_option_a_v2_model.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| training_ir/validate_ir_corpus.py | Some ablation and legacy tests still exercise these paths. | retain_pending_consolidation |
| artifacts/audit/ | Captures reproducibility evidence and local runtime bundles. | retain_pending_artifact_review |
| artifacts/model_bundle/ | Captures reproducibility evidence and local runtime bundles. | retain_pending_artifact_review |
| data/processed/ | Runtime synonym defaults and immutable split membership are required for reproducible training. | retain_pending_dataset_review |
| data/processed_smoke_bird_full/ | Runtime synonym defaults and immutable split membership are required for reproducible training. | retain_pending_dataset_review |
| data/processed_smoke_bird_mini/ | Runtime synonym defaults and immutable split membership are required for reproducible training. | retain_pending_dataset_review |
| data/raw/ | Runtime synonym defaults and immutable split membership are required for reproducible training. | retain_pending_dataset_review |

## Consolidation Candidates

| Area | Classification | Canonical owner | Cleanup action |
| --- | --- | --- | --- |
| retrieval vs retriever | CONSOLIDATE | retrieval/ for index/reranker infrastructure; retriever/ currently retains runtime RetrievalNL2SQLModel wrapper | Plan import migration before deleting retriever/. |
| training vs training_ir | REVIEW_REQUIRED | training/train_model.py for integrated pipeline | Keep until commands are replaced or archived with tests/docs updated. |
| dataset_training vs datasets | KEEP_BOTH | datasets/ adapters; dataset_training/ corpus/split/leakage builders | No merge without API design. |
| models vs model_bundle | KEEP_BOTH | model_bundle/ for production bundles; models/ only ignored local artifact placeholder | Keep models/.gitkeep only. |
| evaluation reports in docs/reports vs artifacts/ | ARCHIVE | artifacts/pipeline/runs/<run_id>/reports for run-scoped generated reports | Move after review; no automatic deletion in this pass. |

## Documentation Cleanup

- Canonical docs retained: 14
- Report docs needing archive/move review: 7
- Legacy docs retained pending retirement: 3

## Artifact Retention Policy

- Active production bundle: retain indefinitely while active.
- Missing production bundle: block production startup and regenerate/promote through the pipeline.
- Candidate bundles: retain latest approved candidates and any candidate tied to audit evidence; remove older failed candidates only after reports and manifests are preserved.
- Frozen splits: immutable; create a new split version when membership changes.
- Raw/processed datasets: keep ignored locally; do not delete without confirming the source can be reacquired.
- Caches/logs: safe to delete and regenerate.

## Configuration Usage

- Configuration files inventoried: 17
- Unknown/stale fields require consumer-level validation before removal.

## Test Cleanup

- Active test modules: 156
- Legacy test modules: 65
- Legacy test action: Review legacy tests individually; do not delete failing/stale tests without replacement coverage.

## Hardcoded Rule Inventory

- Files with sample-retail/business-rule terms: 169
- Runtime/gate hits are marked review-required; test/fixture hits are not automatically defects.

## Comment Coverage

- Retained Python modules analyzed: 507
- Modules with purpose docstrings: 267
- Modules missing docstrings: 240
- Public classes documented/missing: 144/416
- Public functions documented/missing: 358/1898
- Full symbol-level gaps are in repository_inventory.json under python_docstring_coverage.

## Before/After Statistics

| Metric | Before | After |
| --- | --- | --- |
| tracked files | 621 | 621 |
| untracked files | 60 | 60 |
| ignored summary entries | 99 | 54 |
| Python files | 552 | 552 |
| Markdown files | 21 | 21 |
| test files | 204 | 204 |
| configuration files | 15 | 15 |
| generated files in Git | 7 | 7 |

## Rollback Instructions

- Generated reports: rerun `python scripts/generate_repository_cleanup_inventory.py` or remove `artifacts/repository_cleanup/`.
- Low-risk cache/log cleanup: rerun tests or commands to regenerate caches/logs if needed.
- Source changes in this pass: revert `.gitignore`, `docs/REPOSITORY_MAP.md`, and `scripts/generate_repository_cleanup_inventory.py` if the cleanup report path should not be tracked.

## Follow-Up Items

- Review untracked `.github/workflows/ci.yml`, semantic_v2 split files, and `tests/test_training_audit_fixes.py` for tracking.
- Decide whether docs/reports run-specific files should move to artifacts/pipeline/runs/<run_id>/reports/.
- Apply retention policy to old candidate bundles only after preserving required manifests and reports.
- Close module/public docstring gaps in focused source-owner batches.
- Review runtime hardcoded sample-retail terms and keep only schema-signature-gated or config-driven behavior.
