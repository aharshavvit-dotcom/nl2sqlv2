# Canonical File Ownership Matrix

> Generated for the NL-to-SQL Maturity Program. Rule: extend canonical owners, create new modules only for genuinely new responsibilities.

## Core IR

| Responsibility | Canonical Owner | Entry Point | Action |
|---|---|---|---|
| QueryIR v1 models | `ir/query_ir_models.py` | `QueryIR`, `diff_query_ir` | Preserve |
| QueryIR v2 models | `ir/query_ir_v2_models.py` | `QueryNode`, `Expression`, `Predicate` | **Extend** |
| V2 boolean renderer | `ir/query_ir_v2_boolean_renderer.py` | `QueryIRV2NativeRenderer` | **Extend** (decompose internally) |
| V2 validation | `ir/query_ir_v2_validation.py` | `QueryIRV2Validator` | **Extend** |
| V1→V2 migration | `ir/query_ir_migration.py` | `migrate_v1_to_v2`, `convert_v2_to_v1` | **Extend** |
| SQL→V2 conversion | `ir/sql_to_query_ir_v2.py` | `SQLToQueryIRV2Converter` | **Extend** |
| SQL→V1 conversion | `ir/sql_to_ir_converter.py` | `SQLToIRConverter` | Preserve |
| V1 SQL renderer | `ir/ir_to_sql_renderer.py` | `IRToSQLRenderer` | Preserve |
| V2 serialization | `ir/query_ir_v2_serialization.py` | `dumps_query_ir_v2`, `loads_query_ir_v2` | Preserve |
| V2 canonicalization | `ir/query_ir_v2_boolean_canonicalization.py` | `canonicalize_predicate` | Preserve |
| Version detection | `ir/query_ir_version_loader.py` | `detect_query_ir_version` | Preserve |

## Neural/Model

| Responsibility | Canonical Owner | Entry Point | Action |
|---|---|---|---|
| Schema-aware model | `neural_ir/attention_model.py` | `SchemaAwareOptionAIRModel` | **Extend** |
| Legacy model | `neural_ir/model.py` | `OptionAIRModel` | Preserve |
| Training loop | `neural_ir/trainer.py` | `OptionAIRTrainer` | **Extend** |
| Dataset/dataloader | `neural_ir/ir_dataset.py` | `IRTrainingDataset` | **Extend** |
| Label encoder | `neural_ir/ir_label_encoder.py` | `IRLabelEncoder` | **Extend** |
| Schema linker | `neural_ir/schema_linker.py` | `SchemaLinker` | Preserve |
| Candidate builder | `neural_ir/candidate_builder.py` | `SchemaCandidateBuilder` | Preserve |
| Hard negatives | `neural_ir/hard_negative_builder.py` | — | **Extend** |
| Pointer network | `neural_ir/pointer_network.py` | `SchemaPointerNetwork` | Preserve |
| Curriculum | `neural_ir/training_curriculum.py` | — | **Extend** |
| IR repair | `neural_ir/ir_repair.py` | — | Modify (remove hardcoding) |

## Training Infrastructure

| Responsibility | Canonical Owner | Entry Point | Action |
|---|---|---|---|
| Training entry point | `training/train_neural_ir_optimized.py` | CLI | **Extend** |
| Training config | `neural_optimization/training_config.py` | `TrainingConfig` | **Extend** |
| Checkpoint manager | `neural_optimization/checkpoint_manager.py` | `CheckpointManager` | **Extend** |
| Early stopping | `neural_optimization/early_stopping.py` | — | Preserve |
| Optimizer factory | `neural_optimization/optimizer_factory.py` | `build_optimizer` | Preserve |
| Loss weighting | `neural_optimization/loss_weighter.py` | — | **Extend** |
| FFN blocks | `neural_optimization/ffn_blocks.py` | `FeedForwardBlock` | Preserve |

## Dataset/Corpus

| Responsibility | Canonical Owner | Entry Point | Action |
|---|---|---|---|
| IR corpus building | `dataset_training/ir_corpus_builder.py` | — | **Extend** |
| Dataset splits | `dataset_training/split_manager.py` | — | **Extend** |
| Leakage checking | `dataset_training/leakage_checker.py` | — | **Extend** |
| Curriculum building | `dataset_training/curriculum_builder.py` | — | **Extend** |
| Hard-neg corpus | `dataset_training/hard_negative_corpus_builder.py` | — | **Extend** |
| Corpus quality | `dataset_training/corpus_quality.py` | — | Preserve |

## Capabilities

| Responsibility | Canonical Owner | Entry Point | Action |
|---|---|---|---|
| Taxonomy | `capabilities/taxonomy.py` | `Capability`, `SafetyLabel` | Preserve |
| SQL extractor | `capabilities/sql_capability_extractor.py` | `SQLCapabilityExtractor` | **Extend** |
| Contracts | `capabilities/contracts.py` | `CapabilityAnnotation` | **Extend** |

## Inference/Runtime

| Responsibility | Canonical Owner | Entry Point | Action |
|---|---|---|---|
| Orchestrator | `inference/prediction_orchestrator.py` | — | **Modify** |
| Confidence | `inference/prediction_confidence.py` | — | **Extend** |
| Slot resolver | `inference/slot_resolver.py` | — | Modify (remove hardcoding) |
| Schema mapper | `inference/schema_aware_mapper.py` | — | Modify (remove hardcoding) |
| Telemetry | `inference/telemetry_logger.py` | — | **Extend** |

## Validation/Execution

| Responsibility | Canonical Owner | Entry Point | Action |
|---|---|---|---|
| SQL validator | `validation/sql_validator.py` | `SQLValidator` | **Extend** |
| Query executor | `execution/query_executor.py` | `execute_select`, `execute_query` | **Extend** |

## Bundle/Promotion

| Responsibility | Canonical Owner | Entry Point | Action |
|---|---|---|---|
| Bundle builder | `model_bundle/bundle_builder.py` | — | **Extend** |
| Bundle validator | `model_bundle/bundle_validator.py` | — | **Extend** |
| Bundle promoter | `model_bundle/bundle_promoter.py` | — | **Extend** |
| Bundle manifest | `model_bundle/bundle_manifest.py` | — | **Extend** |

## Retrieval

| Responsibility | Canonical Owner | Entry Point | Action |
|---|---|---|---|
| RAG retriever | `retrieval/rag_retriever.py` | — | **Extend** |
| Reranker | `retrieval/retrieval_reranker.py` | — | **Extend** |

## Quality/Orchestration

| Responsibility | Canonical Owner | Entry Point | Action |
|---|---|---|---|
| Quality gates | `quality_gates/model_quality_gate.py` | — | **Extend** |
| Pipeline runner | `orchestration/step_runner.py` | — | **Modify** |

## New Modules (Genuinely New Responsibilities)

| Responsibility | New Module | Justification |
|---|---|---|
| Scope analysis | `ir/query_ir_v2_scope.py` | New capability: derived scope analysis of QueryIR trees |
| Three-valued logic | `ir/query_ir_v2_three_valued_logic.py` | SQL-specific Boolean logic, no existing owner |
| V2 round-trip testing | `ir/query_ir_v2_roundtrip.py` | New validation framework |
| V2 renderer internals | `ir/query_ir_v2_rendering/` (package) | Internal decomposition of existing renderer |
| Schema graph | `neural_ir/schema_graph.py` | New structured schema representation |
| Identifier decomposer | `neural_ir/identifier_decomposer.py` | New subword tokenization |
| Join-path scorer | `neural_ir/join_path_scorer.py` | New neural scoring module |
| Pairwise compatibility | `neural_ir/pairwise_compatibility.py` | New compatibility scoring |
| QueryIR decoder | `neural_ir/queryir_decoder.py` | New grammar-constrained decoder |
| Synthetic generator | `training_data/synthetic_generator.py` | New data generation |
| Schema augmenter | `training_data/schema_renaming_augmenter.py` | New augmentation |
| Coverage report | `training_data/coverage_report.py` | New coverage analysis |
| Calibration v2 | `neural_ir/calibration_v2.py` | New calibration methods |
| Component metrics | `evaluation/component_metrics.py` | New evaluation framework |
| QueryIR metrics | `evaluation/queryir_metrics.py` | New tree metrics |
| SQL metrics | `evaluation/sql_metrics.py` | New SQL-level metrics |
| Generalization evaluator | `evaluation/generalization_evaluator.py` | New generalization framework |
| Statistical reporter | `evaluation/statistical_reporter.py` | New multi-seed reporting |
| Ablation runner | `evaluation/ablation_runner.py` | New ablation management |
| Semantic retriever | `retrieval/semantic_retriever.py` | New bi-encoder retrieval |
| Feedback pipeline | `inference/feedback_pipeline.py` | New feedback flow |

## Hardcoded Rule Inventory

| File | Pattern | Type | Action |
|---|---|---|---|
| `neural_ir/ir_repair.py` L136,145,156 | Revenue expression | Production hardcoding | Regression test → general → remove |
| `neural_ir/option_a_to_ir.py` L119,126 | Revenue mapping | Production hardcoding | Regression test → general → remove |
| `neural_ir/training_curriculum.py` L63 | Revenue detection | Production hardcoding | Regression test → general → remove |
| `neural_ir/ir_label_encoder.py` L281 | Revenue check | Production hardcoding | Regression test → general → remove |
| `neural_ir/hard_negative_builder.py` L179 | Revenue negative | Production hardcoding | Regression test → general → remove |
| `ir/sql_to_ir_converter.py` L65 | `APPROVED_REVENUE_EXPR` | Production hardcoding | Move to config → remove |
| `ir/semantic_metric_resolver.py` L44 | Revenue metric | Semantic config | Move to metadata/config |
| `ir/ir_validator.py` L136-137 | Revenue validation | Production hardcoding | Generalize → remove |
| `scripts/create_sample_db.py` L188 | order_items insert | Test fixture | Keep (benchmark data) |
