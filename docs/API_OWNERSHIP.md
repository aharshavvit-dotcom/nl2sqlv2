# API Ownership

> Generated: 2026-07-19  
> One canonical owner per responsibility. No duplicate implementations.

## Active API Surface

| API | Canonical Module | Status |
|-----|-----------------|--------|
| `QueryIR` (v1, runtime) | `ir/query_ir_models.py` | Active — **target: compat only** |
| `QueryNode` (v2, advanced) | `ir/query_ir_v2_models.py` | Active — **target: becomes canonical QueryIR** |
| `IRToSQLRenderer` | `ir/ir_to_sql_renderer.py` | Active canonical |
| `IRValidator` | `ir/ir_validator.py` | Active canonical (v1) |
| `QueryIRV2Validator` | `ir/query_ir_v2_validation.py` | Active (v2) |
| `SQLValidator` | `validation/sql_validator.py` | Active canonical |
| `execute_select()` | `execution/query_executor.py` | Active canonical |
| `PredictionOrchestrator` | `inference/prediction_orchestrator.py` | Active canonical |
| `NeuralIRPredictor` | `neural_ir/predictor.py` | Active canonical |
| `SchemaAwareOptionAIRModel` | `neural_ir/attention_model.py` | Active canonical |
| `IRTrainingDataset` | `neural_ir/ir_dataset.py` | Active canonical |
| `IRLabelEncoder` | `neural_ir/ir_label_encoder.py` | Active canonical |
| `ModelBundleLoader` | `model_bundle/bundle_loader.py` | Active canonical |
| `ModelBundleValidator` | `model_bundle/bundle_validator.py` | Active canonical |
| `BundlePromoter` | `model_bundle/bundle_promoter.py` | Active canonical |
| `DatasetSplitManager` | `dataset_training/split_manager.py` | Active canonical |
| `DatasetLeakageChecker` | `dataset_training/leakage_checker.py` | Active canonical |
| `GenericIRCorpusBuilder` | `dataset_training/ir_corpus_builder.py` | Active canonical |
| `SQLToIRConverter` | `ir/sql_to_ir_converter.py` | Active canonical |
| `SQLToQueryIRV2Converter` | `ir/sql_to_query_ir_v2.py` | Active canonical |
| `RetrievalNL2SQLModel` | `retriever/retrieval_nl2sql_model.py` | Active canonical |
| `TfidfRetriever` | `nl2sql_v1/retriever.py` | Active — **migration target → retrieval/** |
| `SchemaGraph` | `nl2sql_v1/schema.py` | Active — **migration target → db/** |

## Migration Targets

| Current Location | Target Location | Deadline | Gate |
|-----------------|----------------|----------|------|
| `nl2sql_v1/schema.py` → `SchemaGraph`, `ColumnInfo`, `TableInfo`, `ForeignKeyInfo` | `db/schema_graph.py` | 2026-09-01 | `tests/test_stale_api_absence.py` |
| `nl2sql_v1/retriever.py` → `TfidfRetriever`, `RetrievalResult` | `retrieval/tfidf_retriever.py` | 2026-09-01 | `tests/test_stale_api_absence.py` |
| `ir/query_ir_models.py` → `QueryIR` (v1) | `compat/legacy_query_ir_loader.py` | 2026-09-01 | Runtime uses only canonical QueryIR |

## Stale/Deprecated APIs

| API | Location | Callers | Classification |
|-----|----------|---------|---------------|
| `check_leakage(train_path, val_path)` | Not defined | `train_neural_ir_optimized.py:308` | **STALE CALL** |
| `option_a_model_dir` | `PredictionOrchestrator` | Backward compat alias | DEPRECATED |
| `use_option_a_fallback` | `PredictionOrchestrator` | Backward compat alias | DEPRECATED |
| `option_a_threshold` | `PredictionOrchestrator` | Backward compat alias | DEPRECATED |
| `OptionCToIRConverter` | `ir/__init__.py` | Deprecated alias for `RetrievalIRConverter` | DEPRECATED |
| `_load_model_legacy()` | `app/streamlit_app.py` | Dev mode only | DEPRECATED |
| `schema_aware_queryir_v1` | Model version string | Configs + code | **RENAME** |
