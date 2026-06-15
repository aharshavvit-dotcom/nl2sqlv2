# Migration: Naming Cleanup

This document describes the naming changes applied across the repository to remove demo/sample naming ("Option A", "Option C", "V1", "V2") and replace with production-ready names.

## Name Mapping

| Old Name | New Name | Location |
|----------|----------|----------|
| Option C | Retrieval QueryIR Model | UI, README, code |
| Option A | Neural QueryIR Model | UI, README, code |
| Hybrid | Adaptive QueryIR Router | UI, README, code |
| `OptionCToIRConverter` | `RetrievalIRConverter` | `ir/option_c_to_ir.py` |
| `OptionAIRPredictor` | `NeuralIRPredictor` | `neural_ir/predictor.py` |
| `OptionAToIRConverter` | `NeuralIRToIRConverter` | `neural_ir/option_a_to_ir.py` |
| `OptionAIRRepairer` | `NeuralIRRepairer` | `neural_ir/ir_repair.py` |
| `OptionAConfidenceCalibrator` | `NeuralIRConfidenceCalibrator` | `neural_ir/confidence_calibrator.py` |
| `HybridRouterCalibrator` | `AdaptiveRouterCalibrator` | `neural_ir/calibration.py` |
| `option_c_result` | `retrieval_ir_result` | `PredictionResult` field |
| `option_a_result` | `neural_ir_result` | `PredictionResult` field |
| `option_a_version` | `neural_ir_version` | `PredictionResult` field |
| `source_model="option_c"` | `source_model="retrieval_ir"` | `PredictionResult.source_model` |
| `source_model="option_a"` | `source_model="neural_ir"` | `PredictionResult.source_model` |
| `source_model="hybrid"` | `source_model="adaptive_router"` | `PredictionResult.source_model` |

## Artifact Folder Mapping

| Old Folder | New Folder |
|------------|------------|
| `artifacts/option_c_model/` | `artifacts/retrieval_ir_model/` |
| `artifacts/option_a_ir_model/` | `artifacts/neural_ir_model/` |
| `artifacts/option_a_ir_model_v2/` | `artifacts/neural_ir_model/` |

Run `python scripts/migrate_artifact_names.py` to copy old artifacts to new names.

## Evaluation Data File Mapping

| Old File | New File |
|----------|----------|
| `evaluation/option_a_eval_cases.jsonl` | `evaluation/neural_ir_eval_cases.jsonl` |
| `evaluation/option_a_v2_eval_cases.jsonl` | `evaluation/neural_ir_v2_eval_cases.jsonl` |
| `evaluation/hybrid_benchmark_cases.jsonl` | `evaluation/adaptive_router_benchmark_cases.jsonl` |

## Deprecated Scripts

These scripts still work but print deprecation warnings. Use the new wrappers instead:

| Old Script | New Wrapper |
|------------|-------------|
| `training_ir/train_option_a_model.py` | `training/train_neural_ir_model.py` |
| `training_ir/train_option_a_v2_model.py` | `training/train_neural_ir_model.py` |
| `training_ir/evaluate_option_a_model.py` | `evaluation/run_model_evaluation.py` |
| `training_ir/evaluate_option_a_v2_model.py` | `evaluation/run_model_evaluation.py` |
| `training_ir/benchmark_hybrid_system.py` | `evaluation/run_adaptive_router_benchmark.py` |
| `training_ir/calibrate_hybrid_router.py` | *(no wrapper yet — use directly)* |

## Backward Compatibility

All old class names are available as deprecated aliases:

```python
# These all work:
from ir.option_c_to_ir import OptionCToIRConverter  # alias for RetrievalIRConverter
from neural_ir.predictor import OptionAIRPredictor   # alias for NeuralIRPredictor
from neural_ir.calibration import HybridRouterCalibrator  # alias for AdaptiveRouterCalibrator
```

The `PredictionResult.source_model` field accepts both old values (`"option_c"`, `"option_a"`, `"hybrid"`) and new values (`"retrieval_ir"`, `"neural_ir"`, `"adaptive_router"`).

## Test Consolidation

67 test files were consolidated into 9 focused test files:

| File | Coverage Area |
|------|---------------|
| `test_01_core_ir.py` | QueryIR models, IR validation, rendering, SQL-to-IR |
| `test_02_sql_validation.py` | SQL validator, safe preview, dialect handling |
| `test_03_database_connectors.py` | Connection config, connectors, schema reader |
| `test_04_retrieval_runtime.py` | Retrieval model, orchestrator, confidence |
| `test_05_neural_runtime.py` | Neural tokenizer, vocab, linker, predictor aliases |
| `test_06_adaptive_router.py` | Router decisions, calibration, confidence caps |
| `test_07_training_data_pipeline.py` | Dataset adapters, corpus builder, features |
| `test_08_streamlit_app_helpers.py` | Config forms, masking, UI naming compliance |
| `test_09_end_to_end_smoke.py` | Full pipeline, legacy checks, e2e execution |

Original tests are preserved in `tests/legacy/` for reference.

## PostgreSQL Support

The `db/` package provides:
- `DatabaseConnectionConfig` — unified config for SQLite and PostgreSQL
- `SQLiteConnector` — SQLite schema reading and query execution
- `PostgresConnector` — PostgreSQL via `information_schema` queries
- `read_database_schema()` — unified entry point
- `schema_dict_to_graph()` — conversion to `SchemaGraph` for pipeline compatibility
- Password masking via `safe_config_summary()`
