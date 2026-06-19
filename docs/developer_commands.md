# Developer Commands

These are advanced and internal commands for developers working on the NL-to-SQL system.
For the primary user workflow, see the main [README.md](../README.md).

---

## Corpus Construction

### Build QueryIR Training Corpus
```bash
python training/build_generic_ir_corpus.py \
  --datasets wikisql,spider,bird-mini \
  --max-examples 5000 \
  --output-dir data/processed \
  --artifact-dir artifacts/generic_training
```

### Build Hard-Negative Corpus
```bash
python training/build_hard_negative_corpus.py \
  --input data/processed/generic_ir_train.jsonl \
  --output data/processed/generic_ir_hard_negatives.jsonl \
  --max-negatives-per-example 5
```

---

## Individual Model Training

### Train Retrieval QueryIR Model
```bash
python training/train_retrieval_ir_model.py \
  --datasets wikisql spider bird-mini \
  --max-examples 0
```

### Build Retrieval RAG Index
```bash
python training/build_retrieval_rag_index.py \
  --input data/processed/generic_ir_train.jsonl \
  --output-dir artifacts/retrieval_ir_model
```

### Train Neural QueryIR Model (Standard)
```bash
python training/train_neural_ir_model.py \
  --train data/processed/generic_ir_train.jsonl \
  --validation data/processed/generic_ir_validation.jsonl \
  --hard-negatives data/processed/generic_ir_hard_negatives.jsonl \
  --output-dir artifacts/neural_ir_model \
  --epochs 5 \
  --batch-size 8
```

### Train Neural QueryIR Model (Optimized)
```bash
python training/train_neural_ir_optimized.py \
  --config configs/neural_training_default.yaml \
  --train data/processed/generic_ir_train.jsonl \
  --validation data/processed/generic_ir_validation.jsonl \
  --output-dir artifacts/neural_ir_model
```

### Hyperparameter Grid Search
```bash
python training/run_neural_training_experiments.py \
  --grid configs/neural_experiment_grid.yaml \
  --output-dir artifacts/neural_experiments \
  --max-examples 1000 --epochs 3
```

---

## Self-Improvement Loop

### Automated Self-Improvement
```bash
python training/run_self_improvement_loop.py \
  --train data/processed/generic_ir_train.jsonl \
  --validation data/processed/generic_ir_validation.jsonl \
  --retrieval-model-dir artifacts/retrieval_ir_model \
  --neural-model-dir artifacts/neural_ir_model \
  --output-dir artifacts/self_training \
  --iterations 2 --max-examples 1000
```

### Manual Self-Improvement Steps
```bash
# Evaluate against gold
python training/evaluate_against_gold.py \
  --input data/processed/generic_ir_validation.jsonl \
  --retrieval-model-dir artifacts/retrieval_ir_model \
  --neural-model-dir artifacts/neural_ir_model \
  --output artifacts/self_training/validation_predictions.jsonl \
  --report artifacts/self_training/validation_gold_comparison_report.json

# Mine validation errors
python training/mine_validation_errors.py \
  --predictions artifacts/self_training/predictions.jsonl \
  --output-dir data/processed/self_training

# Build corrections from gold
python training/build_corrections_from_gold.py \
  --predictions artifacts/self_training/predictions.jsonl \
  --output-dir data/processed/self_training

# Train ranking from gold
python training/train_ranking_from_gold.py \
  --predictions artifacts/self_training/predictions.jsonl \
  --output-dir artifacts/adaptive_ranker
```

---

## Evaluation

### Evaluate Models on Split Datasets
```bash
python training/evaluate_generic_models.py \
  --test data/processed/generic_ir_test.jsonl \
  --unseen-db-test data/processed/generic_ir_unseen_db_test.jsonl \
  --retrieval-model-dir artifacts/retrieval_ir_model \
  --neural-model-dir artifacts/neural_ir_model \
  --output artifacts/evaluation/generic_model_evaluation_report.json
```

This command also writes `classification_metrics_report.{json,md}`, calibration reports, and intent/base-table/join/router/error confusion matrices under `artifacts/evaluation/`. Full quality gates require these reports and use macro F1 for imbalanced decision classes.

### Execution-Aware Evaluation
```bash
python training/run_execution_aware_evaluation.py \
  --predictions artifacts/self_training/validation_predictions.jsonl \
  --output artifacts/evaluation/execution_aware_evaluation_report.json
```

### Unseen-DB Benchmark
```bash
python training/run_unseen_db_benchmark.py \
  --input data/processed/generic_ir_unseen_db_test.jsonl \
  --retrieval-model-dir artifacts/retrieval_ir_model \
  --neural-model-dir artifacts/neural_ir_model \
  --output artifacts/evaluation/unseen_db_benchmark_report.json
```

---

## Model Selection & Promotion

### Quality Gate
```bash
python training/run_model_quality_gate.py \
  --evaluation-report artifacts/evaluation/generic_model_evaluation_report.json \
  --thresholds evaluation/model_quality_thresholds.yaml \
  --output artifacts/evaluation/model_quality_gate_report.json
```

### Select Best Model
```bash
python training/select_best_model.py \
  --evaluation-report artifacts/evaluation/generic_model_evaluation_report.json \
  --execution-report artifacts/evaluation/execution_aware_evaluation_report.json \
  --thresholds evaluation/model_quality_thresholds.yaml \
  --output artifacts/evaluation/model_selection_report.json
```

### Promote Model
```bash
python training/promote_model_if_better.py \
  --candidate-dir artifacts/neural_ir_model \
  --model-name neural_ir_model \
  --evaluation-report artifacts/evaluation/generic_model_evaluation_report.json \
  --execution-report artifacts/evaluation/execution_aware_evaluation_report.json \
  --thresholds evaluation/model_quality_thresholds.yaml \
  --output artifacts/model_registry/promotion_report.json
```

When both candidates contain paired per-example results, promotion performs 1,000 deterministic bootstrap resamples and writes `artifacts/evaluation/champion_challenger_statistical_report.{json,md}`. Point estimates are retained only as a compatibility fallback.

---

## Auditing & Verification

### Audit Scripts
```bash
python scripts/audit_generic_nl2sql_readiness.py
python scripts/audit_neural_training_readiness.py
python scripts/audit_self_training_readiness.py
python scripts/audit_execution_pipeline_readiness.py
python scripts/audit_integration_readiness.py
```

### Release Readiness
```bash
python training/run_release_readiness_check.py \
  --audit-report artifacts/audit/generic_nl2sql_readiness_report.json \
  --evaluation-report artifacts/evaluation/generic_model_evaluation_report.json \
  --quality-gate-report artifacts/evaluation/model_quality_gate_report.json \
  --regression-report artifacts/evaluation/regression_suite_report.json \
  --output artifacts/evaluation/release_readiness_report.json
```

---

## Pipeline Orchestration (Internal)

### Old Pipeline Runner (preserved for backward compatibility)
```bash
python training/run_full_training_pipeline.py \
  --config pipeline_configs/smoke_training.yaml

python training/run_full_training_pipeline.py \
  --config pipeline_configs/full_generic_training.yaml
```

> **Note:** The canonical command is now `python training/train_model.py --config configs/training.yaml`.

---

## Feedback (Optional)

### Build Feedback Training Data
```bash
python training/build_feedback_training_data.py \
  --feedback data/feedback/query_feedback.jsonl \
  --output-dir data/processed \
  --artifact-dir artifacts/feedback
```

### Rebuild Feedback Index
```bash
python training/rebuild_feedback_index.py \
  --feedback-examples data/processed/feedback_positive_examples.jsonl \
  --output-dir artifacts/retrieval_ir_model
```

---

## Connected Database Testing

### Generate Regression Cases
```bash
python training/generate_connected_db_regressions.py \
  --schema artifacts/schema/current_schema.json \
  --output artifacts/connected_db_regressions/generated_cases.jsonl
```

### Run Regressions
```bash
python training/run_connected_db_regressions.py \
  --cases artifacts/connected_db_regressions/generated_cases.jsonl \
  --output artifacts/connected_db_regressions/regression_report.json
```

---

## Calibration

### Confidence Calibration
```bash
python training_ir/calibrate_option_a_confidence.py
python training_ir/calibrate_hybrid_router.py
```
