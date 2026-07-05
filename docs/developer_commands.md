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
  --model-bundle-dir artifacts/model_bundle/current \
  --retrieval-model-dir artifacts/retrieval_ir_model \
  --neural-model-dir artifacts/neural_ir_model \
  --output artifacts/evaluation/generic_model_evaluation_report.json
```

This command must generate real model predictions by loading a bundle or artifact directories. It also writes `classification_metrics_report.{json,md}`, calibration reports, and intent/base-table/join/router/error confusion matrices under `artifacts/evaluation/`. Full quality gates require these reports and use macro F1 for imbalanced decision classes.

For debugging only, pass `--allow-gold-replay-baseline`. Reports created this way are labeled with `evaluation_mode = explicit_gold_replay_baseline` and `is_valid_for_quality_gate = false`; quality gates and promotion must not consume them as model-performance evidence.

When evaluating a candidate bundle, pass `--model-bundle-dir artifacts/model_bundle/candidate` so runtime reports, controlled predicted-SQL reports, and bundle validation all describe the same artifact.

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
  --model-bundle-dir artifacts/model_bundle/current \
  --retrieval-model-dir artifacts/retrieval_ir_model \
  --neural-model-dir artifacts/neural_ir_model \
  --output artifacts/evaluation/unseen_db_benchmark_report.json
```

The unseen-DB benchmark fails closed when no runnable model artifacts are available. It may use gold replay only with `--allow-gold-replay-baseline`, and that report is explicitly marked invalid for quality gates.

### Controlled Execution-Aware Evaluation
```bash
python training/run_execution_aware_evaluation.py --run-controlled-fixtures
```
Creates a temporary SQLite database from `evaluation/fixtures/controlled_evaluation.sql`, loads cases from `evaluation/fixtures/controlled_evaluation_cases.jsonl`, executes gold SQL, and verifies row counts and SQL safety. Use this to validate execution evaluation infrastructure without requiring full training.

> **Note:** `BenchmarkRunner` has been renamed to `GoldReplayBenchmarkRunner`. It is a debug-only oracle baseline. All output is forced to `gold_replay_used = true` and `is_valid_for_quality_gate = false`. For real model benchmarks, use `run_unseen_db_benchmark.py` or `evaluate_generic_models.py`.

### Lifecycle Proof Fields

The `bundle_manifest.json` `lifecycle_proof` section records:
- `generic_eval_valid_for_quality_gate` — evaluation passed strict validity checks
- `generic_eval_real_predictions_generated` — count of real (non-gold-replay) predictions
- `generic_eval_predictor_used` — a real predictor callable was used
- `calibration_report_available`, `conformal_threshold_available` — calibration artifacts exist
- `calibration_loaded_in_runtime_smoke` — calibration loaded during runtime smoke test
- `controlled_predicted_sql_report_attached_to_bundle` — controlled predicted-SQL report was copied into the candidate bundle
- `controlled_predicted_sql_report_location` — `bundle`, `root_artifacts`, or `missing`
- `central_sql_validator_used` — predicted SQL used the shared validator before fixture execution
- `predicted_safe_sql_rate`, `predicted_execution_success_rate`, `predicted_row_count_match_rate` — controlled predicted-SQL execution summary
- `evaluation_stability_interpretation` — explains whether seed metrics are evaluation-only stability or full training variance
- `report_identity_validated` — Identity verification confirmed `bundle_id` and pipeline IDs match exactly between reports and candidate bundle
- `primary_seed_included` — At least the primary seed completed successfully
- `production_ready` — all required fields are True

Quality gates enforce: `real_predictions_generated > 0`, `predictor_used = true`, `rows_evaluated > 0`. Zero-prediction reports are always rejected.

### Calibration and Conformal Threshold

Calibration artifacts are written to `artifacts/evaluation/calibration_report.json` and copied into the bundle. The `conformal_confidence_threshold` is the learned threshold below which predictions trigger clarification/abstention. The app runtime smoke test verifies this threshold is behaviorally active (not just loaded).

---

## Model Selection & Promotion

### Quality Gate
```bash
python training/run_model_quality_gate.py \
  --evaluation-report artifacts/evaluation/generic_model_evaluation_report.json \
  --thresholds evaluation/model_quality_thresholds.yaml \
  --output artifacts/evaluation/model_quality_gate_report.json
```

Production mode enforces the production simple-query threshold and requires an explicit `simple_query_pass_rate`; it does not substitute `intent_accuracy_rate`. Smoke and developer modes may use the lower smoke threshold for fast feedback.

Quality gate modes are explicit:

| Mode | Candidate bundle | Missing execution/feedback | Promotion |
|:---|:---|:---|:---|
| `debug` | Built when artifacts are complete | Warning | Never |
| `baseline` | Built and validated for diagnostics | Warning when unavailable | Never |
| `production` | Built only after core gate; final gate includes controlled predictions | Blocking when configured as required | Only after all checks |
| `release` | Same as production plus release evidence | Blocking | Champion/challenger policy also applies |

Run `configs/debug_training.yaml` for app testing, `configs/baseline_training.yaml` for actionable model diagnostics, and `configs/training.yaml` for strict production promotion. A successful neural training step does not imply that a current bundle exists.

The generic evaluation writes `unsafe_sql_examples.jsonl`, detailed `sql_validation_failures.jsonl` rows with deterministic root-cause hints, `sql_validation_failure_breakdown`, before/after repair metrics, filter/dimension linking diagnostics, and abstention metrics. `execution_match_rate` is `null` when execution did not run; baseline/debug gates warn, while production fails with `execution_unavailable` only when execution is configured as required.

### Select Best Model
```bash
python training/select_best_model.py \
  --evaluation-report artifacts/evaluation/generic_model_evaluation_report.json \
  --execution-report artifacts/evaluation/execution_aware_evaluation_report.json \
  --controlled-predicted-sql-report artifacts/evaluation/controlled_predicted_sql_execution_report.json \
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

When both candidates contain paired per-example results, promotion performs 1,000 deterministic bootstrap resamples and writes `artifacts/evaluation/champion_challenger_statistical_report.{json,md}`. Bootstrap coverage is checked per metric. Metrics without bootstrap evidence fall back to point-estimate regression checks instead of being globally skipped.

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

---

## Multi-Seed Evaluation

Enable evaluation-only stability analysis in `configs/training.yaml`:

```yaml
seeds:
  enabled: true
  values: [42, 123, 456]
  metrics: [intent_macro_f1, base_table_accuracy, sql_validation_rate]
```

This re-runs the evaluation step per seed and computes metric variance. It measures **prediction stability**, not training variance. The report field `is_valid_for_training_variance_governance` will be `false`. Full per-seed re-training is a future enhancement.

The variance report records `seed_runs`, model source, `stochastic_inference_enabled`, `stochastic_components`, and `evaluation_stability_interpretation`. Model selection treats these warnings as evaluation-stability signals unless the report explicitly comes from full per-seed re-training.

---

## Controlled Fixture Evaluation

### Gold SQL Fixture Validation (default)
Validates gold SQL executes correctly on a deterministic in-memory SQLite database. Configured via `execution_aware.controlled_fixtures.enabled: true`.

### Predicted SQL Execution (experimental)
Loads the candidate model bundle and runs predictions against fixture questions. Configured via `execution_aware.controlled_predicted_sql.enabled: true`.

Predicted SQL is passed through the central SQL validator before execution. Per-case validator failures use stable `policy_failure_type` values, and reports include `policy_failure_type_counts`. The integrated production pipeline writes `controlled_predicted_sql_execution_report.json`, copies it into the candidate bundle `evaluation/` directory, and can require that attached report before bundle validation.

Required mode fails when the candidate manifest is missing, unreadable, or has no `bundle_id`. Optional mode can fall back to `pipeline_name`, but marks `bundle_id_source: pipeline_name_fallback`, `identity_strength: weak`, and emits `candidate_manifest_missing_for_predicted_sql`. Bootstrap comparison becomes statistical at 10 common stable case IDs and otherwise reports `insufficient_common_cases`.

To inspect a failed candidate in Streamlit without weakening production loading:

```powershell
$env:NL2SQL_ALLOW_CANDIDATE_BUNDLE = "1"
streamlit run app/streamlit_app.py
```

Candidate debug loading is opt-in, visibly labeled, and exposes `bundle_source: candidate_debug`, `quality_gate_passed: false`, `production_ready: false`, and `loaded_for_debug: true`.

### Semantic Correctness Reports

Safety and correctness are separate. `sql_validation_rate` measures whether SQL is safe and valid; execution success measures whether it runs; execution/value match measures whether it answers the question. A high `safe_but_wrong_sql_rate` therefore remains a release blocker.

- `model_quality_gate_report.json`: blocking thresholds and promotion eligibility.
- `controlled_predicted_sql_execution_report.json`: per-case results, QueryIR diffs, semantic failure categories, and safe-but-wrong counts.
- `classification_metrics_report.json`: intent, table, linking, and projection metrics.
- `calibration_report.json`: ECE, confidence diversity, and `calibration_degenerate`.
- `model_selection_report.json`: fresh eligible candidates plus rejected stale/gold/oracle reports.

Common failures: low `filter_value_accuracy_rate` or `dimension_column_accuracy_rate` indicates grounding errors; `execution_unavailable` means no result comparison was possible; `calibration_degenerate` disables confidence-driven abstention; `model_selection_stale` blocks release because evidence is not tied to the current bundle.

---

## Relation-Aware Schema Attention

Enable in `configs/neural_training_default.yaml`:

```yaml
model:
  relation_aware_attention:
    enabled: true
    bias_init: 0.0
```

Candidate-pairwise relation attention uses the unified table/column candidate mask, excluding padding before softmax. Runtime `relation_bias_mode` is one of `disabled`, `schema_token_role_bias`, `schema_pairwise_relation_bias`, `schema_candidate_pairwise_relation_bias`, or `combined`.

`training_diagnostics.json` distinguishes relation IDs that are configured, observed in dataset items, observed in collated batches, and actually used in `model.forward`. Its candidate graph block aggregates real batch-mask counts and padding ratios. Multi-seed regression tests invoke `_run_multi_seed_variance()` directly; the report still represents evaluation-only stability, not full retraining variance.

This adds a lightweight RAT-SQL-style learnable bias per relation type to schema attention. When enabled, the dataset emits explicit schema relation matrices for table, column, primary-key, and foreign-key relationships instead of relying only on question-schema role tags. **Not production behavior** unless explicitly enabled and validated via controlled experiments.
