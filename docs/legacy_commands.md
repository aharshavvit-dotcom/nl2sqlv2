# Legacy Commands

These commands are from older versions of the system and have been superseded
by the integrated training pipeline. They are preserved for backward compatibility
but should not be used in the primary workflow.

> **Preferred command:** `python training/train_model.py --config configs/training.yaml`

Both legacy full-pipeline wrappers and the preferred command now resolve their effective steps through `orchestration.pipeline_config.build_pipeline_steps`; no separate default registry is maintained.

Legacy gold-replay and comparison commands are debugging tools only. Any report that replays `query_ir` labels as predictions must be labeled as `explicit_gold_replay_baseline` and is not valid for quality gates, model promotion, or bundle validation.

Legacy commands do not attach `controlled_predicted_sql_execution_report.json` to candidate bundles, enforce strong candidate-manifest identity, emit normalized SQL `policy_failure_type` counts, collect observed candidate-relation mask statistics, or provide the current multi-seed stability interpretation fields. They may also bypass the integrated requirement that statistical predicted-SQL bootstrap comparisons use at least 10 common stable case IDs. Use the integrated pipeline for production lifecycle proof.

---

## Deprecated Training Scripts

### Legacy Self-Training Loop
```bash
# DEPRECATED — use training/train_model.py instead
python training/run_self_training_loop.py \
  --train data/processed/generic_ir_train.jsonl \
  --validation data/processed/generic_ir_validation.jsonl \
  --test data/processed/generic_ir_test.jsonl \
  --output-dir artifacts/self_training \
  --max-iterations 3 \
  --epochs-per-iteration 10 \
  --batch-size 32
```

### Legacy Batch Predictions
```bash
# DEPRECATED — use training/train_model.py for automated pipeline
python training/run_batch_predictions.py \
  --model-dir artifacts/neural_ir_model \
  --input data/processed/generic_ir_validation.jsonl \
  --output artifacts/self_training/predictions.jsonl
```

### Legacy Gold Comparison
```bash
# DEPRECATED — integrated into training/train_model.py pipeline
python training/run_gold_comparison.py \
  --predictions artifacts/self_training/predictions.jsonl \
  --gold data/processed/generic_ir_validation.jsonl \
  --output artifacts/self_training/comparison_report.json
```

---

## Deprecated IR Training Data Commands

### Legacy IR Training Data Builder
```bash
# DEPRECATED — use training/build_generic_ir_corpus.py instead
python training_ir/build_ir_training_data.py \
  --datasets wikisql,spider,bird-mini \
  --output-dir training_data
```

### Legacy IR Corpus Validation
```bash
# DEPRECATED
python training_ir/validate_ir_corpus.py \
  --input training_data/ir_training_examples.jsonl
```

---

## Deprecated Calibration Commands

### Legacy Confidence Calibration
```bash
# DEPRECATED — calibration is now integrated into the training pipeline
python training_ir/calibrate_option_a_confidence.py
```

### Legacy Hybrid Router Calibration
```bash
# DEPRECATED — router calibration is now integrated
python training_ir/calibrate_hybrid_router.py
```

---

## Old Artifact Naming

If you have artifacts using old naming conventions (`option_a_ir_model`, `option_c_model`, etc.),
run the migration script:

```bash
python scripts/migrate_artifact_names.py
```

See [migration_naming_cleanup.md](migration_naming_cleanup.md) for details.
