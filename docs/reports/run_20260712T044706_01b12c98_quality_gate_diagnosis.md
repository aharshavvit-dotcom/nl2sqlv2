# Run-Scoped Quality Gate Diagnosis

Run ID: `20260712T044706_01b12c98`

Source reports:

- `artifacts/pipeline/runs/20260712T044706_01b12c98/train_model_report.json`
- `artifacts/model_bundle/candidates/20260712T044706_01b12c98/evaluation/model_quality_gate_report.json`
- `artifacts/model_bundle/candidates/20260712T044706_01b12c98/evaluation/controlled_predicted_sql_execution_report.json`

## Verdict

The production candidate failed the final quality gate and is not eligible for promotion.

## Training-Side Evidence

- Checkpoint monitor was `support_weighted_semantic_score` with `save_best_mode=max`.
- Curriculum mode was `ordered_dataset`.
- Hard-negative weight was `0.3`.
- Effective batch size was `8`.
- Validation/train loss ratio was `9.4843`.
- `overfitting_warning` was true.

These are the defects corrected in code by moving checkpoint selection to validation loss, adding selected-checkpoint re-evaluation, disabling ordered curriculum in the canonical diagnostic config, lowering the default hard-negative weight to `0.1`, and adding gradient accumulation.

## Blocking Gate Failures

The final production gate failed these blocking checks:

- `feedback_regression_pass_rate`: `0.6667` expected `>= 0.95`
- `sql_structure_match_rate_min`: `0.1014` expected `>= 0.70`
- `execution_unavailable`: `no_database_connection` expected `execution_available`
- `controlled_predicted_sql_execution_match_rate_min`: `0.5` expected `>= 0.70`
- `controlled_predicted_sql_result_value_match_rate_min`: `0.5` expected `>= 0.70`
- `controlled_predicted_sql_safe_but_wrong_sql_rate_max`: `0.5` expected `<= 0.30`
- `simple_query_pass_rate_production`: `0.8541` expected `>= 0.95`
- `intent_macro_f1_min`: `0.6063` expected `>= 0.80`
- `router_accuracy_min`: `0.8406` expected `>= 0.85`
- `router_macro_f1_min`: `0.4250` expected `>= 0.80`
- `expected_calibration_error_max`: `0.0827` expected `<= 0.08`
- `controlled_predicted_sql_row_count_match_rate_min`: `0.5` expected `>= 0.85`
- `filter_column_accuracy_rate_min`: `0.3031` expected `>= 0.70`
- `filter_value_accuracy_rate_min`: `0.1359` expected `>= 0.70`
- `dimension_column_accuracy_rate_min`: `0.4299` expected `>= 0.75`
- `calibration_metadata_valid`: `false` expected `true`
- `controlled_predicted_sql_passed`: `false` expected `true`
- `projection_exact_match_rate`: `0.2065` expected `>= 0.70`
- `filter_column_accuracy_rate`: `0.3031` expected `>= 0.70`
- `filter_value_accuracy_rate`: `0.1359` expected `>= 0.70`
- `dimension_column_accuracy_rate`: `0.4299` expected `>= 0.65`
- `controlled_predicted_sql_safe_but_wrong_sql_rate`: `0.5` expected `<= 0.30`

## Controlled Predicted-SQL Diagnosis

The controlled predicted-SQL report measured real model predictions from the candidate bundle:

- Cases total: `12`
- Predictions generated: `6`
- Abstentions: `6`
- Execution match rate: `0.5`
- Safe but wrong SQL rate: `0.5`
- Projection exact-match rate: `0.3333`
- Failure areas included `prediction_pipeline`, `filter_grounding`, and `default_projection`.

## Correct Resume Command

To resume or inspect this exact run, use the run ID explicitly:

```powershell
python training/train_model.py `
  --config configs/training.yaml `
  --start-at run_quality_gate `
  --resume `
  --resume-run-id 20260712T044706_01b12c98
```
