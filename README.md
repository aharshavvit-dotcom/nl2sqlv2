# QueryIR NL-to-SQL

A generic, connected-database natural language to SQL system using **QueryIR** (Query Intermediate Representation). The system converts natural language questions into structured SQL by first producing a semantic IR, then rendering dialect-specific SQL.

---

## Architecture

```
Question ──> Schema Mapping ──> Retrieval / Neural / Direct Planning
     └──> QueryIR Construction ──> IR Validation ──> SQL Rendering
     └──> SQL Validation ──> Execution
```

| Model / Component | Description |
|:---|:---|
| **Retrieval QueryIR Model** | TF-IDF retrieval + template matching pipeline. |
| **Neural QueryIR Model** | Pointer-network model trained on IR labels. |
| **Adaptive QueryIR Router** | Confidence-based router selecting the best model. |
| **Generic Direct Planner** | Schema-safe deterministic planner for simple queries. |
| **Model Bundle** | Validated artifact package produced by the training pipeline. |

> [!NOTE]
> Older documentation may reference "Option A", "Option C", "V1", "V2", or "Hybrid". These have been renamed. See [docs/migration_naming_cleanup.md](docs/migration_naming_cleanup.md).

---

## Quick Start

### 1. Install Dependencies

```bash
git clone <repo-url> nl2sqlv2
cd nl2sqlv2
python -m venv venv
source venv/bin/activate    # Linux/macOS
# .\\venv\\Scripts\\Activate.ps1  # Windows
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
python scripts/download_datasets.py --datasets wikisql spider bird-mini
```

### 3. Train the Model (One Command)

```bash
python training/train_model.py --config configs/training.yaml
```

This single command internally runs:
- Corpus building
- Dataset contribution and unsupported SQL reporting
- Retrieval RAG index building
- Hard-negative corpus generation
- Neural QueryIR training
- Gold validation & error mining
- Adaptive ranker training
- Evaluation & quality gates
- Model bundle creation & promotion

Training completion and production availability are separate outcomes. A trained model is promoted to `artifacts/model_bundle/current/` only after the production quality gate, controlled predicted-SQL proof, and bundle validation pass.

Use the lifecycle configs deliberately:

```bash
python training/train_model.py --config configs/debug_training.yaml
python training/train_model.py --config configs/baseline_training.yaml
python training/train_model.py --config configs/training.yaml
```

- `debug`: small, fast, builds a non-promoted candidate; missing execution/feedback evidence is advisory.
- `baseline`: runs full diagnostics and optional controlled predictions without treating unavailable execution as a failed execution.
- `production`: requires configured SQL, feedback, controlled prediction, and execution evidence before promotion.
- `release`: production semantics plus release/regression and champion/challenger evidence when invoked by release tooling.

Full training builds a dataset-balanced generic QueryIR corpus. WikiSQL, Spider, and BIRD Mini each receive their own sampling cap, and the run fails if any requested full-training dataset does not contribute the configured minimum number of converted QueryIR examples. Unsupported SQL is written to `artifacts/generic_training/unsupported_sql_report.json` so current QueryIR coverage gaps are visible.

The connected-database runtime is schema-neutral. Bundled `orders` / `customers` / `products` mappings are enabled only when the complete sample-retail schema signature is present. Other databases derive table, metric, dimension, and filter vocabulary from their own schema; simple single-table questions bypass retrieval and neural routing and cannot add joins.

Evaluation is multi-level: intent, base table, slots, join decisions, router decisions, QueryIR validity, SQL validation, structural/execution match, and safety. Reports include accuracy, macro/micro/weighted F1, confusion matrices, p50/p95/p99 loss/confidence/latency/drift statistics, ECE/Brier calibration, and a conformal abstention threshold. Evaluation reports are valid for quality gates only when `evaluation_mode = real_model_predictions`, `gold_replay_used = false`, and `is_valid_for_quality_gate = true`. Gold replay is a debug baseline only and cannot be promoted.

These statuses are deliberately different:

- **SQL valid/safe**: the central validator allows the read-only statement.
- **SQL executable**: the database runs it without an error.
- **SQL semantically correct**: its rows and values match the expected answer.
- **SQL production-ready**: semantic, linking, calibration, freshness, safety, and release gates all pass.

An executable query with incorrect rows is reported as safe-but-wrong and cannot pass production. Inspect `model_quality_gate_report.json`, `controlled_predicted_sql_execution_report.json`, `classification_metrics_report.json`, `calibration_report.json`, and `model_selection_report.json` together. `calibration_degenerate` disables confidence-threshold abstention; `model_selection_stale` means the report predates or does not identify the candidate bundle.

For a quick smoke test:
```bash
python training/train_model.py --config configs/smoke_training.yaml
```

### 4. Launch the App

```bash
streamlit run app/streamlit_app.py
```

The app loads a validated model bundle from `artifacts/model_bundle/current/`.
It does not fall back to sample examples in normal runtime. If the bundle is invalid, the app shows the blocking validation issues and asks you to rerun the training command.

For local UI testing only, set `NL2SQL_ALLOW_CANDIDATE_BUNDLE=1` or enable **Use candidate bundle for debugging** in the sidebar. The app then loads `artifacts/model_bundle/candidate/` with a persistent non-production warning; this never promotes the candidate or marks it production-ready.

Development candidate mode (PowerShell):

```powershell
$env:NL2SQL_ENV="development"
$env:NL2SQL_ALLOW_CANDIDATE_BUNDLE="1"
streamlit run app/streamlit_app.py
```

Production mode requires `artifacts/model_bundle/current/bundle_manifest.json` and fails closed if it is missing, invalid, not production-gated, or not fully production ready:

```powershell
$env:NL2SQL_ENV="production"
streamlit run app/streamlit_app.py
```

See [docs/deployment.md](docs/deployment.md) for environment variables and the deployment health check.

---

## Database Connectivity

### SQLite
Select **SQLite** in the app sidebar and provide the path to your `.db` file.

### PostgreSQL
Select **PostgreSQL** in the app sidebar and fill in connection parameters.

> [!IMPORTANT]
> **Security Guardrails**: Credentials are never stored in plaintext. All queries are strictly validated as SELECT-only. PostgreSQL sessions use read-only transactions and statement timeouts.

---

## Model Bundle

The training pipeline produces a **validated model bundle** at:
```
artifacts/model_bundle/current/bundle_manifest.json
```

The bundle contains:
- Retrieval IR model artifacts
- Neural IR model artifacts
- Adaptive ranker weights (if enabled)
- Evaluation reports
- Classification and confusion-matrix reports
- Controlled predicted-SQL execution reports
- Calibration, percentile, latency, and schema-drift baselines
- Champion/challenger statistical comparison
- Quality gate results
- Pipeline execution report

The Streamlit app loads this bundle automatically. Individual artifact folders are not guessed at runtime.

### Bundle Lifecycle
```
Training Pipeline → Candidate Bundle → Quality Gate → Current Bundle
```

If the quality gate fails, the candidate bundle is not promoted to current.

Primary governance artifacts are written to:

```text
artifacts/evaluation/classification_metrics_report.json
artifacts/evaluation/confusion_matrices/
artifacts/evaluation/calibration_report.json
artifacts/evaluation/controlled_predicted_sql_execution_report.json
artifacts/evaluation/champion_challenger_statistical_report.json
artifacts/generic_training/split_distribution_report.json
```

Raw heuristic confidence and calibrated confidence are stored separately. A calibrated score below the learned conformal threshold produces clarification/abstention metadata instead of being presented as a trustworthy probability.

Calibration is computed from evaluation outputs, copied into the bundle, and loaded by the same runtime path used by Streamlit. Runtime prediction results expose `raw_confidence`, `calibrated_confidence`, abstention metadata, and schema drift flags. Promotion uses statistical checks per metric when bootstrap evidence exists and falls back to point-estimate regression checks only for metrics without bootstrap coverage.

### Lifecycle Proof

Every `bundle_manifest.json` includes a `lifecycle_proof` section that records:

| Field | Meaning |
|:---|:---|
| `trained_from_generic_corpus` | Bundle was trained from dataset-balanced generic QueryIR corpus |
| `generic_eval_valid_for_quality_gate` | Evaluation passed all strict validity checks |
| `generic_eval_real_predictions` | Real model predictions (not gold replay) |
| `generic_eval_predictor_used` | A real predictor callable was used |
| `generic_eval_rows_evaluated` | Number of rows evaluated |
| `generic_eval_real_predictions_generated` | Number of real predictions generated |
| `calibration_report_available` | Calibration report exists in bundle |
| `calibration_loaded_in_runtime_smoke` | Calibration was loaded during runtime smoke test |
| `conformal_threshold_available` | Conformal abstention threshold was computed |
| `controlled_predicted_sql_report_attached_to_bundle` | Controlled predicted-SQL report was copied into the candidate bundle |
| `controlled_predicted_sql_report_location` | Whether that report came from `bundle`, `root_artifacts`, or is `missing` |
| `central_sql_validator_used` | Predicted SQL was checked by the central SQL validator before execution |
| `predicted_safe_sql_rate` | Share of predicted SQL accepted as safe SELECT-only SQL |
| `predicted_execution_success_rate` | Share of predicted SQL executions that completed successfully |
| `predicted_row_count_match_rate` | Share of predicted SQL results with matching row counts |
| `evaluation_stability_interpretation` | Explains whether seed metrics are evaluation-only stability or full training variance |
| `quality_gate_passed` | Quality gate passed |
| `bundle_runtime_smoke_passed` | Bundle runtime smoke test passed |
| `production_ready` | All required fields are True - bundle is production-safe |
| `report_identity_validated` | Identity verification confirmed that `bundle_id` and pipeline IDs match exactly between report and candidate directory. |
| `primary_seed_included` | At least the primary seed (`seed_runs_completed >= 1`) completed successfully in tracking. |

Evaluation reports are valid for quality gates **only** when `evaluation_mode = real_model_predictions`, `gold_replay_used = false`, `is_valid_for_quality_gate = true`, `predictor_used = true`, and `real_predictions_generated > 0`. Zero-prediction reports are always invalid.
In addition, strict identity enforcement requires `bundle_id` or matching directory paths to prove the evaluation report belongs to the candidate bundle. Stale reports from previous pipelines will block bundle promotion. Single-seed runs are explicitly tracked and count as a completed seed.

The `GoldReplayBenchmarkRunner` (formerly `BenchmarkRunner`) is a debug-only oracle baseline. Its output is always marked `is_valid_for_quality_gate = false`.

### Execution-Aware Evaluation

Controlled execution-aware evaluation uses a known SQLite fixture database:
```bash
python training/run_execution_aware_evaluation.py --run-controlled-fixtures
```
This creates a temporary database from `evaluation/fixtures/controlled_evaluation.sql`, executes gold SQL for each case in `evaluation/fixtures/controlled_evaluation_cases.jsonl`, and verifies row counts and SQL safety.

---

## Future Architecture Roadmap

The following are planned but deferred until baselines are trustworthy:

- **Relation-Aware Champion/Challenger Validation**: Compare the masked relation-aware encoder with the production baseline before promotion
- **Multi-Seed Training Variance**: Automated metric stability analysis across random seeds
- **Execution-Aware Training Signal**: Using SQL execution results as a training reward signal
- **Schema-Conditional Calibration**: Per-schema calibration curves instead of global
- **Continuous Evaluation Dashboard**: Automated nightly evaluation on connected databases

---

## Training Configuration

### Full Training
```yaml
# configs/training.yaml
pipeline:
  name: full_integrated_training
  fail_fast: true
  promote_if_passed: true

datasets:
  names: [wikisql, spider]
  max_examples: 15000
  max_examples_per_dataset:
    wikisql: 5000
    spider: 5000
  min_converted_examples_required:
    wikisql: 100
    spider: 100

neural:
  config: configs/neural_training_default.yaml

quality_gate:
  required: true

bundle:
  build: true
  promote_if_quality_gate_passes: true
```

### Smoke Training
```yaml
# configs/smoke_training.yaml
pipeline:
  name: smoke_integrated_training
  fail_fast: true
  promote_if_passed: false

datasets:
  names: [wikisql]
  max_examples: 100
  max_examples_per_dataset:
    wikisql: 100
  min_converted_examples_required:
    wikisql: 1

neural:
  config: configs/neural_training_smoke.yaml
  epochs: 1
  batch_size: 4

quality_gate:
  required: false

bundle:
  build: true
  promote_if_quality_gate_passes: false
```

### Advanced Training Options
```bash
# Resume from a specific step
python training/train_model.py --config configs/training.yaml --start-at train_neural_ir_model

# Dry-run to see what would execute
python training/train_model.py --config configs/training.yaml --dry-run

# Resume skipping already-completed steps
python training/train_model.py --config configs/training.yaml --resume
```

### Stepwise Developer Commands

The one-command training pipeline above is the normal path. The commands below are for audits, debugging, release checks, or replaying individual stages when a report points to a specific failure.

Dataset-driven gold learning is the primary self-improvement loop. Manual feedback remains optional and is preserved for human-in-the-loop corrections, but it is not the main source of production training data.

```bash
# Generic readiness and corpus stages
python scripts/audit_generic_nl2sql_readiness.py
python training/build_generic_ir_corpus.py
python training/build_capability_annotations.py
python training/build_retrieval_rag_index.py
python training/train_neural_ir_model.py
python training/evaluate_generic_models.py

# Feedback and self-training stages
python scripts/audit_self_training_readiness.py
python training/evaluate_against_gold.py
python training/mine_validation_errors.py
python training/build_corrections_from_gold.py
python training/train_ranking_from_gold.py
python training/run_self_improvement_loop.py
python training/build_feedback_training_data.py
python training/rebuild_feedback_index.py

# Execution, selection, promotion, and release gates
python scripts/audit_execution_pipeline_readiness.py
python training/run_execution_aware_evaluation.py
python training/select_best_model.py
python training/promote_model_if_better.py
python training/run_full_training_pipeline.py
python training/generate_connected_db_regressions.py
python training/run_connected_db_regressions.py
python training/run_model_quality_gate.py
python training/run_regression_suite.py
python training/run_release_readiness_check.py

# Runtime
streamlit run app/streamlit_app.py
```

---

## Run Tests

```bash
pytest tests/ -v
```

Before release, run the complete checklist in [docs/developer_commands.md](docs/developer_commands.md), including compile, tests, cleanup, integration audit, baseline training, production training, and the production health check.

---

## Advanced & Internal Commands

For detailed internal commands (individual training steps, evaluation suites, calibration, etc.):

- **[docs/developer_commands.md](docs/developer_commands.md)** — Advanced commands for developers
- **[docs/legacy_commands.md](docs/legacy_commands.md)** — Deprecated commands from older versions

---

## Troubleshooting

| Issue | Cause | Solution |
|:---|:---|:---|
| `ModuleNotFoundError: psycopg2` | Missing PostgreSQL wrapper. | `pip install psycopg2-binary` |
| PostgreSQL connection fails | Bad credentials. | Verify credentials, check `pg_hba.conf`. |
| No training data | Datasets not downloaded. | `python scripts/download_datasets.py` |
| "No validated model bundle" | Training did not run, or the production quality gate prevented promotion. | Inspect `artifacts/evaluation/model_quality_gate_report.json`; use `configs/debug_training.yaml` only for explicit candidate UI testing. |
| Full training fails dataset minimums | A requested dataset produced too few usable QueryIR rows. | Open `artifacts/generic_training/dataset_contribution_report.json` and `unsupported_sql_report.json`. |
| Low training performance | Insufficient examples or unsupported SQL coverage. | Increase per-dataset caps or expand QueryIR support based on the unsupported SQL report. |

---

## Evaluation & Lifecycle Details

### simple_query_pass

The `simple_query_pass` metric is **behavior-derived** — it is computed from the actual gold and predicted QueryIR structures, not from a magic metric key. A query qualifies as "simple" if the gold intent is one of `{show_records, count_records, simple_filter}` with no joins. The pass condition checks: intent match, base_table match, no predicted joins, and SQL validation. Non-simple queries receive `None` and are excluded from rate calculations.

Production quality gates require the explicit `simple_query_pass_rate_production` threshold and do not fall back from `intent_accuracy_rate` when `simple_query_pass_rate` is missing. Smoke and developer runs may use the lower smoke threshold for fast feedback, but production mode fails closed.

SQL evaluation records pre/post-repair validation rates, deterministic repair actions, normalized failure categories, abstentions, and `post_abstention_unsafe_sql_count`. Missing LIMIT, safely expandable single-table `SELECT *`, and trailing comments/semicolons may be repaired and revalidated. DML, unsafe keywords, ambiguous references, and invalid QueryIR are never repaired. Invalid outputs that remain unsafe are returned as clarification/abstention with `sql = null`.

### Multi-Seed Evaluation

Two modes exist:

| Mode | `mode` value | `is_valid_for_training_variance_governance` | Description |
|:---|:---|:---|:---|
| Single-seed baseline | `single_seed_baseline` | `false` | Only primary run metrics; no variance computed |
| Evaluation-only stability | `evaluation_only_stability` | `false` | Re-runs evaluation step with different seeds; measures prediction stability, **not** training variance |
| Full retrain multi-seed | `full_retrain_multi_seed` | `true` | Retrains isolated neural artifacts per non-primary seed before evaluation |

True training variance (`is_valid_for_training_variance_governance=true`) requires full re-training per seed. Production training enables this through `seeds.mode: full_retrain_multi_seed`.

Seed reports include `seed_runs`, the model source used for each seed, `stochastic_inference_enabled`, `stochastic_components`, and `evaluation_stability_interpretation` so evaluation-only reports cannot be mistaken for multi-seed re-training evidence.

### Controlled Fixture Evaluation

Two types:

| Type | `evaluation_type` | `measures_model_predictions` | Description |
|:---|:---|:---|:---|
| Gold SQL validation | `controlled_gold_sql_fixture_validation` | `false` | Validates gold SQL executes correctly on fixture DB |
| Predicted SQL execution | `controlled_predicted_sql_execution` | `true` | Loads model, generates predictions, compares with gold results |

Predicted SQL is validated by the central SQL validator before any fixture execution. Validation failures are normalized as `policy_failure_type` (`non_select_statement`, `unsafe_keyword`, `syntax_error`, `select_star_blocked`, `limit_policy_failed`, or `unknown`) and summarized in `policy_failure_type_counts`. Production training requires `controlled_predicted_sql_execution_report.json` to be attached under the candidate bundle `evaluation/` directory before bundle validation.

When controlled predicted-SQL is required, the candidate `bundle_manifest.json` must be readable and contain `bundle_id`; the pipeline fails before evaluation otherwise. Optional mode may use `pipeline_name` only as an explicitly weak identity and emits `candidate_manifest_missing_for_predicted_sql`. The controlled fixture corpus contains at least 10 stable paired cases so bootstrap comparison is available at 10 or more common case IDs; smaller comparisons report `insufficient_common_cases`.

### Relation-Aware Schema Attention

Experimental RAT-SQL-style learnable bias per relation type. Disabled by default (`relation_aware_attention.enabled: false`). Ten relation types: `same_table`, `table_has_column`, `column_belongs_to_table`, `fk_to_pk`, `pk_to_fk`, `primary_key`, `foreign_key_column`, `same_column_name`, `same_data_type`, `unrelated`. Candidate-pairwise attention concatenates table and column masks before softmax, so padded candidates cannot influence valid representations.

`relation_bias_mode` reports the path actually used: `disabled`, `schema_token_role_bias`, `schema_pairwise_relation_bias`, `schema_candidate_pairwise_relation_bias`, or `combined`. Training diagnostics separately record relation IDs as `configured`, `observed_in_dataset`, `observed_in_batch`, and `used_in_forward`; candidate graph statistics report observed min/mean/max counts, padded capacity, matrix size, and mean padding ratio. These are runtime facts rather than config-only claims.

### Curriculum Modes

| Mode | Description |
|:---|:---|
| `ordered_dataset` | Current default: examples ordered by curriculum phase within a single pass |
| `phased_epochs` | Future: per-epoch phase gating (not implemented in this pass) |

`phased_epochs` must be implemented explicitly before it can be reported as active. Ordered-dataset fallback is disabled by default in optimized training and must be intentionally allowed in config.

### Runtime Debug Fields

| Field | Description |
|:---|:---|
| `runtime_source` | Where prediction came from: `dev_fallback`, `model_bundle_candidate`, `model_bundle_current`, `artifact_dirs` |
| `bundle_id` | Unique bundle identifier |
| `bundle_dir` | Filesystem path to the loaded artifact directory |
| `bundle_status` | Bundle status: `candidate`, `validated`, `current` |
| `calibration_loaded` | Whether confidence calibration was loaded |
| `schema_drift_baseline_loaded` | Whether drift baseline was loaded |

### production_ready Split

`production_ready` is now decomposed into three levels:
- **`production_ready_core`**: All critical lifecycle checks (quality gate, evaluation truthfulness, calibration, runtime smoke)
- **`controlled_fixture_ready`**: Controlled fixture evaluation passed (when required)
- **`production_ready_full`**: `production_ready_core AND controlled_fixture_ready`

---

## Development Guidelines

1. **No Raw-SQL Rendering**: SQL generation must go through [ir/ir_to_sql_renderer.py](ir/ir_to_sql_renderer.py).
2. **Unified DB Connector**: DB access via `db/` interfaces using `DatabaseConnectionConfig`.
3. **Safe SQL Executions**: Queries validated by `execution/query_executor.py` before execution.
4. **Legacy Wrapper Usage**: `nl2sql_v1/` is deprecated. Implement new features in `inference/`, `ir/`, `neural_ir/`, `retriever/`.
