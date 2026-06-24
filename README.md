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

Full training builds a dataset-balanced generic QueryIR corpus. WikiSQL, Spider, and BIRD Mini each receive their own sampling cap, and the run fails if any requested full-training dataset does not contribute the configured minimum number of converted QueryIR examples. Unsupported SQL is written to `artifacts/generic_training/unsupported_sql_report.json` so current QueryIR coverage gaps are visible.

The connected-database runtime is schema-neutral. Bundled `orders` / `customers` / `products` mappings are enabled only when the complete sample-retail schema signature is present. Other databases derive table, metric, dimension, and filter vocabulary from their own schema; simple single-table questions bypass retrieval and neural routing and cannot add joins.

Evaluation is multi-level: intent, base table, slots, join decisions, router decisions, QueryIR validity, SQL validation, structural/execution match, and safety. Reports include accuracy, macro/micro/weighted F1, confusion matrices, p50/p95/p99 loss/confidence/latency/drift statistics, ECE/Brier calibration, and a conformal abstention threshold. Evaluation reports are valid for quality gates only when `evaluation_mode = real_model_predictions`, `gold_replay_used = false`, and `is_valid_for_quality_gate = true`. Gold replay is a debug baseline only and cannot be promoted.

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
| `quality_gate_passed` | Quality gate passed |
| `bundle_runtime_smoke_passed` | Bundle runtime smoke test passed |
| `production_ready` | All required fields are True — bundle is production-safe |

Evaluation reports are valid for quality gates **only** when `evaluation_mode = real_model_predictions`, `gold_replay_used = false`, `is_valid_for_quality_gate = true`, `predictor_used = true`, and `real_predictions_generated > 0`. Zero-prediction reports are always invalid.

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

- **Relation-Aware Attention**: Cross-table schema encoding in the neural IR model
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
  names: [wikisql, spider, bird-mini]
  max_examples: 5000
  max_examples_per_dataset:
    wikisql: 5000
    spider: 5000
    bird-mini: 5000
  min_converted_examples_required:
    wikisql: 100
    spider: 100
    bird-mini: 100

neural:
  epochs: 5
  batch_size: 8

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

---

## Run Tests

```bash
pytest tests/ -v
```

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
| "No validated model bundle" | Training not run. | `python training/train_model.py --config configs/training.yaml` |
| Full training fails dataset minimums | A requested dataset produced too few usable QueryIR rows. | Open `artifacts/generic_training/dataset_contribution_report.json` and `unsupported_sql_report.json`. |
| Low training performance | Insufficient examples or unsupported SQL coverage. | Increase per-dataset caps or expand QueryIR support based on the unsupported SQL report. |

---

## Development Guidelines

1. **No Raw-SQL Rendering**: SQL generation must go through [ir/ir_to_sql_renderer.py](ir/ir_to_sql_renderer.py).
2. **Unified DB Connector**: DB access via `db/` interfaces using `DatabaseConnectionConfig`.
3. **Safe SQL Executions**: Queries validated by `execution/query_executor.py` before execution.
4. **Legacy Wrapper Usage**: `nl2sql_v1/` is deprecated. Implement new features in `inference/`, `ir/`, `neural_ir/`, `retriever/`.
