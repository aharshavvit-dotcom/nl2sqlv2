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
- Neural QueryIR training
- Gold validation & error mining
- Adaptive ranker training
- Evaluation & quality gates
- Model bundle creation & promotion

For a quick smoke test:
```bash
python training/train_model.py --config configs/smoke_training.yaml
```

### 4. Launch the App

```bash
streamlit run app/streamlit_app.py
```

The app loads a validated model bundle from `artifacts/model_bundle/current/`.

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
- Quality gate results
- Pipeline execution report

The Streamlit app loads this bundle automatically. Individual artifact folders are not guessed at runtime.

### Bundle Lifecycle
```
Training Pipeline → Candidate Bundle → Quality Gate → Current Bundle
```

If the quality gate fails, the candidate bundle is not promoted to current.

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
| Low training performance | Insufficient examples. | Increase `--max-examples` or use more datasets. |

---

## Development Guidelines

1. **No Raw-SQL Rendering**: SQL generation must go through [ir/ir_to_sql_renderer.py](ir/ir_to_sql_renderer.py).
2. **Unified DB Connector**: DB access via `db/` interfaces using `DatabaseConnectionConfig`.
3. **Safe SQL Executions**: Queries validated by `execution/query_executor.py` before execution.
4. **Legacy Wrapper Usage**: `nl2sql_v1/` is deprecated. Implement new features in `inference/`, `ir/`, `neural_ir/`, `retriever/`.
