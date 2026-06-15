# QueryIR NL-to-SQL

A retrieval-augmented natural language to SQL system using **QueryIR** (Query Intermediate Representation). The system converts natural language questions into structured SQL by first producing a semantic IR, then rendering dialect-specific SQL.

## Architecture

```
Question → TF-IDF Retrieval → Template Selection → Slot Extraction
    → Schema Mapping → Join Planning → QueryIR Construction
    → IR Validation → SQL Rendering → SQL Validation → Execution
```

### Model Naming

| Name | Description |
|------|-------------|
| **Retrieval QueryIR Model** | TF-IDF retrieval + template matching pipeline |
| **Neural QueryIR Model** | Pointer-network model trained on IR labels |
| **Adaptive QueryIR Router** | Confidence-based router selecting the best model |

> **Note**: Older documentation and artifact folders may reference "Option A", "Option C", "V1", "V2", or "Hybrid". These have been renamed. See [migration notes](docs/migration_naming_cleanup.md).

## Setup

### 1. Clone and Install

```powershell
git clone <repo-url> nl2sqlv2
cd nl2sqlv2
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Create Sample Database

```powershell
python scripts/create_sample_db.py
```

This creates `data/sample_retail.db` with tables: `customers`, `orders`, `products`, `order_items`.

### 3. Download Datasets (Optional)

```powershell
python scripts/download_datasets.py --datasets wikisql spider bird-mini
python scripts/verify_datasets.py
```

## Build IR Training Data

Convert downloaded datasets into QueryIR training corpus:

```powershell
python training_ir/build_ir_training_data.py `
  --datasets wikisql,spider,bird-mini `
  --output-dir training_data
```

Validate the corpus:

```powershell
python training_ir/validate_ir_corpus.py `
  --input training_data/ir_training_examples.jsonl
```

## Train Retrieval QueryIR Model

```powershell
python training/train_retrieval_ir_model.py `
  --datasets wikisql spider bird-mini `
  --max-examples 0
```

Or use the underlying script directly:

```powershell
python training/train_retriever_from_datasets.py `
  --datasets wikisql,spider,bird-mini `
  --artifact-dir artifacts/retrieval_ir_model
```

## Train Neural QueryIR Model

```powershell
python training/train_neural_ir_model.py `
  --epochs 30 `
  --batch-size 32
```

Calibrate confidence after training:

```powershell
python training_ir/calibrate_option_a_confidence.py
```

Calibrate the Adaptive Router:

```powershell
python training_ir/calibrate_hybrid_router.py
```

## Run Evaluation

```powershell
python evaluation/run_model_evaluation.py
python evaluation/run_adaptive_router_benchmark.py
```

## Run Streamlit App

```powershell
streamlit run app/streamlit_app.py
```

### Connect SQLite

1. Open the app in your browser
2. Select **SQLite** under Database Connection
3. Enter the path to your `.db` file (default: `data/sample_retail.db`)
4. Click **Connect**

### Connect PostgreSQL

1. Open the app in your browser
2. Select **PostgreSQL** under Database Connection
3. Fill in host, port, database, username, password, SSL mode, and schema
4. Click **Connect**

> **Security**: Passwords are never displayed in the UI or logs. All queries are validated as SELECT-only before execution. PostgreSQL connections use `SET TRANSACTION READ ONLY` and `SET statement_timeout = '30s'`.

## Run Tests

Run the consolidated test suite:

```powershell
pytest tests/test_01_core_ir.py `
  tests/test_02_sql_validation.py `
  tests/test_03_database_connectors.py `
  tests/test_04_retrieval_runtime.py `
  tests/test_05_neural_runtime.py `
  tests/test_06_adaptive_router.py `
  tests/test_07_training_data_pipeline.py `
  tests/test_08_streamlit_app_helpers.py `
  tests/test_09_end_to_end_smoke.py `
  -v
```

Or run all tests:

```powershell
pytest tests/ -v
```

Legacy tests are preserved in `tests/legacy/` for reference.

## Artifact Folders

| Folder | Contents |
|--------|----------|
| `artifacts/retrieval_ir_model/` | Trained TF-IDF retriever, vectorizer, examples |
| `artifacts/neural_ir_model/` | Trained neural IR model, vocab, label maps |
| `evaluation/` | Golden test cases, benchmark results |
| `training_data/` | Training examples, IR corpus |
| `feedback/` | User feedback from the Streamlit app |

### Migrate Old Artifact Names

If you have artifacts under old folder names (`option_c_model`, `option_a_ir_model`, `option_a_ir_model_v2`), run the migration script:

```powershell
python scripts/migrate_artifact_names.py
```

This copies old folders to new names without deleting the originals.

## Verify Cleanup

Run the repository cleanup check to verify naming compliance:

```powershell
python scripts/repo_cleanup_check.py
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: psycopg2` | `pip install psycopg2-binary` |
| PostgreSQL connection fails | Check host, port, firewall, and `pg_hba.conf` |
| No training data | Run `python scripts/download_datasets.py` |
| Low accuracy | Train with more datasets, increase `--max-examples` |
| "Sample model" warning | Train from local datasets in the Streamlit app |

## Development Notes

- The `nl2sql_v1/` package is legacy and used only for `SchemaGraph`, `TfidfRetriever`, and `append_feedback`. The active pipeline is in `inference/`, `ir/`, `neural_ir/`, and `retriever/`.
- The `db/` package provides unified database access. Import `DatabaseConnectionConfig` and connectors from there.
- All SQL generation goes through `ir/ir_to_sql_renderer.py`. Never write raw SQL in the Streamlit app.
- All SQL execution goes through `execution/query_executor.py` with mandatory validation.
