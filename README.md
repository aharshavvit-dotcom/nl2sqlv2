# QueryIR NL-to-SQL

A generic, connected-database natural language to SQL system using **QueryIR** (Query Intermediate Representation). The system converts natural language questions into structured SQL by first producing a semantic IR, then rendering dialect-specific SQL.

---

## 1. Architecture Overview

```
Question ──> Schema Mapping ──> TF-IDF Retrieval ──> Template Selection
     └──> Slot Extraction ──> Join Planning ──> QueryIR Construction
     └──> IR Validation ──> SQL Rendering ──> SQL Validation ──> Execution
```

### Generic Schema-Safe Planning
For simple connected-database questions like `list all users`, the runtime uses a deterministic schema-first planner before retrieval, neural generation, metric mapping, or join planning. This prevents unnecessary joins and sample-domain bias when the app is connected to arbitrary SQLite or PostgreSQL databases.

Direct schema-safe queries choose exactly one base table, select explicit non-sensitive columns instead of `SELECT *`, use `COUNT(*)` for simple counts, and set join policy to `none`.

**Examples:**
```sql
-- list all users
SELECT safe_columns FROM users LIMIT 100

-- list all berth_masters
SELECT safe_columns FROM berth_masters LIMIT 100

-- count users
SELECT COUNT(*) FROM users LIMIT 100

-- show users where role is admin
SELECT safe_columns FROM users WHERE role = 'admin' LIMIT 100
```
For joined or analytical questions, the Retrieval QueryIR Model, Neural QueryIR Model, or Adaptive QueryIR Router is used with join policy enforcement.

### Model & Component Naming

| Model / Component | Description |
|:---|:---|
| **Retrieval QueryIR Model** | TF-IDF retrieval + template matching pipeline. |
| **Neural QueryIR Model** | Pointer-network model trained on IR labels. |
| **Adaptive QueryIR Router** | Confidence-based router selecting the best model. |

> [!NOTE]
> Older documentation and legacy artifact folders may reference "Option A", "Option C", "V1", "V2", or "Hybrid". These have been renamed to follow unified naming rules. See [docs/migration_naming_cleanup.md](docs/migration_naming_cleanup.md).

---

## 2. Installation & Quick Start

Follow these steps to get the environment initialized and run a live demo using a sample SQLite database.

### Step 2.1: Clone and Install Dependencies
Clone the repository and install packages within a Python virtual environment.

**Windows (PowerShell):**
```powershell
git clone <repo-url> nl2sqlv2
cd nl2sqlv2
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
git clone <repo-url> nl2sqlv2
cd nl2sqlv2
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2.2: Create the Sample Database
Run the helper script [scripts/create_sample_db.py](scripts/create_sample_db.py) to build `data/sample_retail.db`. This database includes tables for `customers`, `orders`, `products`, and `order_items`.
```powershell
python scripts/create_sample_db.py
```

### Step 2.3: Launch the Streamlit Interface (Quick Demo)
Launch the interactive web application using Streamlit:
```powershell
streamlit run app/streamlit_app.py
```

#### Connecting Databases in the App:
*   **SQLite**: Select **SQLite** in the sidebar, input the path `data/sample_retail.db` (or any other `.db` file), and click **Connect**.
*   **PostgreSQL**: Select **PostgreSQL** in the sidebar, fill in connection parameters (host, port, username, password, database name, and schema), and click **Connect**.

> [!IMPORTANT]
> **Security Guardrails**: Database credentials are never displayed in logs or stored in plaintext. All queries are strictly validated as SELECT-only before execution. PostgreSQL sessions run with read-only transaction limits (`SET TRANSACTION READ ONLY`) and statement timeouts (`SET statement_timeout = '30s'`).

---

## 3. Data Processing & Corpus Construction

To train and evaluate models at scale, you can ingest gold standard datasets (WikiSQL, Spider, BIRD Mini) and build unified QueryIR training splits.

### Step 3.1: Download and Verify Datasets (Optional)
Fetch external datasets and verify their integrity on your filesystem:
```powershell
python scripts/download_datasets.py --datasets wikisql spider bird-mini
python scripts/verify_datasets.py
```

### Step 3.2: Audit Initial Pipeline Readiness
Run the audit script to check that paths, packages, and database access are ready for dataset operations:
```powershell
python scripts/audit_generic_nl2sql_readiness.py
```

### Step 3.3: Construct QueryIR Training Corpus
Choose between the legacy standalone builder or the unified generic split builder.

#### Method A: Unified Generic Corpus Builder (Recommended)
This script builds training, validation, held-out testing, and unseen-database testing splits. It also checks for data leakage and outputs detailed split quality reports.
```powershell
python training/build_generic_ir_corpus.py `
  --datasets wikisql,spider,bird-mini `
  --max-examples 5000 `
  --output-dir data/processed `
  --artifact-dir artifacts/generic_training
```
This produces the following artifacts:
*   `data/processed/generic_ir_train.jsonl`
*   `data/processed/generic_ir_validation.jsonl`
*   `data/processed/generic_ir_test.jsonl`
*   `data/processed/generic_ir_unseen_db_test.jsonl`
*   `data/processed/generic_ir_unsupported.jsonl`

#### Method B: Standalone IR Training Data Builder (Legacy)
Alternative workflow to construct and validate training examples directly:
```powershell
python training_ir/build_ir_training_data.py `
  --datasets wikisql,spider,bird-mini `
  --output-dir training_data

python training_ir/validate_ir_corpus.py `
  --input training_data/ir_training_examples.jsonl
```

### Step 3.4: Build a Hard-Negative Corpus
Mine hard negatives from your training splits to help models learn contrastive patterns:
```powershell
python training/build_hard_negative_corpus.py `
  --input data/processed/generic_ir_train.jsonl `
  --output data/processed/generic_ir_hard_negatives.jsonl `
  --max-negatives-per-example 5
```

---

## 4. Model Training & Optimization

Once the data is prepped, you can train retrieval and neural components using standard or optimized parameters.

### Step 4.1: Train the Retrieval QueryIR Model & Build RAG Index
Train the TF-IDF and template retrieval components, and build the local RAG indexing structure.

**Train Retrieval Model:**
```powershell
python training/train_retrieval_ir_model.py `
  --datasets wikisql spider bird-mini `
  --max-examples 0
```
Or run the raw training script directly:
```powershell
python training/train_retriever_from_datasets.py `
  --datasets wikisql,spider,bird-mini `
  --artifact-dir artifacts/retrieval_ir_model
```

**Build Retrieval RAG Index:**
```powershell
python training/build_retrieval_rag_index.py `
  --input data/processed/generic_ir_train.jsonl `
  --output-dir artifacts/retrieval_ir_model
```
> [!TIP]
> The RAG index combines TF-IDF example lookup with schema overlap constraints and intent penalties for analytical query mismatches.

### Step 4.2: Train the Neural QueryIR Model (Standard / Baseline)
Train the default sequence-to-sequence neural model, calibrate prediction confidence, and configure the router.

**Train Baseline Neural Model:**
```powershell
python training/train_neural_ir_model.py `
  --epochs 30 `
  --batch-size 32
```
Or train with hard negatives:
```powershell
python training/train_neural_ir_model.py `
  --train data/processed/generic_ir_train.jsonl `
  --validation data/processed/generic_ir_validation.jsonl `
  --hard-negatives data/processed/generic_ir_hard_negatives.jsonl `
  --output-dir artifacts/neural_ir_model `
  --epochs 5 `
  --batch-size 8
```

**Calibrate Model Confidence & Router Scores:**
```powershell
# Calibrate prediction confidence thresholds
python training_ir/calibrate_option_a_confidence.py

# Calibrate confidence-based Adaptive Router weights
python training_ir/calibrate_hybrid_router.py
```

### Step 4.3: Train the Neural QueryIR Model (Optimized Framework)
The optimized training framework supports custom optimizer algorithms (AdamW, SGD, RMSProp), activations (GELU, LeakyReLU), scheduler policies (ReduceLROnPlateau, CosineAnnealing), FFN heads, gradient clipping, early stopping, training diagnostics, and caching.

**1. Audit Optimization Readiness:**
```powershell
python scripts/audit_neural_training_readiness.py
```

**2. Run a Quick Smoke Test:**
Validate the training configuration on a tiny sample slice before embarking on full training:
```powershell
python training/train_neural_ir_optimized.py `
  --config configs/neural_training_smoke.yaml `
  --max-examples 100 --epochs 1
```

**3. Run Optimized Neural Training:**
Train the model using configurable settings loaded from a YAML configuration:
```powershell
python training/train_neural_ir_optimized.py `
  --config configs/neural_training_default.yaml `
  --train data/processed/generic_ir_train.jsonl `
  --validation data/processed/generic_ir_validation.jsonl `
  --output-dir artifacts/neural_ir_model
```

**4. Perform Hyperparameter Grid Search:**
Run parallel experiment configurations defined in grid files and rank the results automatically:
```powershell
python training/run_neural_training_experiments.py `
  --grid configs/neural_experiment_grid.yaml `
  --output-dir artifacts/neural_experiments `
  --max-examples 1000 --epochs 3
```

---

## 5. Dataset-Driven Self-Improvement Loop

The self-improvement pipeline leverages gold datasets rather than manual human feedback. The training manager automatically runs model predictions, compares them against ground truth, flags failure modes, generates correction examples/hard negatives, retrains models, and produces diagnostic reports.

```
Train Baseline ──> Batch Predict ──> Compare vs Gold ──> Mine Validation Errors 
     └──> Retrain Models <── Generate Corrections & Rankers <── Classify Mistakes
```

### Option A: Run the Automated Self-Improvement Loop
Execute the full multi-iteration loop in a single run:
```powershell
python training/run_self_improvement_loop.py `
  --train data/processed/generic_ir_train.jsonl `
  --validation data/processed/generic_ir_validation.jsonl `
  --retrieval-model-dir artifacts/retrieval_ir_model `
  --neural-model-dir artifacts/neural_ir_model `
  --output-dir artifacts/self_training `
  --iterations 2 `
  --max-examples 1000
```
Or execute the legacy training loop script:
```powershell
python training/run_self_training_loop.py `
  --train data/processed/generic_ir_train.jsonl `
  --validation data/processed/generic_ir_validation.jsonl `
  --test data/processed/generic_ir_test.jsonl `
  --output-dir artifacts/self_training `
  --max-iterations 3 `
  --epochs-per-iteration 10 `
  --batch-size 32
```

### Option B: Execute Individual Improvement Steps Manually
If you need granular control, you can run the pipeline stages step-by-step:

**1. Generate Batch Predictions on Validation Data:**
```powershell
python training/run_batch_predictions.py `
  --model-dir artifacts/neural_ir_model `
  --input data/processed/generic_ir_validation.jsonl `
  --output artifacts/self_training/predictions.jsonl
```

**2. Compare Output against Gold Labels:**
```powershell
python training/run_gold_comparison.py `
  --predictions artifacts/self_training/predictions.jsonl `
  --gold data/processed/generic_ir_validation.jsonl `
  --output artifacts/self_training/comparison_report.json
```

**3. Classify Errors & Mine Validation Mistakes:**
```powershell
# Analyze error taxonomies (mismatched joints, table selection mistakes, etc.)
python training/run_error_analysis.py `
  --predictions artifacts/self_training/predictions.jsonl `
  --gold data/processed/generic_ir_validation.jsonl `
  --output artifacts/self_training/error_report.json

# Mine specific validation errors for retraining
python training/mine_validation_errors.py `
  --predictions artifacts/self_training/predictions.jsonl `
  --output-dir data/processed/self_training
```

**4. Construct Correction Examples from Gold Matches:**
Create training instances to correct predicted syntax failures:
```powershell
python training/build_corrections_from_gold.py `
  --predictions artifacts/self_training/predictions.jsonl `
  --output-dir data/processed/self_training
```

**5. Train Neural Candidate Ranker / Classifier:**
Train a candidate ranker model using error/score matrices:
```powershell
# Option A: Train ranking module from gold
python training/train_ranking_from_gold.py `
  --predictions artifacts/self_training/predictions.jsonl `
  --output-dir artifacts/adaptive_ranker

# Option B: Train FFN candidate ranker from features
python training/train_neural_candidate_ranker.py `
  --ranking-data data/processed/self_training/ranking_examples.jsonl `
  --output-dir artifacts/neural_candidate_ranker
```

**6. Generate Self-Improvement Report:**
```powershell
python training/run_improvement_report.py `
  --history-dir artifacts/self_training `
  --output artifacts/self_training/improvement_report.json
```

---

## 6. Evaluation, Model Selection & Promotion

To select models for deployment, evaluate their metrics against static targets and check for regressions.

### Step 6.1: Run Evaluation Suites
Calculate top-k accuracies, unseen-schema generalizations, and router thresholds.

**Evaluate Models on Split Datasets:**
```powershell
python training/evaluate_generic_models.py `
  --test data/processed/generic_ir_test.jsonl `
  --unseen-db-test data/processed/generic_ir_unseen_db_test.jsonl `
  --retrieval-model-dir artifacts/retrieval_ir_model `
  --neural-model-dir artifacts/neural_ir_model `
  --output artifacts/evaluation/generic_model_evaluation_report.json
```

**Evaluate Predictions against Gold IR Labels:**
```powershell
python training/evaluate_against_gold.py `
  --input data/processed/generic_ir_validation.jsonl `
  --retrieval-model-dir artifacts/retrieval_ir_model `
  --neural-model-dir artifacts/neural_ir_model `
  --output artifacts/self_training/validation_predictions.jsonl `
  --report artifacts/self_training/validation_gold_comparison_report.json
```

**Run Baseline Router & Model Evaluation Benchmarks:**
```powershell
python evaluation/run_model_evaluation.py
python evaluation/run_adaptive_router_benchmark.py
```

**Run Unseen-DB Evaluation:**
```powershell
python training/run_unseen_db_benchmark.py `
  --input data/processed/generic_ir_unseen_db_test.jsonl `
  --retrieval-model-dir artifacts/retrieval_ir_model `
  --neural-model-dir artifacts/neural_ir_model `
  --output artifacts/evaluation/unseen_db_benchmark_report.json
```

**Run Execution-Aware SQL Evaluation:**
Evaluate the correctness of rendered queries by executing them on safe target databases:
```powershell
python training/run_execution_aware_evaluation.py `
  --predictions artifacts/self_training/validation_predictions.jsonl `
  --output artifacts/evaluation/execution_aware_evaluation_report.json
```

### Step 6.2: Model Selection & Promotion Gates
Apply quality thresholds to decide if the trained candidate is eligible to replace the active production weights.

**1. Run Model Quality Gate:**
Evaluate metrics against thresholds defined in YAML configs (e.g. minimum accuracy, maximum failure rates):
```powershell
python training/run_model_quality_gate.py `
  --evaluation-report artifacts/evaluation/generic_model_evaluation_report.json `
  --thresholds evaluation/model_quality_thresholds.yaml `
  --output artifacts/evaluation/model_quality_gate_report.json
```

**2. Select the Best Model Configuration:**
```powershell
python training/select_best_model.py `
  --evaluation-report artifacts/evaluation/generic_model_evaluation_report.json `
  --execution-report artifacts/evaluation/execution_aware_evaluation_report.json `
  --thresholds evaluation/model_quality_thresholds.yaml `
  --output artifacts/evaluation/model_selection_report.json
```

**3. Promote Candidate Model (if quality gate passes):**
```powershell
python training/promote_model_if_better.py `
  --candidate-dir artifacts/neural_ir_model `
  --model-name neural_ir_model `
  --evaluation-report artifacts/evaluation/generic_model_evaluation_report.json `
  --execution-report artifacts/evaluation/execution_aware_evaluation_report.json `
  --thresholds evaluation/model_quality_thresholds.yaml `
  --output artifacts/model_registry/promotion_report.json
```

---

## 7. Connected Database Adaptation & Regression Testing

When the application connects to a new live SQLite or PostgreSQL database, the runtime maps schema graphs, resolves ambiguities, and runs target validation tests.

### Step 7.1: Semantic Layer Adapters
When connecting to a new database, the application builds a **Semantic Profile**. This profile maps language expressions to:
*   Primary entities and lookup tables.
*   Bridge tables and join paths.
*   Safe columns (non-sensitive fields) and date dimensions.
*   Likely filters and metrics.

> [!CAUTION]
> If mapping aliases or filter targets are ambiguous, the application stalls execution and requests user clarification before running any generated query.

### Step 7.2: Schema-Specific Regression Suites
Create and run schema-derived regression tests automatically to prevent accuracy loss on targeted databases.

**1. Audit Execution Pipeline Readiness:**
```powershell
python scripts/audit_execution_pipeline_readiness.py
```

**2. Generate Regression Scenarios from Schema Metamodel:**
```powershell
python training/generate_connected_db_regressions.py `
  --schema artifacts/schema/current_schema.json `
  --output artifacts/connected_db_regressions/generated_cases.jsonl
```

**3. Run Connected-Database Regressions:**
```powershell
python training/run_connected_db_regressions.py `
  --cases artifacts/connected_db_regressions/generated_cases.jsonl `
  --output artifacts/connected_db_regressions/regression_report.json
```

**4. Run Baseline Regression Suite:**
```powershell
python training/run_regression_suite.py `
  --cases evaluation/generic_benchmark_cases.jsonl `
  --feedback-regressions data/processed/feedback_safety_regressions.jsonl `
  --output artifacts/evaluation/regression_suite_report.json
```

---

## 8. Release Verification & Pipeline Automation

Validate final artifacts and deploy production packages.

### Step 8.1: Audit Release Readiness
Verify audits, metrics, gate scores, and regressions before making a release:
```powershell
python training/run_release_readiness_check.py `
  --audit-report artifacts/audit/generic_nl2sql_readiness_report.json `
  --evaluation-report artifacts/evaluation/generic_model_evaluation_report.json `
  --quality-gate-report artifacts/evaluation/model_quality_gate_report.json `
  --regression-report artifacts/evaluation/regression_suite_report.json `
  --output artifacts/evaluation/release_readiness_report.json
```

### Step 8.2: Run Full Automated Pipelines (YAML Orchestration)
You can run the entire audit-corpus-train-evaluate pipeline in a single step using pipeline config files:

**Smoke Pipeline Run:**
```powershell
python training/run_full_training_pipeline.py `
  --config pipeline_configs/smoke_training.yaml
```

**Full Generic Pipeline Run:**
```powershell
python training/run_full_training_pipeline.py `
  --config pipeline_configs/full_generic_training.yaml
```

---

## 9. Human-in-the-Loop Feedback (Optional)

Manual user feedback is entirely optional, as dataset-driven self-improvement is the primary training loop. However, you can still collect manual annotations via the Streamlit interface to augment your datasets.

**1. Collect Manual Feedback:**
Submit ratings, corrected SQL statements, or notes within the Streamlit application. Feedback logs are recorded securely in `data/feedback/query_feedback.jsonl` (database connection credentials or secrets are never saved).

**2. Build Feedback Training Data:**
```powershell
python training/build_feedback_training_data.py `
  --feedback data/feedback/query_feedback.jsonl `
  --output-dir data/processed `
  --artifact-dir artifacts/feedback
```

**3. Rebuild Feedback Index:**
Merge feedback-derived examples back into the retrieval model:
```powershell
python training/rebuild_feedback_index.py `
  --feedback-examples data/processed/feedback_positive_examples.jsonl `
  --output-dir artifacts/retrieval_ir_model
```

---

## 10. Run Tests

Verify code correctness using pytest.

**Run Consolidated Module Tests:**
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

**Run All Tests (Including New Module Tests):**
```powershell
pytest tests/ -v
```
*Legacy files are kept in `tests/legacy/` for historical reference.*

---

## 11. Maintenance, Troubleshooting & Dev Notes

### Key Artifact Folders

| Directory | Contents |
|:---|:---|
| `artifacts/retrieval_ir_model/` | Retrieval vectorizers, index mappings, TF-IDF examples |
| `artifacts/neural_ir_model/` | Trained PyTorch weights, vocab configuration, class encoders |
| `artifacts/self_training/` | Self-improvement predictions, error analyses, improvement reports |
| `artifacts/connected_db_regressions/` | Automated schema regression scenarios and execution summaries |
| `evaluation/` | Benchmarks, validation datasets, target quality definitions |
| `training_data/` | Processed intermediate training examples |
| `data/feedback/` | User feedback database logs and positive/negative splits |

### Migrating Older Artifacts
If your local directory contains model files using option letters, run the name migration utility to copy files to correct targets:
```powershell
python scripts/migrate_artifact_names.py
```
To verify the naming compliance of your workspace directory, run:
```powershell
python scripts/repo_cleanup_check.py
```

### Troubleshooting FAQ

| Issue | Cause | Solution |
|:---|:---|:---|
| `ModuleNotFoundError: psycopg2` | Missing PostgreSQL wrapper binary. | Run `pip install psycopg2-binary`. |
| PostgreSQL connection fails | Bad credentials or local host block. | Verify credentials, check port routing, check `pg_hba.conf`. |
| No training data | Datasets have not been downloaded. | Run `python scripts/download_datasets.py`. |
| Low training performance | Insufficient examples or training limit. | Train on more datasets, increase `--max-examples`, or tune YAML configs. |
| "Sample model" warning | Unfitted weights or missing artifacts. | Run a retrieval or neural training run prior to starting the web app. |

### Development Guidelines

1.  **Legacy Wrapper Usage**: The package `nl2sql_v1/` is deprecated. It is only kept for `SchemaGraph` and `TfidfRetriever`. Implement all new features in `inference/`, `ir/`, `neural_ir/`, `retriever/`, `feedback/`, or `neural_optimization/`.
2.  **Unified DB Connector**: DB access must be instantiated via `db/` interfaces. Always import configuration using `DatabaseConnectionConfig`.
3.  **No Raw-SQL Rendering**: SQL generation must go through [ir/ir_to_sql_renderer.py](ir/ir_to_sql_renderer.py). Do not format sql text directly in application scripts or streamlit files.
4.  **Safe SQL Executions**: Queries must be validated by `execution/query_executor.py` before execution.
