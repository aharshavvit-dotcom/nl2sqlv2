# Local QueryIR NL-to-SQL

A CPU-only, local-first NL-to-SQL prototype for SQLite retail analytics.

It uses:

- Streamlit one-page UI
- SQLAlchemy schema inspection
- scikit-learn TF-IDF retrieval
- Runtime schema-aware prediction orchestration
- QueryIR conversion, validation, and SQL rendering
- RapidFuzz slot, schema, and candidate reranking
- SQLGlot validation
- pandas result display
- pytest tests

It does not use APIs, LLMs, LangChain, Vanna, transformers, or GPU.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\create_sample_db.py
python scripts\train_retriever.py
pytest tests/
python scripts\run_golden_tests.py --db data\sample_retail.db --artifact-dir artifacts\option_c_model
streamlit run app\streamlit_app.py
```

Default sample database:

```text
data/sample_retail.db
```

Main smoke question:

```text
Top 5 customers by sales
```

Expected SQL shape:

```sql
SELECT
  customers.customer_name AS customer,
  SUM(orders.amount) AS revenue
FROM orders
JOIN customers ON orders.customer_id = customers.customer_id
GROUP BY customers.customer_name
ORDER BY revenue DESC
LIMIT 5
```

## Flow

1. User provides a SQLite DB path.
2. App reads schema with SQLAlchemy.
3. User asks a natural language question.
4. `RetrievalNL2SQLModel.load(...)` loads the local TF-IDF artifact or the sample fallback.
5. `model.predict(question, schema)` retrieves top-k candidate examples.
6. Runtime reranking scores candidates by retrieval score, schema compatibility, template fit, and slot detectability.
7. Template selection canonicalizes dataset templates such as `rank_dimension` into runtime templates such as `top_n_metric_by_dimension`.
8. Slot resolution detects metric, dimension, entity, limit, sort direction, and date grain.
9. Schema-aware mapping binds slots to real non-sensitive table columns in the connected SQLite schema.
10. Join planning finds required foreign-key paths at runtime.
11. Option C runtime state is converted into QueryIR.
12. QueryIR validation checks required metrics/dimensions, schema references, joins, limits, and sensitive columns.
13. `IRToSQLRenderer` renders bounded SQLite `SELECT` SQL from QueryIR.
14. The central `SQLValidator` checks one safe statement, known tables/columns, no `SELECT *`, no mutation, no comments, bounded `LIMIT`, and no sensitive columns.
15. The app shows SQL, confidence, intent/template, slots, schema mapping, join plan, IR validation, SQL validation checks, candidates, warnings, clarifications, and optional QueryIR debug details.
16. Optional executor runs the SQL read-only only after central validation passes and displays a pandas dataframe.
17. Feedback is appended to `feedback/feedback.jsonl`.

## QueryIR Runtime Refactor

The previous runtime rendered SQL directly inside `PredictionOrchestrator`. The canonical runtime now creates a shared QueryIR first, validates that IR, renders SQL from QueryIR, then validates the final SQL with one central validator.

Architecture:

```text
Question
-> RetrievalNL2SQLModel
-> PredictionOrchestrator
-> retrieval + rerank + slots + schema mapping + joins
-> OptionCToIRConverter
-> QueryIR
-> IRValidator
-> IRToSQLRenderer
-> SQLValidator
-> SQL / Execution
```

The primary interface is:

```python
result = RetrievalNL2SQLModel.load().predict(question, schema)
```

`PredictionResult` includes `query_ir`, `ir_validation`, generated `sql`, SQL `validation`, confidence, slots, schema mapping, join plan, retrieved candidates, warnings, and clarification questions.

This runtime also supports an optional Option A neural QueryIR fallback. Option C remains the primary safe baseline; Option A predicts structured QueryIR labels and schema pointers, then reuses the same IR validator, SQL renderer, SQL validator, Streamlit output, and execution safety path.

Useful verification commands:

```powershell
python scripts\create_sample_db.py
python -m compileall .
pytest tests/
python scripts\run_golden_tests.py --db data\sample_retail.db --artifact-dir artifacts\option_c_model
python scripts\evaluate.py --db data\sample_retail.db --artifact-dir artifacts\option_c_model
streamlit run app\streamlit_app.py
```

Runtime state:

- Canonical runtime: `RetrievalNL2SQLModel -> PredictionOrchestrator -> QueryIR -> IRToSQLRenderer -> SQLValidator`.
- Streamlit uses the QueryIR runtime and executes SQL only through `execution.query_executor`.
- `scripts/evaluate.py` is the compatibility CLI for the active QueryIR evaluator in `scripts/evaluate_runtime.py`.
- `nl2sql_v1/` is legacy/reference code and is not used as the active SQL generation runtime.
- `data/templates.yaml` is legacy/sample template config for IDs and training compatibility; SQL is rendered from QueryIR.
- Product-level revenue uses `SUM(order_items.quantity * order_items.price)` when item-level columns exist.

## Project Layout

```text
app/
  safe_preview.py
  streamlit_app.py
data/
  sample_retail.db
  synonyms.yaml
  templates.yaml
datasets/
  bird_adapter.py
  corpus_builder.py
  spider_adapter.py
  wikisql_adapter.py
feedback/
  .gitkeep
evaluation/
  golden_runtime_tests.jsonl
inference/
  candidate_generator.py
  candidate_reranker.py
  prediction_orchestrator.py
  runtime_schema_context.py
  runtime_join_planner.py
  schema_aware_mapper.py
  slot_resolver.py
  template_selector.py
ir/
  option_c_to_ir.py
  ir_validator.py
  ir_to_sql_renderer.py
  query_ir_models.py
  semantic_metric_resolver.py
validation/
  sql_validator.py
execution/
  query_executor.py
models/
  .gitkeep
nl2sql_v1/
  engine.py
  executor.py
  feedback.py
  join_resolver.py
  renderer.py
  retriever.py
  schema.py
  schema_matcher.py
  slot_extractor.py
  template_adapter.py
  validator.py
retriever/
  retrieval_nl2sql_model.py
scripts/
  check_project.py
  create_sample_db.py
  dataset_paths.py
  download_datasets.py
  evaluate.py
  evaluate_runtime.py
  run_golden_tests.py
  train_retriever.py
  verify_datasets.py
tests/
training/
  train_retriever_from_datasets.py
  evaluate_retriever.py
training_data/
  examples.jsonl
```

## CLI Evaluation

```powershell
python scripts\evaluate.py --db data\sample_retail.db --artifact-dir artifacts\option_c_model
```

The active evaluator runs the structured QueryIR golden suite and writes `evaluation/runtime_evaluation_report.json`.
It reports SQL validity, execution success, QueryIR match rate, failure categories, and per-case details.
`scripts\evaluate_runtime.py` is the implementation module behind the CLI.

## Notes

- The system is deterministic and template-based.
- `SQLGlot` rejects non-`SELECT` statements and multi-statement SQL.
- The executor also enables SQLite `PRAGMA query_only = ON`.
- The sample retail schema has customers, orders, order items, products, stores, and sales reps.

## Roadmap

After runtime stabilization:

1. Build a SQL-to-IR dataset converter.
2. Convert supported and unsupported examples into IR labels.
3. Improve Option A neural IR quality with better schema linking, more data, and error analysis.

## Option A Preparation: SQL-to-IR Corpus Builder

The current runtime already uses QueryIR. The next preparation step is to create gold QueryIR labels from Spider, WikiSQL, and BIRD SQL examples. This does not train a neural model yet. It prepares the data required for a future model that learns `question + schema -> QueryIR`.

Build IR labels from existing dataset adapters:

```powershell
python training_ir\build_ir_training_data.py --datasets wikisql,spider,bird-mini --max-examples 5000 --output-dir data\processed --artifact-dir artifacts\option_a_ir_data
```

Validate a generated IR corpus:

```powershell
python training_ir\validate_ir_corpus.py --input data\processed\ir_training_examples.jsonl
```

Evaluate conversion output:

```powershell
python training_ir\evaluate_ir_conversion.py --input data\processed\ir_test_examples.jsonl --output artifacts\option_a_ir_data\ir_conversion_eval.json
```

Focused conversion tests:

```powershell
pytest tests\test_sql_to_ir_converter.py tests\test_ir_conversion_golden.py
```

Expected generated files:

```text
data/processed/ir_training_examples.jsonl
data/processed/ir_validation_examples.jsonl
data/processed/ir_test_examples.jsonl
data/processed/ir_unsupported_examples.jsonl
data/processed/ir_dataset_stats.json
artifacts/option_a_ir_data/ir_corpus_report.json
artifacts/option_a_ir_data/ir_conversion_eval.json
```

## Option A V1: Neural QueryIR Model

Option A V1 is a lightweight CPU model for structured QueryIR prediction. It does not generate raw SQL. It predicts an intent/template plus schema-linked slots such as base table, metric column, dimension column, date column, filter column, aggregation, order direction, and limit bucket.

The existing runtime safety chain is reused:

```text
Option A labels + schema pointers
-> QueryIR
-> IRValidator
-> IRToSQLRenderer
-> SQLValidator
-> safe execution
```

Option C is still the primary path. Hybrid routing runs Option C first and only tries Option A when Option C confidence is low or SQL validation fails. If the Option A model artifact is missing, the app continues with Option C.

Train a smoke or local Option A model:

```powershell
python training_ir\train_option_a_model.py `
  --train data\processed\ir_training_examples.jsonl `
  --validation data\processed\ir_validation_examples.jsonl `
  --output-dir artifacts\option_a_ir_model `
  --max-examples 500 `
  --epochs 2 `
  --batch-size 8
```

Train a longer local model:

```powershell
python training_ir\train_option_a_model.py `
  --train data\processed\ir_training_examples.jsonl `
  --validation data\processed\ir_validation_examples.jsonl `
  --output-dir artifacts\option_a_ir_model `
  --epochs 5 `
  --batch-size 16
```

Evaluate it:

```powershell
python training_ir\evaluate_option_a_model.py `
  --model-dir artifacts\option_a_ir_model `
  --test data\processed\ir_test_examples.jsonl `
  --output artifacts\option_a_ir_model\evaluation_report.json
```

Run a single prediction:

```powershell
python training_ir\predict_with_option_a.py `
  --model-dir artifacts\option_a_ir_model `
  --db data\sample_retail.db `
  --question "Top 5 customers by sales"
```

Run Streamlit:

```powershell
streamlit run app\streamlit_app.py
```

Limitations:

- Option A V1 is a lightweight CPU model, not a GPT-style LLM.
- It supports structured QueryIR prediction for known intent families.
- It depends on high-quality SQL-to-IR training labels.
- The first pointer heads are fixed-size and intentionally simple.
- It is not expected to be highly accurate immediately; this stage proves local training, loading, QueryIR prediction, validation, rendering, and hybrid fallback.

## Option A V1.5: Quality Improvement and Hybrid Calibration

Option A V1 was the smoke trainable model. Option A V1.5 improves the local path with schema candidates, lexical schema linking, candidate masks, masked pointer loss, curriculum training, error analysis, curated eval cases, and calibrated hybrid routing.

Train with a lightweight curriculum:

```powershell
python training_ir\train_option_a_curriculum.py `
  --train data\processed\ir_training_examples.jsonl `
  --validation data\processed\ir_validation_examples.jsonl `
  --output-dir artifacts\option_a_ir_model `
  --epochs-per-phase 2 `
  --batch-size 8
```

Evaluate with the generated split plus curated sample-retail cases:

```powershell
python training_ir\evaluate_option_a_model.py `
  --model-dir artifacts\option_a_ir_model `
  --test data\processed\ir_test_examples.jsonl `
  --eval-cases evaluation\option_a_eval_cases.jsonl `
  --db data\sample_retail.db `
  --output artifacts\option_a_ir_model\evaluation_report.json
```

Analyze errors by intent, dataset, slot, and validation failure:

```powershell
python training_ir\analyze_option_a_errors.py `
  --model-dir artifacts\option_a_ir_model `
  --test data\processed\ir_test_examples.jsonl `
  --output artifacts\option_a_ir_model\error_analysis_report.json
```

Calibrate the Option C / Option A router:

```powershell
python training_ir\calibrate_hybrid_router.py `
  --eval-cases evaluation\option_a_eval_cases.jsonl `
  --db data\sample_retail.db `
  --option-a-model-dir artifacts\option_a_ir_model `
  --output artifacts\option_a_ir_model\hybrid_calibration.json
```

Use BIRD Mini first. BIRD Full is optional and should be added only after the pipeline is stable.

## Option A V2: Schema-Aware Neural QueryIR Model

Option A V2 improves the neural QueryIR model with schema-aware attention, reusable pointer networks, hard-negative examples, IR repair, and calibrated confidence. It still does not generate raw SQL. The model predicts QueryIR labels and schema pointers, then the same safe runtime handles:

```text
QueryIR -> IRValidator -> IRToSQLRenderer -> SQLValidator -> safe execution
```

Option C remains the default runtime. Option A V2 is used only as an optional fallback/hybrid candidate when its artifact exists and fallback is enabled.

Build hard negatives:

```bash
python training_ir/build_hard_negative_data.py \
  --input data/processed/ir_training_examples.jsonl \
  --output data/processed/ir_hard_negative_examples.jsonl \
  --max-negatives-per-example 5
```

Train a CPU-friendly V2 smoke model:

```bash
python training_ir/train_option_a_v2_model.py \
  --train data/processed/ir_training_examples.jsonl \
  --validation data/processed/ir_validation_examples.jsonl \
  --hard-negatives data/processed/ir_hard_negative_examples.jsonl \
  --output-dir artifacts/option_a_ir_model_v2 \
  --max-examples 500 \
  --epochs 2 \
  --batch-size 8
```

Evaluate V2:

```bash
python training_ir/evaluate_option_a_v2_model.py \
  --model-dir artifacts/option_a_ir_model_v2 \
  --test data/processed/ir_test_examples.jsonl \
  --eval-cases evaluation/option_a_v2_eval_cases.jsonl \
  --db data/sample_retail.db \
  --output artifacts/option_a_ir_model_v2/evaluation_report.json
```

Benchmark Option C, Option A, and hybrid routing:

```bash
python training_ir/benchmark_hybrid_system.py \
  --eval-cases evaluation/hybrid_benchmark_cases.jsonl \
  --db data/sample_retail.db \
  --option-a-model-dir artifacts/option_a_ir_model_v2 \
  --output artifacts/hybrid_benchmark_report.json
```

Analyze IR dataset quality:

```bash
python training_ir/analyze_ir_dataset_quality.py \
  --input data/processed/ir_training_examples.jsonl \
  --unsupported data/processed/ir_unsupported_examples.jsonl \
  --output artifacts/option_a_ir_data/dataset_quality_report.json
```

## Dataset Training Pipeline

The project can now ingest public Text-to-SQL datasets and build a larger Option C retrieval corpus without using LLM APIs, LangChain, Vanna, transformers, GPU, or neural training.

Supported dataset names:

- `wikisql`: WikiSQL, downloaded automatically from Salesforce GitHub.
- `spider`: Spider, downloaded with `gdown` when possible, with manual fallback instructions.
- `bird-mini` or `bird-mini-dev`: BIRD Mini-Dev, downloaded from Hugging Face or read from the normalized manual folder.
- `bird-full`: Full BIRD, only when explicitly requested with `--include-full-bird`. Manual downloads can be prepared without extracting the multi-GB database ZIPs when you only need retrieval or QueryIR-label training.

Folder layout:

```text
data/
  raw/
    spider/
    wikisql/
    bird/
      mini_dev/
        mini_dev_sqlite.json
        dev_tables.json
        dev_databases/
      mini_dev_hf/
      mini_dev_mysql/
      mini_dev_postgresql/
      full/
  processed/
    unified_examples.jsonl
    supported_examples.jsonl
    unsupported_examples.jsonl
    schema_registry.jsonl
    dataset_stats.json
artifacts/
  option_c_model/
    tfidf_vectorizer.pkl
    tfidf_matrix.pkl
    training_examples.jsonl
    train_examples.jsonl
    validation_examples.jsonl
    test_examples.jsonl
    supported_patterns.json
    dataset_stats.json
    training_report.json
    evaluation_report.json
```

Download datasets:

```powershell
python scripts\download_datasets.py --datasets wikisql,spider,bird-mini
```

Manual BIRD Mini-Dev downloads should be normalized to:

```text
data/raw/bird/mini_dev/
  mini_dev_sqlite.json
  dev_tables.json
  dev_databases/
data/raw/bird/mini_dev_mysql/
data/raw/bird/mini_dev_postgresql/
```

Manual BIRD Full downloads should be prepared after placing the official train/dev folders under `data/raw/bird/full/`:

```powershell
python scripts\prepare_bird_full.py --raw-dir data\raw\bird\full
```

This preserves the original downloaded folders and ZIPs, then creates the application-ready split files:

```text
data/raw/bird/full/train.json
data/raw/bird/full/validation.json
data/raw/bird/full/test.json
data/raw/bird/full/train_tables.json
data/raw/bird/full/dev_tables.json
data/raw/bird/full/bird_full_prepared_manifest.json
```

The default split policy keeps official BIRD train rows as `train` and splits official dev databases as close to 50/50 as possible into database-disjoint `validation` and `test` sets.

The loader prefers the SQLite Mini-Dev split at `data/raw/bird/mini_dev/`. The Hugging Face saved dataset remains supported at `data/raw/bird/mini_dev_hf/`.

Spider may require a manual download if Google Drive blocks automated access. In that case, download Spider from the official Yale Spider page and extract it into:

```text
data/raw/spider/
```

The Spider loader expects these normalized files:

```text
data/raw/spider/
  train_spider.json
  train_others.json
  dev.json
  test.json
  tables.json
  database/
```

Full BIRD is intentionally not downloaded by default:

```powershell
python scripts\download_datasets.py --datasets bird-full --include-full-bird
```

That command uses the official BIRD train/dev ZIP links and may download tens of GB of data.
If the connection times out, rerun the same command to resume. For slower networks:

```powershell
python scripts\download_datasets.py --datasets bird-full --include-full-bird --read-timeout 900 --retries 20
```

If you download Full BIRD manually, place or extract it under:

```text
data/raw/bird/full/
```

Then run `python scripts\verify_datasets.py`. A partial or corrupt archive such as an incomplete `train.zip` is reported as `incomplete` and is not treated as a ready training source.

Verify local datasets:

```powershell
python scripts\verify_datasets.py
```

Train the large TF-IDF retriever:

```powershell
python training\train_retriever_from_datasets.py --datasets wikisql,spider,bird-mini --artifact-dir artifacts\option_c_model
```

Include Full BIRD only after `verify_datasets.py` shows it as ready:

```powershell
python training\train_retriever_from_datasets.py --datasets wikisql,spider,bird-mini,bird-full --artifact-dir artifacts\option_c_model
```

Split names are normalized across adapters:

- `train` for training examples.
- `validation` for Spider/WikiSQL dev and BIRD Mini-Dev.
- `test` for test files when SQL labels are present.

The trainer writes separate `train_examples.jsonl`, `validation_examples.jsonl`, and `test_examples.jsonl` files. The retriever trains on `train` by default and falls back to all supported examples only when a selected dataset has no train split, such as BIRD Mini-Dev by itself.

Evaluate retrieval/template accuracy:

```powershell
python training\evaluate_retriever.py --artifact-dir artifacts\option_c_model
```

Run Streamlit:

```powershell
streamlit run app\streamlit_app.py
```

The Streamlit app shows artifact status, dataset coverage, template coverage, training date, and evaluation metrics. If `artifacts/option_c_model/` is present, the app loads the large artifact; otherwise it falls back to the original sample retriever in `models/tfidf_retriever.joblib`.
Internally, Streamlit loads the artifact through `RetrievalNL2SQLModel.load(...)` and generates SQL with `model.predict(question, schema)`.

`model.predict(...)` returns a structured `PredictionResult` with generated SQL, confidence, confidence tier, selected template, resolved slots, schema mapping, join plan, validation checklist, retrieved candidates, warnings, and clarification questions.

`supported_examples.jsonl` feeds the current Option C retrieval/template pipeline. `unsupported_examples.jsonl` is retained for a future Option A neural IR/model path, where more complex nested, set-operation, window, or unsupported SQL patterns can be learned later.

Full validation:

```powershell
pytest tests/
```
