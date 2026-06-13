# Local Retrieval NL-to-SQL V1

A CPU-only, local-first NL-to-SQL prototype for SQLite retail analytics.

It uses:

- Streamlit one-page UI
- SQLAlchemy schema inspection
- scikit-learn TF-IDF retrieval
- Runtime schema-aware prediction orchestration
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
python -m pytest
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
14. The central `SQLValidator` checks one safe statement, known tables/columns, no `SELECT *`, no mutation, no comments, and no sensitive columns.
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
-> OptionCToIRConverter
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

This prepares the project for a future Option A neural model without adding one now. Option A can later replace retrieval, slot resolution, and schema mapping with a model that emits the same QueryIR, while keeping the IR validator, SQL renderer, SQL validator, Streamlit output, and execution safety unchanged.

Useful verification commands:

```powershell
python scripts\create_sample_db.py
python -m compileall .
pytest tests/
streamlit run app\streamlit_app.py
```

## Project Layout

```text
app/
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
  create_sample_db.py
  dataset_paths.py
  download_datasets.py
  evaluate.py
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
python scripts\evaluate.py --db data\sample_retail.db
```

The evaluator reports retrieval accuracy, SQL validation rate, executable query rate, and the generated SQL for the main smoke question.

## Notes

- The system is deterministic and template-based.
- `SQLGlot` rejects non-`SELECT` statements and multi-statement SQL.
- The executor also enables SQLite `PRAGMA query_only = ON`.
- The sample retail schema has customers, orders, order items, products, stores, and sales reps.

## Dataset Training Pipeline

The project can now ingest public Text-to-SQL datasets and build a larger Option C retrieval corpus without using LLM APIs, LangChain, Vanna, transformers, GPU, or neural training.

Supported dataset names:

- `wikisql`: WikiSQL, downloaded automatically from Salesforce GitHub.
- `spider`: Spider, downloaded with `gdown` when possible, with manual fallback instructions.
- `bird-mini` or `bird-mini-dev`: BIRD Mini-Dev, downloaded from Hugging Face or read from the normalized manual folder.
- `bird-full`: Full BIRD, only when explicitly requested with `--include-full-bird` and only usable after complete ZIP extraction.

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

Include Full BIRD only after `verify_datasets.py` shows it as present/complete:

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
