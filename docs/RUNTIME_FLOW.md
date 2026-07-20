# Runtime Flow

> Generated: 2026-07-19  
> Canonical runtime: `inference/prediction_orchestrator.py`

## Streamlit → SQL Execution Call Graph

```
app/streamlit_app.py
    ├── ModelBundleLoader.load(bundle_dir)
    │   └── bundle_manifest.json → resolved artifact paths
    │
    ├── _load_model_from_bundle(bundle_info)
    │   └── RetrievalNL2SQLModel.load(...)
    │       ├── TfidfRetriever (nl2sql_v1/retriever.py)
    │       └── PredictionOrchestrator(bundle=bundle_info)
    │
    ├── db/schema_reader.read_database_schema(config)
    │   └── schema_dict_to_graph() → SchemaGraph
    │
    └── model.predict(question, schema, ...)
        └── PredictionOrchestrator.predict()
            │
            ├── Route: DIRECT_PLANNER
            │   └── TableIntentResolver → QueryIR → render → validate
            │
            ├── Route: RETRIEVAL
            │   ├── CandidateGenerator.generate_candidates()
            │   ├── CandidateReranker.rerank_candidates()
            │   ├── TemplateSelector.select_template()
            │   ├── SlotResolver.resolve_slots()
            │   ├── SchemaAwareMapper.map_slots_to_schema()
            │   ├── RuntimeJoinPlanner.plan_joins()
            │   ├── OptionCToIRConverter.convert() → QueryIR
            │   ├── IRValidator.validate(query_ir)
            │   ├── IRToSQLRenderer.render(query_ir) → SQL
            │   └── SQLValidator.validate(sql)
            │
            ├── Route: NEURAL
            │   ├── NeuralIRPredictor.predict(question, schema)
            │   ├── Neural QueryIR → Slot overlay
            │   └── (falls through to retrieval rendering)
            │
            └── Route: HYBRID (adaptive router)
                ├── Retrieval confidence check
                ├── Neural fallback if low confidence
                └── choose_route(retrieval_conf, neural_conf)

            Final:
            ├── PredictionConfidenceCalculator.calculate()
            ├── Abstention check
            └── PredictionResult
                ├── query_ir
                ├── sql
                ├── confidence
                ├── source (retrieval/neural/direct)
                └── validation_result
```

## SQL Safety Chain

```
Generated SQL
    ↓
SQLValidator.validate(sql, schema)
    ├── Parse via sqlglot AST
    ├── Statement type check (SELECT only)
    ├── Schema column validation
    ├── Table existence validation
    └── Policy check (no DDL/DML/admin)
    ↓
execute_select(connection, sql)
    ├── Read-only connection enforcement
    ├── Single statement check
    ├── Statement timeout
    └── Row limit
    ↓
Result DataFrame
```

## Current Issues

1. **No NL2SQLService facade**: Streamlit directly constructs RetrievalNL2SQLModel and PredictionOrchestrator
2. **Legacy model loading**: `_load_model_legacy()` path exists for dev mode with no bundle
3. **Dual artifact directories**: `NEURAL_IR_ARTIFACT_DIR` and `NEURAL_IR_V2_ARTIFACT_DIR` both point to same path
4. **Backward-compatible aliases**: PredictionOrchestrator accepts `option_a_*` parameter names
