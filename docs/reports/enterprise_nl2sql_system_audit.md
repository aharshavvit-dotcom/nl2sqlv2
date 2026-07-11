# Enterprise NL-to-SQL System Audit

Date: 2026-07-11

Scope: current-state audit for the master engineering prompt. This report intentionally stops before architecture modification. It maps the actual training, inference, IR, validation, execution, retrieval, evaluation, and production bundle paths present in the repository, then records gaps that must be addressed before claiming QueryIR v2, advanced SQL, relation-aware production modeling, or production maturity.

## Audit Decision

The system is more mature than a prototype: it has a canonical integrated training command, run-scoped pipeline state, dataset contribution and leakage checks, QueryIR conversion, retrieval/RAG, neural QueryIR prediction, central SQL validation, execution-aware evaluation, model bundle validation, quality gates, promotion governance, runtime abstention, and privacy-hardened telemetry.

The system is not yet the target architecture described in the master prompt. The active QueryIR is flat, the active neural output space is template/slot based, relation-aware attention exists but is disabled in the canonical config, unsupported advanced SQL examples are quarantined rather than converted into rich partial supervision, and there is no first-class capability taxonomy head.

Do not promote future model or architecture changes without an ablation against the frozen baseline and the evaluation gates already present in this repo.

## Current Architecture Diagram

```text
training/train_model.py
  -> training/config_loader.py
  -> orchestration/pipeline_config.py
  -> orchestration/pipeline_runner.py
  -> orchestration/step_runner.py
      -> DatasetRegistry / GenericIRCorpusBuilder
      -> DatasetSplitManager / DatasetLeakageChecker
      -> RAGIndexBuilder
      -> HardNegativeCorpusBuilder
      -> training/train_neural_ir_optimized.py
      -> evaluate_against_gold / evaluate_generic_models
      -> ModelQualityGate
      -> ModelBundleBuilder / Validator / Promoter

app/streamlit_app.py
  -> ModelBundleLoader
  -> RetrievalNL2SQLModel
  -> PredictionOrchestrator
      -> Generic Direct Planner
      -> RAG Retriever + runtime reranking
      -> Retrieval QueryIR converter
      -> optional NeuralIRPredictor fallback
      -> IRValidator
      -> IRToSQLRenderer
      -> SQLValidator.validate_with_repair
      -> confidence calibration and abstention
  -> execution/query_executor.py
      -> SQLiteConnector or PostgresConnector
```

Evidence:

- Canonical training entry point and pipeline runner: `training/train_model.py:180`, `orchestration/pipeline_config.py:10`, `orchestration/pipeline_runner.py:15`.
- Pipeline step implementation: `orchestration/step_runner.py:251`, `orchestration/step_runner.py:315`, `orchestration/step_runner.py:656`, `orchestration/step_runner.py:922`, `orchestration/step_runner.py:1089`.
- Runtime orchestration: `retriever/retrieval_nl2sql_model.py:196`, `inference/prediction_orchestrator.py:154`.
- IR and SQL safety: `ir/query_ir_models.py:89`, `ir/ir_validator.py:25`, `ir/ir_to_sql_renderer.py:30`, `validation/sql_validator.py:100`.
- Bundle policy: `model_bundle/bundle_loader.py:42`, `model_bundle/bundle_loader.py:167`, `model_bundle/bundle_validator.py:26`, `model_bundle/bundle_promoter.py:248`.

## Target Architecture Delta

```text
Current:
  NL question + schema
    -> direct / retrieval / neural template-slot path
    -> flat QueryIR v1
    -> deterministic SQL
    -> central validation
    -> read-only execution

Target:
  NL question + structured schema graph + values + relations
    -> capability taxonomy head
    -> relation-aware question/schema/candidate encoders
    -> recursive QueryIR v2 decoder
    -> QueryIR v2 structural validation
    -> dialect-aware deterministic renderer
    -> SQL AST/schema/safety validation
    -> calibrated confidence, abstention, execution, monitoring
```

Immediate architectural conclusion: keep deterministic QueryIR-to-SQL. Expand representation and validation before enlarging or replacing the model.

## End-to-End Data Flow

| Stage | Actual owner | Input contract | Output contract | Config source | Failure handling | Tests/evidence | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Raw datasets | `dataset_training/dataset_registry.py`, `datasets/*_adapter.py` | WikiSQL, Spider, BIRD-style source files | `Text2SQLExample`, schema objects | `configs/training.yaml` dataset names and caps | missing datasets fail full training unless explicitly allowed | dataset registry and split tests | BIRD full is present as supported but not canonical default |
| SQL parsing and feature extraction | `ir/sql_to_ir_converter.py:68`, `datasets/sql_feature_extractor.py:9` | source SQL string and schema | parsed AST features, QueryIR or unsupported reason | converter dialect, max limit | unsupported constructs return structured unsupported result | SQL-to-IR legacy tests | Advanced constructs are rejected, not partially supervised |
| Schema extraction | `db/schema_reader.py`, `db/sqlite_connector.py`, `db/postgres_connector.py` | DB connection config | normalized schema dict and `SchemaGraph` | DB config from UI/runtime | connector errors shown, no auto fallback in production | connector tests | PostgreSQL real integration coverage remains incomplete |
| Label generation | `neural_ir/ir_label_encoder.py:29` | flat QueryIR and schema items | intent and pointer labels | hardcoded label maps | missing pointers become `-1` | neural dataset tests | Label space lacks capabilities, HAVING, CASE, subqueries, windows, set ops |
| QueryIR construction | `ir/sql_to_ir_converter.py`, `ir/option_c_to_ir.py`, `neural_ir/option_a_to_ir.py` | SQL AST or runtime slots/neural decode | flat `QueryIR` | converter defaults and runtime schema | validation failure blocks SQL | IR tests | No version field or recursive expression tree |
| Split and leakage | `dataset_training/split_manager.py:105`, `dataset_training/leakage_checker.py:41` | supported and unsupported corpus rows | train/validation/test/unseen/unsupported JSONL | split ratios, manifest | database leakage raises failure | split/leakage tests | Current builder forces new manifest version during integrated build |
| RAG retrieval | `retrieval/rag_index_builder.py:18`, `retrieval/rag_retriever.py:18` | train JSONL | example, schema, pattern indexes | pipeline artifact dirs | missing artifact blocks bundle validation | retrieval tests | Learned semantic reranker not implemented |
| Batch construction | `neural_ir/ir_dataset.py:26` | generic IR JSONL | tensors, masks, relation tensors, labels | neural config max lengths | bad labels masked or capped | neural dataset tests | Relation tensors built even when model config disables relation-aware use |
| Neural model | `neural_ir/attention_model.py:99` | question/schema/candidate tensors | multi-head logits and pointer logits | `configs/neural_training_default.yaml` | masked cross entropy, hard-negative optional | model forward tests | Relation-aware attention disabled in default config |
| Optimization | `training/train_neural_ir_optimized.py` | loaders and config | `model.pt`, metrics, manifests | canonical neural config | hard negatives required outside smoke if configured | training smoke tests | Learned/dynamic loss weighting is not implemented; weights are static |
| Inference preprocessing | `retriever/retrieval_nl2sql_model.py`, `inference/prediction_orchestrator.py` | question, schema graph, retriever | `PredictionResult` | bundle manifest and runtime env | production bundle load fails closed | runtime tests | A guarded sample-retail correction exists outside QueryIR modeling |
| Routing | `inference/prediction_orchestrator.py:844` | retrieval result and neural raw result | selected route, result | hybrid calibration and bundle policy | neural errors fall back to retrieval unless forced diagnostics | router tests | No capability-aware router head |
| SQL rendering | `ir/ir_to_sql_renderer.py:30` | flat QueryIR | deterministic SELECT SQL | dialect from schema/IR | invalid IR skips rendering | renderer tests | No HAVING, CASE, subquery, window, set-op rendering |
| SQL validation | `validation/sql_validator.py:100` | SQL and optional schema | validation payload and repair | max limit, dialect | non-SELECT and unsafe keywords rejected | SQL validation tests | Allows only `exp.Select`; CTE support is not explicit |
| Execution | `execution/query_executor.py:13`, `db/sqlite_connector.py:90`, `db/postgres_connector.py:156` | validated SQL | dataframe/rows | DB config, max rows | validation or DB errors raise/return error | DB tests | Connector-level SQL validation currently runs without schema in connector methods |
| Confidence and abstention | `inference/prediction_confidence.py`, `inference/prediction_orchestrator.py:1093` | validation, mapping, route, calibration | raw/calibrated confidence and abstention reason | calibration report | invalid IR/SQL forces abstention | confidence/router tests | No stochastic agreement or conformal per-schema calibration |
| Feedback and telemetry | `feedback/*`, `inference/telemetry_logger.py:36` | user feedback, prediction result | JSONL feedback/telemetry | env flags for raw logging | raw text disabled by default, PII scrubbed | telemetry tests | Feedback review workflow exists partially, not full human approval registry |
| Bundle/promotion | `model_bundle/*` | work artifacts and reports | candidate/current bundle | training config and quality gate | validation/promotion fail closed | bundle lifecycle tests | Promotion recovery is documented but not fully integration-proven |

## Capability Matrix

| Capability | Current status | Evidence | Gap |
| --- | --- | --- | --- |
| Simple `SELECT` | Supported | `IRToSQLRenderer.render_select`, `SQLValidator` | Direct planner can default projection, but semantic accuracy must be monitored |
| Multi-column select | Partial | `QueryIR.dimensions` and record select rendering | Neural label encoder predicts one dimension pointer |
| `WHERE` filters | Supported for AND-compatible basic filters | `IRFilter`, SQL-to-IR rules, runtime slot resolver | OR filters rejected |
| Aggregation | Supported for COUNT/SUM/AVG/MIN/MAX | `IRMetric`, label maps, renderer | Arithmetic limited to approved revenue expression |
| GROUP BY | Supported for one dimension/trend | renderer group by | Multi-dimension grouping absent in neural labels |
| ORDER BY/LIMIT | Supported | `IROrderBy`, renderer limit | Limit buckets are coarse |
| Joins | Supported for equality joins and runtime FK paths | `IRJoin`, `RuntimeJoinPlanner` | No learned join-path scorer as primary head |
| HAVING | Unsupported | `SQLToIRConverter._reject_unsupported` | Needs QueryIR field, parser, renderer, validator, data |
| CASE | Unsupported | converter rejects CASE | Needs expression tree support |
| Subqueries | Unsupported | converter rejects nested query | Needs recursive query nodes |
| Window functions | Unsupported in active QueryIR | converter rejects window | Needs window expression and renderer |
| Set operations | Unsupported | converter rejects set operation | Needs branch QueryIR validation |
| DML/DDL/Admin | Rejected | `SQLValidator.BLOCKED_KEYWORDS`, select-only check | Need capability classifier labels for user-facing reason |
| Capability taxonomy | Absent | no `SUPPORTED_*` / `UNSUPPORTED_*` head found | Implement separate understanding/support taxonomy |
| Confidence calibration | Partial | ECE/Brier/conformal threshold and runtime loading | Degenerate calibration disables thresholding |
| Production bundle loading | Supported | `ModelBundleLoader.load_current` | Current bundle must contain full proof to be deployable |

## Traceability Matrix

| Component | Training | Validation | Inference | Production | Tested | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Dataset registry | Yes | Yes | No | Indirect | Yes | Full dataset presence depends on local files |
| SQL-to-IR converter | Yes | Yes | No | Indirect | Yes | Rejects advanced SQL without partial supervision |
| QueryIR v1 models | Yes | Yes | Yes | Yes | Yes | Not versioned recursive QueryIR v2 |
| IR validator | Yes | Yes | Yes | Yes | Yes | No advanced structural validation |
| SQL renderer | Yes | Yes | Yes | Yes | Yes | Missing HAVING/CASE/subquery/window/set |
| SQL validator | Yes | Yes | Yes | Yes | Yes | Schema validation depends on schema being passed |
| RAG indexes | Yes | Yes | Yes | Yes | Yes | Semantic learned reranker absent |
| Neural model | Yes | Yes | Yes | Bundle gated | Yes | Relation-aware mode disabled by canonical config |
| Capability classifier | No | No | No | No | No | Required new component |
| Hard negatives | Yes | Yes | Indirect | Bundle proof | Yes | Not tied to all capability classes |
| Curriculum | Yes | Metrics | No | Manifest proof | Yes | Ordered buckets only, not phased epochs |
| Calibration | Eval-derived | Yes | Yes | Bundle gated | Yes | Per-schema/stochastic confidence absent |
| Execution-aware eval | Yes | Yes | No | Quality gate | Yes | Connected DB integration not fully verified |
| Telemetry | No | No | Yes | Yes | Yes | No dashboard implementation |
| Model bundle | Yes | Yes | Yes | Yes | Yes | Promotion rollback needs integration proof |
| Feedback loop | Partial | Partial | Yes | Partial | Yes | Human approval/versioned registry incomplete |

## Current-State Findings and Root Causes

### F1. QueryIR is flat and v1-style

Current implementation: `QueryIR` has top-level fields for metrics, dimensions, filters, date filters, joins, group_by, order_by, and limit. It has no `query_ir_version`, recursive expression nodes, subquery node, window definition, set operation, HAVING field, CASE expression, `offset`, or `required_capabilities`.

Root cause: model labels, renderer, validator, and SQL-to-IR converter are all built around template slots. The valid output space cannot include constructs the IR cannot represent.

Correction: introduce QueryIR v2 in parallel with a migration layer. Do not remove v1 until bundle loading, renderer, evaluator, and tests support both.

Test proof: serialization round trip, v1-to-v2 migration tests, renderer parse-back tests, schema validation tests for each new expression type.

### F2. Advanced SQL is rejected cleanly but supervision is lost

Current implementation: SQL-to-IR rejects nested queries, set operations, windows, HAVING, CASE, OR filters, and most computed select expressions. Unsupported rows are retained with unsupported reason and feature reports.

Root cause: representation and labels cannot encode these constructs. The corpus builder has a quarantine/report path, not a partial-label extraction path.

Correction: add a capability extraction pass that records required tables, columns, join paths, aggregation, filters, nesting depth, window info, set-op type, and safety class even when full QueryIR conversion fails.

Test proof: unsupported SQL examples produce capability labels and auxiliary schema-linking labels without executable target SQL.

### F3. Relation-aware attention exists but is not active baseline

Current implementation: `SchemaAwareOptionAIRModel` has relation-bias paths, relation tensors are built by `IRTrainingDataset`, and `NeuralIRPredictor` passes relation tensors when enabled. Canonical `configs/neural_training_default.yaml` has `relation_aware_attention.enabled: false`.

Root cause: relation-aware code is experimental and has no ablation evidence in the canonical training path.

Correction: run A0 baseline, then enable one relation path at a time with controlled ablations and compare unseen DB, renamed schema, join accuracy, latency, and memory.

Test proof: relation mode appears in model outputs, non-padded masks are enforced, and checkpoint reports record active relation path.

### F4. Capability understanding is not separated from installed support

Current implementation: unsupported reasons and abstention reasons exist, but there is no explicit capability taxonomy head such as `SUPPORTED_SUBQUERY_SCALAR` or `UNSUPPORTED_MUTATION`.

Root cause: intent labels are template IDs, and unsupported SQL is not converted into capability-supervision rows.

Correction: add `capability_labels.py`, capability extraction from SQL/question, a model head `P(capability | q, S)`, and runtime policy mapping capability-to-support.

Test proof: mutation, DDL, unsupported dialect, ambiguous request, missing schema element, subquery/window/set examples are classified separately from generic failure.

### F5. Runtime contains a guarded sample-retail corrective rule

Current implementation: `RetrievalNL2SQLModel._normalize_runtime_result` rewrites a customer-sales query into a hardcoded customer/orders SQL pattern under specific sample schema conditions.

Root cause: business semantic correction was added in runtime normalization rather than learned/schema-general QueryIR planning.

Correction: move this behavior into semantic metric resolution, training examples, and explicit regression tests; remove hardcoded runtime SQL rewrite after equivalent QueryIR path passes.

Test proof: customer-sales regression passes without runtime SQL rewrite.

### F6. Connector execution safety is strong but schema validation is not owned by connectors

Current implementation: app execution passes the prior validation result into `execute_query`; connector `execute_readonly` methods also run SQLValidator, but without schema.

Root cause: semantic/schema validation is expected earlier in the orchestrator, while connector methods act as a safety boundary.

Correction: pass schema into connector validation or require an explicit `validated_schema_fingerprint` proof when using `execute_readonly`.

Test proof: direct connector execution rejects unknown columns/tables when schema is available.

### F7. Static multi-task loss weighting

Current implementation: `MultiTaskLossWeighter` combines configured static weights. Hard-negative loss is optional and configured.

Root cause: learned loss weighting and gradient-balancing are not implemented.

Correction: experiment with uncertainty weighting or GradNorm after baseline metrics are frozen. Keep static weights as A0.

Test proof: per-head losses and gradient norms are logged, and slot metrics do not regress.

## QueryIR v2 Contract To Implement Next

Do not replace v1 abruptly. Add v2 as a versioned model with migration.

Minimal v2 fields:

```json
{
  "query_ir_version": "2.0",
  "query_type": "SELECT",
  "select_items": [],
  "from_items": [],
  "joins": [],
  "where": null,
  "group_by": [],
  "having": null,
  "window_definitions": [],
  "qualify": null,
  "order_by": [],
  "limit": null,
  "offset": null,
  "set_operation": null,
  "required_capabilities": [],
  "confidence": {},
  "metadata": {}
}
```

Required expression node types:

- `COLUMN`
- `LITERAL`
- `FUNCTION`
- `AGGREGATION`
- `BINARY_OPERATION`
- `UNARY_OPERATION`
- `CASE_EXPRESSION`
- `SUBQUERY`
- `WINDOW_EXPRESSION`

Initial implementation order:

1. Versioning, schema validation, v1-to-v2 migration, v2-to-v1 compatibility where possible.
2. HAVING and OR filters.
3. CASE expressions.
4. Scalar aggregate subqueries.
5. `IN` and `NOT IN` subqueries.
6. `EXISTS` and `NOT EXISTS`.
7. Basic window expressions.
8. Set operations.

## Required Experiments

| ID | Independent variable | Baseline | Metrics | Decision rule |
| --- | --- | --- | --- | --- |
| A0 | Current canonical config | Current bundle/work artifacts | all existing quality gate metrics | freeze as baseline |
| A1 | capability taxonomy extraction only | A0 | capability accuracy, unsupported reporting | implement if no training regression |
| A2 | QueryIR v2 migration only | A0 | round-trip, SQL validation, backward load | implement if v1 behavior unchanged |
| A3 | relation-aware schema pairwise bias | A0 | unseen DB, renamed schema, join exact match | keep only if statistically useful |
| A4 | candidate pairwise relation bias | A3 or A0 | pointer accuracy, latency | keep if pointer improvement offsets cost |
| A5 | learned loss weighting | best previous | per-head F1, semantic pass | keep if no dominant regression |
| A6 | semantic RAG reranker | best previous | retrieval precision, final semantic pass | keep if final accuracy improves |

## Test Strategy Additions

Add before architecture changes:

- `tests/test_query_ir_v2_models.py`: schema validation, recursive expressions, v1 migration.
- `tests/test_capability_taxonomy.py`: SQL/question capability labels and safety labels.
- `tests/test_sql_to_ir_partial_supervision.py`: unsupported SQL still emits auxiliary labels.
- `tests/test_renderer_v2_constructs.py`: HAVING, CASE, scalar subquery, IN, EXISTS, window, set op snapshots plus sqlglot parse.
- `tests/test_training_inference_parity.py`: same example produces identical tokenization, candidate ordering, masks, relation matrices, and label maps.
- `tests/test_connector_schema_validation.py`: connector path rejects unknown table/column when schema proof is supplied.

## Risk Register

| Risk | Severity | Evidence | Mitigation |
| --- | --- | --- | --- |
| Representation cannot express requested SQL | High | Flat QueryIR v1 | Build QueryIR v2 first |
| Model appears improved by memorizing templates | High | template/slot labels | unseen DB, renamed schema, opaque schema evaluations |
| Advanced examples become only unsupported counts | High | unsupported rows lack auxiliary labels | capability and partial supervision extractor |
| Relation-aware code assumed active | Medium | default disabled | manifest/runtime proof plus ablations |
| Runtime hardcoded correction hides model failure | Medium | customer-sales rewrite | move to semantic layer and remove after tests |
| SQL is validated without schema at connector boundary | Medium | connector calls omit schema | pass schema/proof into connector execution |
| Calibration degeneracy disables abstention | Medium | quality gate handles degenerate calibration | require non-degenerate production calibration |
| Promotion rollback not integration-proven | Medium | docs note incomplete verification | add transactional promotion integration test |

## Roadmap

Phase 0, baseline freeze:

- Run the readiness scripts and current smoke tests.
- Archive current effective config, dataset contribution report, unsupported SQL report, generic evaluation report, calibration report, and bundle validation report.
- Record current relation-aware flag and QueryIR version as baseline facts.

Phase 1, data and evaluation hardening:

- Implement capability taxonomy and partial supervision for unsupported rows.
- Add capability metrics and confusion matrices.
- Add training/inference parity tests.

Phase 2, QueryIR expansion:

- Implement QueryIR v2 and migration.
- Add HAVING, CASE, subquery, window, and set-op representation in staged order.
- Expand renderer and validators with parse-back tests.

Phase 3, schema representation:

- Promote identifier splitting, datatype/key/node-role embeddings, and relation graph features into the canonical model config through ablation.

Phase 4, relation-aware architecture:

- Enable relation-aware attention only after A0/A1/A2 evidence exists.
- Compare each relation path independently.

Phase 5, production integration:

- Bundle v2 artifacts with compatibility metadata.
- Add observability dashboards and feedback review registry.
- Require controlled predicted-SQL execution evidence before promotion.

## Verification Run

Executed on 2026-07-11 after the audit and README command-map update:

| Command | Result | Notes |
| --- | --- | --- |
| `python scripts/audit_integration_readiness.py` | PASS | 24 passed, 0 failed |
| `python scripts/audit_execution_pipeline_readiness.py` | PASS | 5 passed, 0 failed, 0 warnings |
| `python scripts/audit_generic_nl2sql_readiness.py` | PASS | 13 passed, 0 failed, 0 warnings |
| `python scripts/audit_self_training_readiness.py` | PASS | 6 passed, 0 failed, 0 warnings |

The initial execution, generic, and self-training audits reported README documentation gaps. `README.md` now includes the required stepwise developer commands and explicitly states that dataset-driven gold learning is the primary self-improvement loop while manual feedback is optional.

## Acceptance Gate For This Audit

This audit is complete when:

- Current training and runtime owners are mapped.
- Active capabilities are separated from dormant/experimental code.
- Known QueryIR/model/data/evaluation/security gaps are documented.
- No architecture/code mutation is made before this baseline.
- Readiness checks are run or failures are reported. Current status: all four readiness audits pass.
