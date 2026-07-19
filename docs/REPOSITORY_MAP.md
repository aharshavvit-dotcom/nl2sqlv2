# Repository Map

Generated: 2026-07-15T17:19:37+00:00
Branch: repository-cleanup/2026-07-15
HEAD: 739def558686b0c6caa1b398e07920f5e77b2356
Requested baseline commit: 739def558686b0c6caa1b398e07920f5e77b2356

## Baseline State

- Working tree clean at start: False
- Modified/untracked entries at start: 208
- Active model-bundle path: artifacts/model_bundle/current
- Current production bundle exists locally: False
- Singleton candidate bundle exists locally: True
- Run-scoped candidate bundles found: 11
- Active dataset split: semantic_v2
- Latest successful training run: 20260712T082307_834f4eb9
- Latest failed training run: 20260712T044706_01b12c98

## Folder Inventory

| Folder | Purpose | Why required | Canonical owner | Generated or source | Cleanup rule |
| --- | --- | --- | --- | --- | --- |
| .github | Continuous-integration workflow configuration. | Runs automated validation outside the local development machine. | DevOps/Test | configuration | Keep when workflows match supported commands; review untracked workflow additions before commit. |
| app | User-facing Streamlit application and safe preview helpers. | Provides the runtime interface for connected-database NL-to-SQL usage. | Runtime | source | Keep canonical app entry points; do not import training paths in normal runtime. |
| artifacts | Generated training, audit, evaluation, and model-bundle outputs. | Captures reproducibility evidence and local runtime bundles. | MLOps | generated | Keep only active/protected artifacts and cleanup reports in Git; most contents stay ignored. |
| artifacts/repository_cleanup | Machine-readable cleanup inventory and governance reports. | Records evidence for retained, deleted, archived, and review-required items. | Documentation/MLOps | generated documentation | Regenerate with scripts/generate_repository_cleanup_inventory.py after structural changes. |
| capabilities | SQL capability taxonomy, extraction, and reporting contracts. | Training and quality gates need consistent supported/unsupported SQL labels. | ML/Data | source | Keep as canonical capability contract unless replaced by a tested taxonomy module. |
| clarification | Clarification state, ambiguity detection, and question generation. | Runtime needs safe abstention and follow-up behavior for ambiguous requests. | Runtime | source | Keep while prediction orchestration exposes clarification metadata. |
| configs | Canonical runtime and training configuration files. | Training, smoke, baseline, and production paths are driven by validated config. | MLOps | configuration | Keep consumed configs; remove fields only after consumer checks prove non-use. |
| connected_db_testing | Generated connected-database regression case support. | Validates schema-general behavior against live or generated schemas. | Database/Test | source | Keep if referenced by regression scripts or tests. |
| data | Lightweight semantic source data and frozen split manifests. | Runtime synonym defaults and immutable split membership are required for reproducible training. | Data | mixed source/generated | Track small canonical YAML and frozen split manifests; keep raw/processed datasets ignored. |
| data/splits | Frozen dataset split manifests and ID lists. | Training and evaluation must not silently change split membership. | Data/MLOps | generated governance data | Never overwrite a frozen split; create a new version when membership changes. |
| dataset_training | Dataset construction, split, leakage, and corpus-quality tooling. | Canonical training needs reproducible corpora and leakage checks. | Data/ML | source | Keep canonical builders; consolidate only with tests covering corpus outputs. |
| datasets | External dataset adapters and schema normalization utilities. | Generic training depends on normalized WikiSQL, Spider, and BIRD records. | Data | source | Keep adapters referenced by dataset loaders and training builders. |
| db | Database connection, schema reading, and dialect boundaries. | Runtime and evaluation require safe schema discovery for SQLite/PostgreSQL. | Database | source | Keep connector abstractions and dialect-specific implementations. |
| deployment | Production-readiness helpers. | Deployment checks summarize whether the repository is safe to run. | DevOps | source | Keep while scripts or tests import production readiness checks. |
| docs | Canonical architecture, policy, developer, and specification documentation. | New engineers need current command, runtime, and governance guidance. | Documentation | documentation | Keep canonical docs; move run-specific generated reports out of general docs after review. |
| docs/architecture | Architecture and governance policy documents. | Explains production runtime, bundle lifecycle, privacy, database execution, and lineage. | Architecture/Documentation | documentation | Keep when paths and commands validate against current code. |
| docs/reports | Historical/generated audit reports. | Some reports preserve decision history, but run-specific reports should not be primary docs. | Documentation/MLOps | mixed documentation/generated reports | Archive or move run-scoped reports to artifacts/pipeline/runs/<run_id>/reports/ after review. |
| docs/specs | QueryIR contract specifications. | Renderer, validator, migration, and tests need stable IR semantics. | Architecture | documentation | Keep versioned specs that match active QueryIR code. |
| evaluation | Reusable evaluation code, thresholds, and golden cases. | Quality gates and regression reports depend on stable evaluation inputs. | ML/Test | mixed source/test data | Keep source and checked-in fixtures; generated reports stay ignored. |
| evaluation/fixtures | Small controlled execution fixtures. | Execution-aware evaluation needs stable SQL/database cases. | Test | test data | Keep while tests or controlled evaluation reference them. |
| execution | Read-only SQL execution boundary. | Runtime must execute only validated, safe SELECT statements. | Security/Runtime | source | Keep as canonical execution boundary. |
| execution_eval | SQL canonicalization, structural comparison, and execution matching. | Evaluation needs reusable semantic/structural comparison utilities. | Evaluation | source | Keep while execution-aware tests and reports depend on it. |
| feedback | Reviewed feedback models, storage, and conversion workflows. | Self-training and governance use feedback only through typed flows. | ML/Data | source | Keep active feedback contracts; generated feedback JSONL remains ignored. |
| generic_planner | Schema-safe deterministic planner for direct simple queries. | Simple connected-database requests bypass model routing safely. | Runtime | source | Keep while runtime direct planning and generic join policy tests pass through it. |
| inference | Runtime prediction orchestration, confidence, telemetry, grounding, and slot resolution. | The app and smoke tests need a single runtime prediction path. | Runtime | source | Keep canonical runtime modules; remove hidden schema rules only after regression coverage. |
| inference/grounding | Schema and literal grounding services. | Connected databases need schema-specific projection/filter value grounding. | Runtime | source | Keep while slot resolver and grounding tests import these modules. |
| ir | QueryIR models, SQL conversion, migration, validation, and rendering. | QueryIR is the deterministic contract between models and executable SQL. | Architecture/Runtime | source | Keep canonical QueryIR v2 modules; compatibility code needs explicit removal conditions. |
| ir/query_ir_v2_rendering | QueryIR v2 renderer internals. | SQL generation is split into query, predicate, and expression rendering. | Runtime | source | Keep as canonical renderer implementation. |
| model_bundle | Model bundle build, manifest, validation, loading, and promotion logic. | Runtime loads artifacts only through validated bundle manifests. | MLOps/Runtime | source | Keep canonical bundle lifecycle code; never delete active bundle evidence automatically. |
| model_registry | Model artifact registry and manifest versioning helpers. | Training/promotion need structured artifact identity. | MLOps | source | Keep while quality gates and model selection tests import it. |
| model_selection | Champion/challenger and promotion selection logic. | Release readiness depends on controlled model candidate comparison. | MLOps | source | Keep while promotion governance tests use these policies. |
| models | Ignored local trained model outputs with a tracked placeholder. | Keeps the artifact directory available without committing model binaries. | MLOps | generated placeholder | Track only .gitkeep unless a specific lightweight artifact is approved. |
| neural_ir | Neural QueryIR architecture, labels, tokenizer, calibration, and prediction utilities. | The neural model path and related tests depend on these contracts. | ML | source | Keep active neural components; legacy experiments require review before deletion. |
| neural_optimization | Optimizers, schedulers, checkpoints, ranker, and training diagnostics. | Training wrappers share neural optimization infrastructure. | ML | source | Keep while training and neural tests import these helpers. |
| nl2sql_v1 | Legacy v1 NL-to-SQL implementation and compatibility reference. | Migration and legacy tests still compare or validate older behavior. | Architecture/Test | legacy source | Do not delete until legacy tests and migration docs are retired. |
| orchestration | Pipeline configuration, state, contract validation, step execution, and reporting. | Integrated training depends on one auditable pipeline runner. | MLOps | source | Keep canonical pipeline orchestration and fail unknown steps. |
| pipeline_configs | Pipeline-level config presets. | Supports smoke and full generic training orchestration. | MLOps | configuration | Keep configs consumed by orchestration or migrate into configs/ with compatibility proof. |
| quality_gates | Model quality, release, threshold, and regression-gate code. | Promotion must fail closed when safety or quality evidence is missing. | MLOps/Test | source | Keep while bundle validation and release readiness use these gates. |
| retrieval | Retrieval indexes, RAG retriever, reranker, schema indexes, and artifact compatibility. | Runtime/training use retrieval artifacts and metadata policy. | Runtime/ML | source | Keep canonical retrieval infrastructure; distinguish from retriever/ runtime model wrapper. |
| retriever | Runtime retrieval NL-to-SQL model wrapper. | The app and tests still import RetrievalNL2SQLModel from this package. | Runtime | source | Keep as active runtime wrapper unless imports migrate to retrieval/. |
| reward | Reward features, scoring, and candidate reranking helpers. | Self-training and candidate selection use reward signals. | ML | source | Keep while training/self-training paths reference it. |
| scripts | Supported operational, audit, dataset, and smoke commands. | Developers need stable command entry points outside package internals. | DevOps/MLOps | source | Keep supported scripts with purpose and usage docs; delete one-off scripts after proof. |
| self_training | Self-training loops, candidate generation, correction, and improvement tracking. | Provides governed feedback/improvement workflows. | ML | source/configuration | Keep while readiness audits and tests cover self-training. |
| semantic_layer | Schema profiling, semantic profiles, glossary, metrics, and dimensions. | Connected databases need schema-derived semantic metadata. | Runtime/Data | source | Keep while runtime schema mapping and semantic tests use it. |
| tests | Active unit, integration, regression, safety, and legacy tests. | Cleanup is safe only when behavior remains covered. | Test | test source/data | Keep active tests; classify legacy tests individually before removal. |
| tests/legacy | Legacy compatibility and research-path regression tests. | They guard migrations and older APIs that may still be referenced. | Test | legacy test source | Review for update/archive/delete; do not delete just because the folder says legacy. |
| training | Canonical training, evaluation, promotion, and report commands. | Integrated model production starts from training/train_model.py. | ML/MLOps | source | Keep supported entry points; consolidate old wrappers only with command/doc/test updates. |
| training_data | Small checked-in training examples and generated local IR corpora. | Examples seed tests; generated JSONL corpora are reproducible and ignored. | Data/ML | mixed source/generated | Track examples.jsonl and stats only when intentionally curated; generated JSONL stays ignored. |
| training_ir | Legacy/experimental QueryIR training and calibration commands. | Some ablation and legacy tests still exercise these paths. | ML | legacy/experimental source | Review for consolidation into training/ after proving command replacement. |
| validation | Central SQL validation package. | Execution safety requires a shared SELECT-only validator. | Security | source | Keep as canonical SQL validation boundary. |

## Entry Points

| Name | Command | Main module | Input configuration | Generated artifacts | Failure behaviour | Production relevance |
| --- | --- | --- | --- | --- | --- | --- |
| Streamlit application | streamlit run app/streamlit_app.py | app/streamlit_app.py | NL2SQL_ENV, NL2SQL_ALLOW_CANDIDATE_BUNDLE, app sidebar database settings | Runtime telemetry/prediction cache when configured | Fails closed when production bundle is missing or invalid | primary runtime UI |
| Production training pipeline | python training/train_model.py --config configs/training.yaml | training/train_model.py | configs/training.yaml | artifacts/pipeline/runs/<run_id>, artifacts/model_bundle/candidates/<run_id>, optional current bundle | Pipeline step failures block promotion | canonical production training command |
| Smoke training pipeline | python training/train_model.py --config configs/smoke_training.yaml | training/train_model.py | configs/smoke_training.yaml | smoke-scoped pipeline and candidate artifacts | Fast integration failure signal | developer validation only |
| Baseline training pipeline | python training/train_model.py --config configs/baseline_training.yaml | training/train_model.py | configs/baseline_training.yaml | baseline pipeline reports and candidate bundle | Full diagnostics without all production promotion requirements | release evidence, not production promotion by itself |
| Dataset verification | python scripts/verify_datasets.py | scripts/verify_datasets.py | data/raw and data/processed dataset paths | none expected | Reports missing/unusable datasets | training readiness |
| Dataset download | python scripts/download_datasets.py --datasets wikisql spider bird-mini | scripts/download_datasets.py | dataset arguments and local data/ paths | data/raw/ and data/processed/ | Stops on unavailable downloads or invalid destinations | data preparation |
| BIRD full preparation | python scripts/prepare_bird_full.py | scripts/prepare_bird_full.py | data/raw/bird/full | prepared BIRD full manifest and normalized data | Fails when source files are missing or malformed | large dataset preparation |
| Golden tests | python scripts/run_golden_tests.py | scripts/run_golden_tests.py | evaluation/golden_tests.jsonl | evaluation/golden_test_results.json | Nonzero on failed golden cases | regression validation |
| Integration readiness audit | python scripts/audit_integration_readiness.py | scripts/audit_integration_readiness.py | repo source, configs, docs, and artifacts | artifacts/audit/integration_readiness_report.* | Nonzero when required integration evidence is missing | release gate |
| Execution pipeline audit | python scripts/audit_execution_pipeline_readiness.py | scripts/audit_execution_pipeline_readiness.py | execution/runtime/evaluation modules | artifacts/audit/execution_pipeline_readiness_report.* | Nonzero on missing execution safety evidence | release gate |
| Generic NL-to-SQL audit | python scripts/audit_generic_nl2sql_readiness.py | scripts/audit_generic_nl2sql_readiness.py | runtime/training generic NL-to-SQL paths | artifacts/audit/generic_nl2sql_readiness_report.* | Nonzero on genericity gaps | release gate |
| Self-training audit | python scripts/audit_self_training_readiness.py | scripts/audit_self_training_readiness.py | self_training and feedback modules | artifacts/audit/self_training_readiness_report.* | Nonzero on missing governance controls | self-training gate |
| Test suite | python -m pytest tests/ --tb=short | tests/ | pytest.ini | .pytest_cache/ and coverage outputs when enabled | Nonzero on test failure | required regression gate |

## Repository Statistics

| Metric | Value |
| --- | --- |
| tracked files | 621 |
| untracked files | 60 |
| ignored summary entries | 54 |
| Python files | 552 |
| Markdown files | 21 |
| test files | 204 |
| configuration files | 15 |
| generated files in Git | 7 |

## Cleanup Rule

Retain production-critical artifacts and frozen splits unless a later manifest proves replacement, references, and reproducibility. Delete only generated caches/logs automatically.
