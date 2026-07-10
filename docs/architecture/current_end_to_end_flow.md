# Current End-to-End Flow

Generated for the semantic hardening pass on 2026-07-09.

## Training

Canonical entry point: `training/train_model.py`.

Flow:

1. Load `configs/training.yaml` and resolve the canonical neural config through `training/config_loader.py`.
2. Create or resume `pipeline_run_id`.
3. Write run-scoped `artifacts/pipeline/runs/<pipeline_run_id>/effective_config.yaml`.
4. Run `orchestration/PipelineRunner` with run-scoped `pipeline_state.json`.
5. `StepRunner` executes dataset verification, corpus build, retrieval index build, hard negatives, neural training, evaluation, quality gate, bundle build, validation, and optional promotion.
6. Candidate bundle output is run-scoped under `artifacts/model_bundle/candidates/<pipeline_run_id>`.

Canonical owners:

- Config resolution: `training/config_loader.py`
- Step registry: `orchestration/pipeline_config.py`
- Step execution: `orchestration/step_runner.py`
- Run state: `orchestration/pipeline_state.py`
- Dataset splits: `dataset_training/split_manager.py`
- Leakage checks: `dataset_training/leakage_checker.py`
- Corpus build: `dataset_training/ir_corpus_builder.py`
- Neural training: `training/train_neural_ir_optimized.py`
- Bundle build/load/validate/promote: `model_bundle/*`

## Runtime

Canonical runtime owner: `inference/prediction_orchestrator.py`.

Flow:

1. Application loads the canonical current bundle with `ModelBundleLoader`.
2. Bundle manifest supplies artifact paths and routing policy.
3. `PredictionOrchestrator` chooses direct, retrieval, or neural route.
4. Slot resolution and QueryIR validation occur before SQL rendering.
5. SQL validation and execution safety are enforced before database execution.

Production loader rule: `ModelBundleLoader.load_current(..., runtime_mode="production")` loads only `artifacts/model_bundle/current` and does not mutate `NL2SQL_ENV`.

## Artifact Lifecycle

Canonical lifecycle:

1. Work artifacts: `artifacts/work/*`
2. Run candidate: `artifacts/model_bundle/candidates/<pipeline_run_id>`
3. Validated candidate
4. Promotion decision
5. Current bundle: `artifacts/model_bundle/current`

The old singleton candidate path is no longer used as an active default.

## Dataset Lifecycle

Canonical lifecycle:

1. Source dataset example
2. Source split lineage preserved in row fields
3. Internal split assigned by `DatasetSplitManager`
4. Augmentations inherit parent lineage
5. Leakage checker blocks strict failures
6. Training consumes only eligible train rows

Active placeholder manifest `data/splits/semantic_v1/split_manifest.json` was removed; the placeholder copy lives under `tests/fixtures/splits/`.
