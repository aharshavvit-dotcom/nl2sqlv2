# Model Bundle Lifecycle

Canonical roots:

- Current: `artifacts/model_bundle/current`
- Candidates: `artifacts/model_bundle/candidates/<pipeline_run_id>`

Bundle creation is owned by `model_bundle/bundle_builder.py` and called through `orchestration/step_runner.py`.

Bundle loading is owned by `model_bundle/bundle_loader.py`.

Production policy:

- Load current only.
- No candidate fallback in production.
- No work-artifact fallback in production.
- Runtime mode is explicit and no longer mutates `NL2SQL_ENV`.

Promotion remains owned by `model_bundle/bundle_promoter.py`. Full recoverable transactional promotion, including lock, journal, temporary current, backup restore, and recovery validation, is not fully verified in this pass.
