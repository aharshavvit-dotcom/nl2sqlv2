# Validation Results

Generated: 2026-07-15

## Final validation commands

| Command | Result | Notes |
| --- | --- | --- |
| `python -m compileall .` | PASS | Completed successfully before the final audit-policy patch; patched audit scripts were executed afterward. |
| `python -m pytest tests/ --tb=short` | PASS | `973 passed, 1 skipped, 1 warning in 148.57s`. |
| `python scripts/audit_execution_pipeline_readiness.py` | PASS | `5/5` checks passed. |
| `python scripts/audit_generic_nl2sql_readiness.py` | PASS | `13/13` checks passed. |
| `python scripts/audit_self_training_readiness.py` | PASS | `6/6` checks passed. |
| `python scripts/audit_integration_readiness.py` | PASS | `24/24` checks passed after aligning the canonical neural checkpoint expectation to validation loss. |
| `python scripts/repo_cleanup_check.py` | PASS | `42/42` checks passed after the same checkpoint-policy alignment. |
| `python scripts/generate_repository_cleanup_inventory.py --delete-low-risk` | PASS | Refreshed cleanup reports and removed only verified generated cache/log paths. |

## Warning observed

`tests/test_134_database_integration.py::test_postgres_read_only_and_timeout_enforced` emitted a pandas DB-API connection warning from `db/postgres_connector.py:173`. This was pre-existing behavior surfaced by the integration test and did not fail the suite.

## Runtime smoke coverage

Separate app-start, API-start, inference, and bundle-load smoke commands were not run in this pass. The audit suite and pytest suite exercised the execution pipeline and integration readiness, but the local `artifacts/model_bundle/current` production bundle path is absent, so a production bundle-load smoke should be run after the canonical bundle is restored or promoted.
