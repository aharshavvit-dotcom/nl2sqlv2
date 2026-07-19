# Test Suite Validation Results

Generated: 2026-07-15

## Collection and lanes

| Command | Result |
| --- | --- |
| `python -m pytest --collect-only -q tests --tb=short` | PASS, 958 tests collected |
| `python -m pytest tests/unit tests/integration/test_query_ir_v2_execution.py tests/integration/test_database_and_connected_regression.py --tb=short` | PASS, 259 passed |
| `python -m pytest -m "unit or contract or safety" --tb=short` | PASS, 847 passed, 1 skipped, 110 deselected |
| `python -m pytest -m "integration or regression" --tb=short` | PASS, 88 passed, 870 deselected |
| `python -m pytest -m "training and not slow" --tb=short` | PASS, 106 passed, 1 skipped, 851 deselected |
| `python -m pytest tests/ --tb=short` | PASS, 957 passed, 1 skipped, 1 warning in 45.55s |

## Readiness audits

| Command | Result |
| --- | --- |
| `python scripts/audit_execution_pipeline_readiness.py` | PASS, 5/5 |
| `python scripts/audit_generic_nl2sql_readiness.py` | PASS, 13/13 |
| `python scripts/audit_self_training_readiness.py` | PASS, 6/6 |
| `python scripts/audit_integration_readiness.py` | PASS, 24/24 |
| `python scripts/repo_cleanup_check.py` | PASS, 44/44 |

## Coverage accounting

The merged source files contained 240 AST-counted test bodies, and the 13 canonical target modules contain the same 240 AST-counted test bodies. This verifies that the consolidation preserved the executable test bodies for the merged clusters.

Line coverage, branch coverage, mutation score, and per-file runtime were not measured in this pass.

Standalone runtime smoke and production bundle smoke were not run. The local
`artifacts/model_bundle/current` production bundle is still absent, so bundle
smoke remains a release follow-up.

## Warning observed

`tests/integration/test_database_and_connected_regression.py::test_postgres_read_only_and_timeout_enforced` emits the existing pandas DB-API connection warning from `db/postgres_connector.py:173`.
