# Test Suite Rationalization Commands

Generated: 2026-07-15

```powershell
python scripts/rationalize_test_suite.py
python scripts/rationalize_test_suite.py --apply
python -m pytest --collect-only -q tests --tb=short
python -m pytest tests/unit tests/integration/test_query_ir_v2_execution.py tests/integration/test_database_and_connected_regression.py --tb=short
python -m pytest -m "unit or contract or safety" --tb=short
python -m pytest -m "integration or regression" --tb=short
python -m pytest -m "training and not slow" --tb=short
python -m pytest tests/ --tb=short
python scripts/audit_execution_pipeline_readiness.py
python scripts/audit_generic_nl2sql_readiness.py
python scripts/audit_self_training_readiness.py
python scripts/audit_integration_readiness.py
python scripts/repo_cleanup_check.py
```
