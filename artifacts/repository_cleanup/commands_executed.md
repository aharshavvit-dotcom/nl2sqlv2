# Commands Executed

Generated: 2026-07-15

This file records the material commands used for the cleanup pass. Read-only inspection commands such as `rg`, `Get-Content`, and `Get-ChildItem` were also used to inspect local files and generated artifacts.

## Baseline and branch setup

```powershell
git status --short
git rev-parse HEAD
git -c safe.directory=D:/nl2sqlv2 log --oneline --decorate -10
git diff --stat
git -c safe.directory=D:/nl2sqlv2 ls-files --others --exclude-standard
git -c safe.directory=D:/nl2sqlv2 status --short --ignored
git -c safe.directory=D:/nl2sqlv2 switch -c repository-cleanup/2026-07-15
```

## Inventory and cleanup reports

```powershell
python scripts/generate_repository_cleanup_inventory.py
python scripts/generate_repository_cleanup_inventory.py --delete-low-risk
```

## Validation

```powershell
python -m compileall .
python -m pytest tests/ --tb=short
python scripts/audit_execution_pipeline_readiness.py
python scripts/audit_generic_nl2sql_readiness.py
python scripts/audit_self_training_readiness.py
python scripts/audit_integration_readiness.py
python scripts/repo_cleanup_check.py
```

## Notes

The first run of `python scripts/audit_integration_readiness.py` reported a stale canonical neural configuration expectation. The cleanup pass updated both `scripts/audit_integration_readiness.py` and `scripts/repo_cleanup_check.py` to reflect the current checkpoint-selection policy: validation loss is the production selection metric, while semantic support score remains diagnostic.
