<!-- This file centralizes the fail-closed runtime contract and deployment commands. -->
# Deployment

The app serves a promoted model bundle and a user-connected SQLite or PostgreSQL database. It never auto-connects sample data.

Local development:

```bash
streamlit run app/streamlit_app.py
```

Explicit candidate debugging in PowerShell:

```powershell
$env:NL2SQL_ENV="development"
$env:NL2SQL_ALLOW_CANDIDATE_BUNDLE="1"
streamlit run app/streamlit_app.py
```

Production:

```powershell
$env:NL2SQL_ENV="production"
python scripts/audit_integration_readiness.py --production-health
streamlit run app/streamlit_app.py
```

Production requires `artifacts/model_bundle/current/bundle_manifest.json`. Startup rejects a missing or invalid manifest, a non-current bundle, a non-production quality gate, a failed gate, or `production_ready_full=false`. Candidate loading is forbidden in production even if its debug flag is set.

Environment variables:

| Variable | Purpose |
|---|---|
| `NL2SQL_ENV` | `development`, `staging`, or `production` |
| `NL2SQL_ALLOW_CANDIDATE_BUNDLE` | Explicit candidate access outside production (`1` enables) |
| `NL2SQL_MODEL_BUNDLE_DIR` | Override the default current bundle directory |
| `NL2SQL_DB_STATEMENT_TIMEOUT_SECONDS` | PostgreSQL statement timeout; default `30` |
| `NL2SQL_LOG_LEVEL` | Runtime logging level |

The production health check verifies the current bundle exists, its manifest is readable and production ready, the app entry point passes import preflight, and the central SQL validator accepts a bounded SELECT.

Training lifecycle:

- Debug/smoke uses the explicitly marked smoke neural config and creates a candidate only.
- Baseline uses `configs/neural_training_default.yaml` (10 epochs, batch size 8) as a production-like diagnostic and never promotes.
- Production uses the same canonical neural config, strict dataset and semantic gates, and promotes only a validated eligible candidate.
- Production application startup loads `current` only; candidate access is development/staging debug behavior.

After changing sklearn versions, old serialized retrieval/ranker objects must be rebuilt. The loader checks embedded version metadata, but these commands remove known stale work artifacts before retraining:

```powershell
Remove-Item -Recurse -Force artifacts\work\retrieval_ir -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force artifacts\work\adaptive_ranker -ErrorAction SilentlyContinue
```
