# File Change List

Generated: 2026-07-15

## Cleanup-owned source changes

| Path | Change |
| --- | --- |
| `.gitignore` | Allowed `artifacts/repository_cleanup/` to be tracked while keeping other generated artifacts ignored. |
| `scripts/generate_repository_cleanup_inventory.py` | Added a reproducible repository inventory, report-generation, and low-risk cache/log cleanup tool. |
| `scripts/audit_integration_readiness.py` | Updated the canonical neural checkpoint expectation from semantic support score maximization to validation loss minimization. |
| `scripts/repo_cleanup_check.py` | Updated the same checkpoint-policy expectation used by the repository cleanup gate. |
| `docs/REPOSITORY_MAP.md` | Added the generated canonical map of active folders, entry points, data/artifact ownership, and cleanup boundaries. |
| `artifacts/repository_cleanup/*` | Added generated cleanup manifests, inventory files, validation notes, and final cleanup reports. |

## Low-risk generated files removed

Only generated cache/log paths were deleted through `python scripts/generate_repository_cleanup_inventory.py --delete-low-risk`. The final `artifacts/repository_cleanup/deletion_manifest.json` records the generated cache/log paths removed after validation.

No source modules, tests, datasets, model bundles, candidate bundles, checkpoints, frozen split files, raw data, or documentation sources were deleted in this pass.

## Pre-existing working tree changes

The repository already had modified and untracked files before this cleanup branch was created. Those files were treated as user-owned work and were not reverted.
