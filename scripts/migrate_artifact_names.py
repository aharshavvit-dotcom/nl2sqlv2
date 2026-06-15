"""Migrate artifact folder names from legacy to new naming convention.

This script copies or creates symlinks from old artifact folder names
to new canonical names. It NEVER deletes old folders.

Usage:
    python scripts/migrate_artifact_names.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts"

MIGRATIONS = [
    ("option_c_model", "retrieval_ir_model"),
    ("option_a_ir_model", "neural_ir_model"),
    ("option_a_ir_model_v2", "neural_ir_model"),
]


def migrate() -> list[str]:
    """Run artifact folder migration. Returns a list of actions taken."""
    actions: list[str] = []
    for old_name, new_name in MIGRATIONS:
        old_path = ARTIFACT_ROOT / old_name
        new_path = ARTIFACT_ROOT / new_name

        if not old_path.exists():
            actions.append(f"SKIP  {old_name} → {new_name} (source does not exist)")
            continue
        if new_path.exists():
            actions.append(f"SKIP  {old_name} → {new_name} (target already exists)")
            continue

        try:
            shutil.copytree(old_path, new_path)
            actions.append(f"COPY  {old_name} → {new_name}")
        except Exception as exc:
            actions.append(f"ERROR {old_name} → {new_name}: {exc}")

    return actions


def main() -> None:
    print("Artifact folder migration")
    print("=" * 60)
    actions = migrate()
    for action in actions:
        print(f"  {action}")
    print("=" * 60)
    print(f"Done. {len(actions)} items processed.")
    print("Old folders were NOT deleted.")


if __name__ == "__main__":
    main()
