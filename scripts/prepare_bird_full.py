from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dataset_paths import BIRD_FULL_DIR


def prepare_bird_full(
    raw_dir: Path = BIRD_FULL_DIR,
    test_ratio: float = 0.5,
    overwrite: bool = True,
) -> dict[str, Any]:
    raw_dir = raw_dir.resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"BIRD full directory not found: {raw_dir}")
    if raw_dir.name.lower() != "full" or "data" not in {part.lower() for part in raw_dir.parts}:
        raise ValueError(f"Refusing to prepare unexpected directory: {raw_dir}")

    train_json = first_existing(raw_dir / "train" / "train" / "train.json", raw_dir / "train.json")
    train_tables = first_existing(raw_dir / "train" / "train" / "train_tables.json", raw_dir / "train_tables.json")
    train_gold = first_existing(raw_dir / "train" / "train" / "train_gold.sql", raw_dir / "train_gold.sql", required=False)
    dev_json = first_existing(raw_dir / "dev_20240627" / "dev.json", raw_dir / "dev.json", raw_dir / "validation.json")
    dev_tables = first_existing(raw_dir / "dev_20240627" / "dev_tables.json", raw_dir / "dev_tables.json")
    dev_sql = first_existing(raw_dir / "dev_20240627" / "dev.sql", raw_dir / "dev.sql", required=False)

    train_rows = load_json_list(train_json)
    dev_rows = load_json_list(dev_json)
    validation_rows, test_rows, test_db_ids = split_dev_rows_by_db_id(dev_rows, test_ratio=test_ratio)

    write_json(raw_dir / "train.json", train_rows, overwrite=overwrite)
    write_json(raw_dir / "validation.json", validation_rows, overwrite=overwrite)
    write_json(raw_dir / "test.json", test_rows, overwrite=overwrite)
    copy_file(train_tables, raw_dir / "train_tables.json", overwrite=overwrite)
    copy_file(dev_tables, raw_dir / "dev_tables.json", overwrite=overwrite)
    if train_gold:
        copy_file(train_gold, raw_dir / "train_gold.sql", overwrite=overwrite)
    if dev_sql:
        copy_file(dev_sql, raw_dir / "dev.sql", overwrite=overwrite)

    manifest = {
        "raw_dir": str(raw_dir),
        "source_files": {
            "train_json": str(train_json),
            "dev_json": str(dev_json),
            "train_tables": str(train_tables),
            "dev_tables": str(dev_tables),
            "train_gold": str(train_gold) if train_gold else None,
            "dev_sql": str(dev_sql) if dev_sql else None,
        },
        "prepared_files": {
            "train": str(raw_dir / "train.json"),
            "validation": str(raw_dir / "validation.json"),
            "test": str(raw_dir / "test.json"),
            "train_tables": str(raw_dir / "train_tables.json"),
            "dev_tables": str(raw_dir / "dev_tables.json"),
        },
        "counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
            "dev_source": len(dev_rows),
        },
        "split_policy": {
            "train": "official BIRD full train split",
            "validation": "database-disjoint portion of official dev split",
            "test": "held-out database-disjoint portion of official dev split",
            "test_ratio_of_dev": test_ratio,
            "test_db_ids": test_db_ids,
        },
        "notes": [
            "Original downloaded folders and ZIP files are preserved.",
            "Database ZIPs do not need to be extracted for retrieval or QueryIR label generation.",
        ],
    }
    (raw_dir / "bird_full_prepared_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def first_existing(*paths: Path, required: bool = True) -> Path | None:
    for path in paths:
        if path.exists() and path.is_file() and not is_ignored_artifact(path):
            return path
    if required:
        joined = ", ".join(str(path) for path in paths)
        raise FileNotFoundError(f"None of these files exist: {joined}")
    return None


def is_ignored_artifact(path: Path) -> bool:
    return any(part == "__MACOSX" for part in path.parts) or path.name.startswith("._")


def load_json_list(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [dict(item) for item in raw]


def split_dev_rows_by_db_id(
    rows: list[dict[str, Any]],
    test_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if not rows:
        return [], [], []
    if not 0 < test_ratio < 1:
        raise ValueError("--test-ratio must be greater than 0 and less than 1")
    if len(rows) == 1:
        return rows, [], []

    groups: dict[str, list[dict[str, Any]]] = {}
    row_group_keys: list[str] = []
    for index, row in enumerate(rows):
        db_id = row_group_key(row, index)
        row_group_keys.append(db_id)
        groups.setdefault(db_id, []).append(row)

    target = min(max(int(round(len(rows) * test_ratio)), 1), len(rows) - 1)
    if len(groups) < 2:
        validation_rows = rows[:-target]
        test_rows = rows[-target:]
        return validation_rows, test_rows, sorted(groups)

    test_db_ids = choose_group_subset(groups, target)
    validation_rows = [row for row, db_id in zip(rows, row_group_keys) if db_id not in test_db_ids]
    test_rows = [row for row, db_id in zip(rows, row_group_keys) if db_id in test_db_ids]
    return validation_rows, test_rows, sorted(test_db_ids)


def row_group_key(row: dict[str, Any], index: int) -> str:
    return str(row.get("db_id") or row.get("database_id") or f"__row_{index}")


def choose_group_subset(groups: dict[str, list[dict[str, Any]]], target: int) -> set[str]:
    choices: dict[int, tuple[str, ...]] = {0: ()}
    total_rows = sum(len(rows) for rows in groups.values())
    for db_id, db_rows in groups.items():
        count = len(db_rows)
        additions = {
            total + count: selected + (db_id,)
            for total, selected in choices.items()
        }
        for total, selected in additions.items():
            if total not in choices or selected < choices[total]:
                choices[total] = selected

    non_empty_choices = {
        total: selected
        for total, selected in choices.items()
        if selected and total < total_rows
    }
    best_total, best_selected = min(
        non_empty_choices.items(),
        key=lambda item: (abs(item[0] - target), item[0] > target, len(item[1]), item[1]),
    )
    return set(best_selected)


def write_json(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def copy_file(source: Path, target: Path, overwrite: bool) -> None:
    if source.resolve() == target.resolve():
        return
    if target.exists() and not overwrite:
        return
    tmp = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, tmp)
    tmp.replace(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare manually downloaded BIRD full files for local training.")
    parser.add_argument("--raw-dir", type=Path, default=BIRD_FULL_DIR)
    parser.add_argument("--test-ratio", type=float, default=0.5, help="Fraction of official dev rows to reserve as test.")
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare_bird_full(raw_dir=args.raw_dir, test_ratio=args.test_ratio, overwrite=not args.no_overwrite)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
