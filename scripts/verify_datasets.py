from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.bird_adapter import _external_hf_datasets
from scripts.dataset_paths import (
    BIRD_FULL_DIR,
    BIRD_MINI_DEV_DIR,
    BIRD_MINI_DEV_HF_DIR,
    SPIDER_DIR,
    WIKISQL_DIR,
    ensure_dataset_dirs,
    resolve_bird_mini_dir,
)


@dataclass(frozen=True)
class DatasetVerification:
    dataset: str
    status: str
    files_found: int
    example_count: int | None
    notes: str

    @property
    def ready(self) -> bool:
        return self.status == "ready"


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def verify_wikisql(raw_dir: Path = WIKISQL_DIR) -> DatasetVerification:
    required = [
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "train.tables.jsonl",
        "dev.tables.jsonl",
        "test.tables.jsonl",
    ]
    found = sum(1 for item in required if (raw_dir / item).exists())
    count = sum(count_jsonl(raw_dir / item) for item in ["train.jsonl", "dev.jsonl", "test.jsonl"])
    status = "ready" if found == len(required) else "missing"
    return DatasetVerification("WikiSQL", status, found, count, str(raw_dir))


def verify_spider(raw_dir: Path = SPIDER_DIR) -> DatasetVerification:
    required = ["train_spider.json", "dev.json", "tables.json"]
    found = sum(1 for item in required if (raw_dir / item).exists())
    has_db_dir = (raw_dir / "database").exists() or (raw_dir / "databases").exists()
    count = 0
    for item in ["train_spider.json", "train_others.json", "dev.json"]:
        path = raw_dir / item
        if path.exists():
            try:
                count += len(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
    status = "ready" if found == len(required) and has_db_dir else "missing"
    notes = str(raw_dir) if has_db_dir else "database/ or databases/ folder missing"
    return DatasetVerification("Spider", status, found + int(has_db_dir), count, notes)


def verify_bird_mini(raw_dir: Path | None = None) -> DatasetVerification:
    raw_dir = raw_dir or resolve_bird_mini_dir()
    real_files = [path for path in raw_dir.iterdir() if path.name != ".gitkeep"] if raw_dir.exists() else []
    if not raw_dir.exists() or not real_files:
        return DatasetVerification("BIRD Mini-Dev", "missing", 0, None, str(raw_dir))
    sqlite_json = raw_dir / "mini_dev_sqlite.json"
    tables_json = raw_dir / "dev_tables.json"
    db_dir = raw_dir / "dev_databases"
    if sqlite_json.exists() and tables_json.exists() and db_dir.exists():
        try:
            count = len(json.loads(sqlite_json.read_text(encoding="utf-8")))
        except Exception:
            count = None
        files_found = sum(1 for item in [sqlite_json, tables_json, db_dir] if item.exists())
        return DatasetVerification("BIRD Mini-Dev", "ready", files_found, count, str(raw_dir))

    try:
        with _external_hf_datasets() as hf_datasets:
            dataset = hf_datasets.load_from_disk(str(raw_dir))
        split_names = list(dataset.keys()) if hasattr(dataset, "keys") else ["dataset"]
        count = sum(len(dataset[name]) for name in split_names) if hasattr(dataset, "keys") else len(dataset)
        return DatasetVerification("BIRD Mini-Dev", "ready", len(split_names), count, ", ".join(split_names))
    except Exception as exc:
        local_json = list(raw_dir.glob("*.json")) + list(raw_dir.glob("*.jsonl"))
        if local_json:
            return DatasetVerification("BIRD Mini-Dev", "ready", len(local_json), None, "local JSON fallback")
        if raw_dir != BIRD_MINI_DEV_HF_DIR and BIRD_MINI_DEV_HF_DIR.exists():
            return verify_bird_mini(BIRD_MINI_DEV_HF_DIR)
        return DatasetVerification("BIRD Mini-Dev", "missing", 0, None, f"cannot load: {exc}")


def verify_bird_full(raw_dir: Path = BIRD_FULL_DIR) -> DatasetVerification:
    if not raw_dir.exists():
        return DatasetVerification("BIRD Full", "skipped", 0, None, "folder empty")
    files = [path for path in raw_dir.rglob("*") if path.is_file() and path.name != ".gitkeep"]
    if not files:
        return DatasetVerification("BIRD Full", "skipped", 0, None, "folder empty")
    zip_files = [path for path in files if path.suffix.lower() == ".zip"]
    invalid_zips = [path.name for path in zip_files if not zipfile.is_zipfile(path)]
    if invalid_zips:
        return DatasetVerification(
            "BIRD Full",
            "incomplete",
            len(files),
            None,
            f"partial/corrupt archive: {', '.join(invalid_zips)}",
        )
    return DatasetVerification("BIRD Full", "present", len(files), None, str(raw_dir))


def verify_all() -> list[DatasetVerification]:
    ensure_dataset_dirs()
    return [verify_wikisql(), verify_spider(), verify_bird_mini(), verify_bird_full()]


def print_table(rows: list[DatasetVerification]) -> None:
    headers = ["Dataset", "Status", "Files Found", "Example Count", "Notes"]
    values = [
        [row.dataset, row.status, str(row.files_found), "" if row.example_count is None else str(row.example_count), row.notes]
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(value[index]) for value in values))
        for index in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for value in values:
        print(" | ".join(value[index].ljust(widths[index]) for index in range(len(headers))))


def main() -> int:
    argparse.ArgumentParser().parse_args()
    rows = verify_all()
    print_table(rows)
    return 0 if any(row.ready for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
