from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .leakage_checker import DatasetLeakageChecker
from .utils import write_json


SPLIT_NAMES = ("train", "validation", "test", "unseen_db_test", "unsupported")


class DatasetSplitManager:
    def __init__(
        self,
        seed: int = 42,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        unseen_db_test_ratio: float = 0.15,
    ):
        self.seed = seed
        total = max(train_ratio + validation_ratio + test_ratio, 0.0001)
        self.train_ratio = train_ratio / total
        self.validation_ratio = validation_ratio / total
        self.test_ratio = test_ratio / total
        self.unseen_db_test_ratio = max(0.0, min(unseen_db_test_ratio, 0.8))

    def split_examples(self, examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        return self.split_by_database(examples)

    def split_by_database(self, examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        supported = [row for row in examples if not row.get("unsupported_reason") and row.get("query_ir") is not None]
        unsupported = [dict(row, split="unsupported") for row in examples if row not in supported]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in supported:
            grouped[str(row.get("db_id") or "__unknown_db__")].append(row)

        db_ids = sorted(grouped)
        random.Random(self.seed).shuffle(db_ids)
        unseen_count = 0
        if len(db_ids) >= 2 and self.unseen_db_test_ratio > 0:
            unseen_count = max(1, int(round(len(db_ids) * self.unseen_db_test_ratio)))
            unseen_count = min(unseen_count, len(db_ids) - 1)
        unseen_dbs = set(db_ids[:unseen_count])
        regular_dbs = db_ids[unseen_count:]

        regular_rows = [row for db_id in regular_dbs for row in grouped[db_id]]
        random.Random(self.seed + 1).shuffle(regular_rows)
        train_end = int(len(regular_rows) * self.train_ratio)
        validation_end = train_end + int(len(regular_rows) * self.validation_ratio)
        splits = {
            "train": regular_rows[:train_end],
            "validation": regular_rows[train_end:validation_end],
            "test": regular_rows[validation_end:],
            "unseen_db_test": [row for db_id in unseen_dbs for row in grouped[db_id]],
            "unsupported": unsupported,
        }
        if not splits["train"] and regular_rows:
            splits["train"] = regular_rows[:1]
            splits["test"] = regular_rows[1:]
        for name, rows in splits.items():
            splits[name] = [self._with_split(row, name) for row in rows]

        leakage = DatasetLeakageChecker().check_database_leakage(splits)
        if leakage["has_database_leakage"]:
            raise ValueError(f"Database leakage detected: {leakage['database_overlap']}")
        return splits

    def split_by_dataset_and_database(self, examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in examples:
            by_dataset[str(row.get("dataset_name") or "unknown")].append(row)
        merged = {name: [] for name in SPLIT_NAMES}
        for dataset, rows in sorted(by_dataset.items()):
            dataset_splits = self.split_by_database(rows)
            for split_name, split_rows in dataset_splits.items():
                merged[split_name].extend(split_rows)
        return merged

    def save_split_report(self, splits: dict[str, list[dict[str, Any]]], output_path: str) -> None:
        report = {
            "split_counts": {name: len(rows) for name, rows in splits.items()},
            "database_counts": {name: len({row.get("db_id") for row in rows}) for name, rows in splits.items()},
            "databases": {name: sorted({str(row.get("db_id")) for row in rows if row.get("db_id")}) for name, rows in splits.items()},
        }
        write_json(Path(output_path), report)

    @staticmethod
    def _with_split(row: dict[str, Any], split: str) -> dict[str, Any]:
        updated = dict(row)
        updated["split"] = split
        if isinstance(updated.get("query_ir"), dict):
            updated["query_ir"] = dict(updated["query_ir"])
            updated["query_ir"].setdefault("metadata", {})["split"] = split
        return updated
