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

        ordered_dbs = self._stratified_database_order(regular_dbs, grouped)
        train_count, validation_count = self._database_split_counts(len(ordered_dbs))
        train_dbs = set(ordered_dbs[:train_count])
        validation_dbs = set(ordered_dbs[train_count:train_count + validation_count])
        test_dbs = set(ordered_dbs[train_count + validation_count:])
        splits = {
            "train": [row for db_id in ordered_dbs if db_id in train_dbs for row in grouped[db_id]],
            "validation": [row for db_id in ordered_dbs if db_id in validation_dbs for row in grouped[db_id]],
            "test": [row for db_id in ordered_dbs if db_id in test_dbs for row in grouped[db_id]],
            "unseen_db_test": [row for db_id in unseen_dbs for row in grouped[db_id]],
            "unsupported": unsupported,
        }
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
            **{
                name: {
                    "by_dataset": _distribution(rows, lambda row: row.get("dataset_name") or "unknown"),
                    "by_intent": _distribution(rows, lambda row: row.get("intent") or (row.get("query_ir") or {}).get("intent") or "unknown"),
                    "by_complexity": _distribution(rows, lambda row: row.get("complexity") or "unknown"),
                    "by_join_count": _distribution(rows, lambda row: len((row.get("query_ir") or {}).get("joins") or [])),
                    "by_aggregation_type": _distribution(rows, _aggregation_type),
                }
                for name, rows in splits.items()
            },
        }
        write_json(Path(output_path), report)
        target = Path(output_path)
        lines = ["# Split Distribution Report", ""]
        for name in SPLIT_NAMES:
            lines.extend([f"## {name}", "", f"- examples: {len(splits.get(name, []))}", f"- databases: {report['database_counts'].get(name, 0)}", f"- intents: {report.get(name, {}).get('by_intent', {})}", f"- complexity: {report.get(name, {}).get('by_complexity', {})}", ""])
        target.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _with_split(row: dict[str, Any], split: str) -> dict[str, Any]:
        updated = dict(row)
        updated["split"] = split
        if isinstance(updated.get("query_ir"), dict):
            updated["query_ir"] = dict(updated["query_ir"])
            updated["query_ir"].setdefault("metadata", {})["split"] = split
        return updated

    def _database_split_counts(self, count: int) -> tuple[int, int]:
        if count <= 1:
            return count, 0
        train = max(1, int(round(count * self.train_ratio)))
        test = max(1, int(round(count * self.test_ratio)))
        validation = max(1, count - train - test) if count >= 3 else 0
        while train + validation + test > count and train > 1:
            train -= 1
        while train + validation + test > count and validation > 0:
            validation -= 1
        return train, validation

    def _stratified_database_order(
        self,
        db_ids: list[str],
        grouped: dict[str, list[dict[str, Any]]],
    ) -> list[str]:
        strata: dict[tuple[Any, ...], list[str]] = defaultdict(list)
        for db_id in db_ids:
            rows = grouped[db_id]
            signature = (
                _dominant(rows, lambda row: row.get("dataset_name") or "unknown"),
                _dominant(rows, lambda row: row.get("intent") or (row.get("query_ir") or {}).get("intent") or "unknown"),
                _dominant(rows, lambda row: row.get("complexity") or "unknown"),
                _dominant(rows, lambda row: len((row.get("query_ir") or {}).get("joins") or [])),
                _dominant(rows, _aggregation_type),
            )
            strata[signature].append(db_id)
        rng = random.Random(self.seed + 1)
        for values in strata.values():
            rng.shuffle(values)
        ordered: list[str] = []
        queues = [values for _, values in sorted(strata.items(), key=lambda item: str(item[0]))]
        while any(queues):
            for values in queues:
                if values:
                    ordered.append(values.pop())
        return ordered


def _distribution(rows: list[dict[str, Any]], key: Any) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(key(row))] += 1
    return dict(sorted(counts.items()))


def _dominant(rows: list[dict[str, Any]], key: Any) -> str:
    counts = _distribution(rows, key)
    return max(counts, key=counts.get) if counts else "unknown"


def _aggregation_type(row: dict[str, Any]) -> str:
    metrics = (row.get("query_ir") or {}).get("metrics") or []
    aggregations = sorted({str(item.get("aggregation") or "none") for item in metrics if isinstance(item, dict)})
    return "+".join(aggregations) if aggregations else "none"
