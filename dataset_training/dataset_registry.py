from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import normalize_dataset_name


SUPPORTED_DATASETS = ("wikisql", "spider", "bird-mini", "bird-full")


class DatasetRegistry:
    def __init__(self, root_dir: str = "data/raw"):
        self.root_dir = Path(root_dir)

    def discover(self) -> dict[str, dict[str, Any]]:
        return {name: self._dataset_report(name) for name in SUPPORTED_DATASETS}

    def get_available_datasets(self) -> list[str]:
        return [name for name, report in self.discover().items() if report["available"]]

    def get_dataset_paths(self, dataset_name: str) -> dict[str, str]:
        report = self._dataset_report(dataset_name)
        return report["paths"]

    def validate_dataset_presence(self, dataset_names: list[str]) -> dict[str, dict[str, Any]]:
        discovered = self.discover()
        return {
            normalize_dataset_name(name): discovered.get(normalize_dataset_name(name), self._missing_report(normalize_dataset_name(name)))
            for name in dataset_names
        }

    def _dataset_report(self, dataset_name: str) -> dict[str, Any]:
        name = normalize_dataset_name(dataset_name)
        candidates = self._candidate_dirs(name)
        base = next((path for path in candidates if path.exists()), candidates[0])
        expected = self._expected_files(name)
        missing = [item for item in expected if not (base / item).exists()]
        available = base.exists() and (not expected or len(missing) < len(expected))
        paths = {"root": str(base)}
        for item in expected:
            path = base / item
            if path.exists():
                paths[item.rstrip("/").replace("/", "_")] = str(path)
        return {"available": available, "paths": paths if available else {}, "missing_files": missing}

    def _candidate_dirs(self, name: str) -> list[Path]:
        if name == "wikisql":
            return [self.root_dir / "wikisql"]
        if name == "spider":
            return [self.root_dir / "spider", self.root_dir / "spider" / "spider_data"]
        if name == "bird-mini":
            return [
                self.root_dir / "bird" / "mini",
                self.root_dir / "bird" / "mini_dev",
                self.root_dir / "bird" / "minidev" / "MINIDEV",
                self.root_dir / "bird" / "mini_dev_hf",
            ]
        if name == "bird-full":
            return [self.root_dir / "bird" / "full"]
        return [self.root_dir / name]

    @staticmethod
    def _expected_files(name: str) -> list[str]:
        if name == "wikisql":
            return ["train.jsonl", "dev.jsonl", "test.jsonl"]
        if name == "spider":
            return ["train_spider.json", "dev.json", "tables.json"]
        if name == "bird-mini":
            return ["mini_dev_sqlite.json", "dev_tables.json"]
        if name == "bird-full":
            return ["train.json", "validation.json", "test.json"]
        return []

    @staticmethod
    def _missing_report(name: str) -> dict[str, Any]:
        return {"available": False, "paths": {}, "missing_files": [f"unsupported dataset: {name}"]}
