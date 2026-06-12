from __future__ import annotations

from pathlib import Path

from scripts.dataset_paths import BIRD_FULL_DIR, SPIDER_DIR, WIKISQL_DIR, resolve_bird_mini_dir

from .bird_adapter import BirdAdapter
from .models import DatabaseSchema, Text2SQLExample
from .spider_adapter import SpiderAdapter
from .wikisql_adapter import WikiSQLAdapter


class DatasetLoader:
    def __init__(self, raw_root: Path | None = None):
        self.raw_root = raw_root

    def load(
        self,
        dataset_name: str,
        max_examples: int | None = None,
    ) -> tuple[list[Text2SQLExample], dict[str, DatabaseSchema]]:
        normalized = dataset_name.strip().lower().replace("_", "-")
        if normalized == "wikisql":
            adapter = WikiSQLAdapter()
            raw_dir = self._dataset_dir(WIKISQL_DIR, "wikisql")
            return adapter.load_examples(raw_dir), adapter.load_schemas(raw_dir)
        if normalized == "spider":
            adapter = SpiderAdapter()
            raw_dir = self._dataset_dir(SPIDER_DIR, "spider")
            return adapter.load_examples(raw_dir), adapter.load_schemas(raw_dir)
        if normalized in {"bird", "bird-mini", "bird-mini-dev"}:
            adapter = BirdAdapter()
            raw_dir = self._dataset_dir(resolve_bird_mini_dir(), "bird/mini_dev")
            return adapter.load_examples(raw_dir, split_type="mini-dev", max_examples=max_examples), adapter.load_schemas(raw_dir)
        if normalized == "bird-full":
            adapter = BirdAdapter()
            raw_dir = self._dataset_dir(BIRD_FULL_DIR, "bird/full")
            return adapter.load_examples(raw_dir, split_type="full", max_examples=max_examples), adapter.load_schemas(raw_dir)
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def load_datasets(
        self,
        dataset_names: list[str],
        max_examples: int | None = None,
    ) -> tuple[list[Text2SQLExample], dict[str, DatabaseSchema]]:
        all_examples: list[Text2SQLExample] = []
        all_schemas: dict[str, DatabaseSchema] = {}
        for dataset_name in dataset_names:
            examples, schemas = self.load(dataset_name, max_examples=max_examples)
            all_examples.extend(examples)
            all_schemas.update(schemas)
        return all_examples, all_schemas

    def _dataset_dir(self, default: Path, relative: str) -> Path:
        if self.raw_root is None:
            return default
        return self.raw_root / relative
