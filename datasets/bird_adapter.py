from __future__ import annotations

import importlib
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import DatabaseSchema, Text2SQLExample
from .schema_normalizer import SchemaNormalizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BirdAdapter:
    dataset_name = "bird"

    def load_examples(
        self,
        raw_dir: Path,
        split_type: str = "mini-dev",
        max_examples: int | None = None,
    ) -> list[Text2SQLExample]:
        if self._looks_like_hf_saved_dataset(raw_dir):
            return self._load_hf_mini_dev(raw_dir, max_examples=max_examples)
        return self._load_local_json(raw_dir, split_type=split_type, max_examples=max_examples)

    def load_schemas(self, raw_dir: Path) -> dict[str, DatabaseSchema]:
        schemas: dict[str, DatabaseSchema] = {}
        for tables_path in self._table_files(raw_dir):
            try:
                records = json.loads(tables_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict) or not record.get("db_id"):
                    continue
                schema = self._schema_from_table_record(record, raw_dir)
                schemas[schema.db_id] = schema

        for sqlite_path in raw_dir.rglob("*.sqlite"):
            db_id = sqlite_path.stem
            if db_id in schemas and not schemas[db_id].db_path:
                schemas[db_id].db_path = str(sqlite_path)
                continue
            if db_id in schemas:
                continue
            schema = DatabaseSchema(
                db_id=db_id,
                dataset_name=self.dataset_name,
                db_path=str(sqlite_path),
                tables={},
                foreign_keys=[],
                primary_keys=[],
            )
            schema.serialized_schema = SchemaNormalizer.serialize_schema(schema)
            schemas[db_id] = schema
        return schemas

    def _load_hf_mini_dev(self, raw_dir: Path, max_examples: int | None) -> list[Text2SQLExample]:
        try:
            with _external_hf_datasets() as hf_datasets:
                dataset = hf_datasets.load_from_disk(str(raw_dir))
        except Exception:
            return self._load_local_json(raw_dir, split_type="mini-dev", max_examples=max_examples)

        split_names = self._split_names(dataset)
        preferred = self._choose_sqlite_split(split_names)
        examples: list[Text2SQLExample] = []
        for split in preferred:
            rows = dataset[split] if split_names else dataset
            for row in rows:
                examples.append(
                    self._normalize_row(
                        dict(row),
                        split=self._canonical_split(split),
                        dataset_name="bird-mini",
                    )
                )
                if max_examples and len(examples) >= max_examples:
                    return examples
        return examples

    def _load_local_json(
        self,
        raw_dir: Path,
        split_type: str,
        max_examples: int | None,
    ) -> list[Text2SQLExample]:
        examples: list[Text2SQLExample] = []
        schemas = self.load_schemas(raw_dir)
        for path in self._example_files(raw_dir, split_type=split_type):
            rows = self._read_rows(path)
            split = self._canonical_split(path.stem)
            dataset_name = "bird-mini" if split_type in {"mini-dev", "bird-mini"} else "bird-full"
            for row in rows:
                example = self._normalize_row(
                    row,
                    split=split,
                    dataset_name=dataset_name,
                    source_file=str(path.relative_to(raw_dir)),
                )
                schema = schemas.get(example.db_id)
                if schema:
                    example.db_path = schema.db_path
                    example.schema = schema.to_dict()
                    example.tables = list(schema.tables)
                    example.columns = self._columns_from_schema(schema)
                examples.append(example)
                if max_examples and len(examples) >= max_examples:
                    return examples
        return examples

    def _normalize_row(
        self,
        row: dict[str, Any],
        split: str,
        dataset_name: str,
        source_file: str | None = None,
    ) -> Text2SQLExample:
        question = row.get("question") or row.get("Question") or row.get("utterance") or ""
        sql = row.get("SQL") or row.get("query") or row.get("sql") or ""
        db_id = row.get("db_id") or row.get("database_id") or row.get("db_name") or ""
        difficulty = row.get("difficulty") or row.get("difficulty_level")
        suffix = row.get("question_id") or row.get("id") or _stable_suffix(question, sql, db_id)
        return Text2SQLExample(
            example_id=f"{dataset_name}:{split}:{suffix}",
            dataset_name=dataset_name,
            db_id=str(db_id),
            question=str(question),
            sql=str(sql),
            split=split,
            difficulty=str(difficulty) if difficulty is not None else None,
            source_file=source_file,
        )

    @staticmethod
    def _looks_like_hf_saved_dataset(raw_dir: Path) -> bool:
        return (raw_dir / "dataset_dict.json").exists() or (raw_dir / "state.json").exists()

    @staticmethod
    def _split_names(dataset: Any) -> list[str]:
        if hasattr(dataset, "keys"):
            try:
                return [str(key) for key in dataset.keys()]
            except Exception:
                return []
        return []

    @staticmethod
    def _choose_sqlite_split(split_names: list[str]) -> list[str]:
        if not split_names:
            return [""]
        sqlite_like = [name for name in split_names if "sqlite" in name.lower()]
        return sqlite_like or split_names

    @staticmethod
    def _canonical_split(name: str) -> str:
        lowered = name.lower()
        if "train" in lowered:
            return "train"
        if "test" in lowered:
            return "test"
        if "dev" in lowered or "valid" in lowered:
            return "validation"
        return lowered or "validation"

    @staticmethod
    def _table_files(raw_dir: Path) -> list[Path]:
        preferred = [
            raw_dir / "train_tables.json",
            raw_dir / "dev_tables.json",
            raw_dir / "validation_tables.json",
            raw_dir / "test_tables.json",
            raw_dir / "tables.json",
        ]
        prepared = [path for path in preferred if path.exists() and not BirdAdapter._skip_path(path)]
        if prepared:
            return prepared
        candidates = []
        for path in raw_dir.rglob("*.json"):
            if BirdAdapter._skip_path(path):
                continue
            name = path.name.lower()
            if name.endswith("_tables.json") or name == "tables.json":
                candidates.append(path)
        return sorted(candidates)

    @staticmethod
    def _example_files(raw_dir: Path, split_type: str) -> list[Path]:
        if split_type in {"mini-dev", "bird-mini"}:
            preferred = raw_dir / "mini_dev_sqlite.json"
            if preferred.exists():
                return [preferred]
            legacy = raw_dir / "dev.json"
            if legacy.exists():
                return [legacy]

        if split_type in {"full", "bird-full"}:
            prepared = [
                raw_dir / "train.json",
                raw_dir / "validation.json",
                raw_dir / "test.json",
            ]
            if any(path.exists() for path in prepared):
                return [path for path in prepared if path.exists()]

        files: list[Path] = []
        for path in raw_dir.rglob("*.json"):
            if BirdAdapter._skip_path(path):
                continue
            name = path.name.lower()
            if "tables" in name or "tied" in name or "manifest" in name:
                continue
            if split_type in {"mini-dev", "bird-mini"} and "sqlite" not in name and "dev" not in name:
                continue
            if any(marker in name for marker in ["train", "dev", "valid", "test", "mini_dev"]):
                files.append(path)
        return sorted(files, key=lambda item: (BirdAdapter._split_sort_key(item.name), item.as_posix()))

    @staticmethod
    def _split_sort_key(name: str) -> int:
        split = BirdAdapter._canonical_split(name)
        order = {"train": 0, "validation": 1, "test": 2}
        return order.get(split, 3)

    @staticmethod
    def _skip_path(path: Path) -> bool:
        return any(part == "__MACOSX" for part in path.parts) or path.name.startswith("._")

    def _schema_from_table_record(self, record: dict[str, Any], raw_dir: Path) -> DatabaseSchema:
        db_id = str(record["db_id"])
        table_names = record.get("table_names_original") or record.get("table_names") or []
        column_names = record.get("column_names_original") or record.get("column_names") or []
        column_types = record.get("column_types") or []
        tables: dict[str, dict[str, Any]] = {
            str(table_name): {"name": str(table_name), "columns": []}
            for table_name in table_names
        }
        for idx, column in enumerate(column_names):
            table_idx, column_name = column
            if table_idx < 0 or column_name == "*":
                continue
            table_name = str(table_names[table_idx])
            tables[table_name]["columns"].append(
                {
                    "name": str(column_name),
                    "type": column_types[idx] if idx < len(column_types) else None,
                    "index": idx,
                }
            )

        primary_keys = [self._column_ref(idx, table_names, column_names) for idx in record.get("primary_keys", [])]
        foreign_keys = [
            {
                "from": self._column_ref(left_idx, table_names, column_names),
                "to": self._column_ref(right_idx, table_names, column_names),
            }
            for left_idx, right_idx in record.get("foreign_keys", [])
        ]
        schema = DatabaseSchema(
            db_id=db_id,
            dataset_name=self.dataset_name,
            db_path=str(self._find_sqlite_path(raw_dir, db_id)) if self._find_sqlite_path(raw_dir, db_id) else None,
            tables=tables,
            foreign_keys=foreign_keys,
            primary_keys=primary_keys,
        )
        schema.serialized_schema = SchemaNormalizer.serialize_schema(schema)
        return schema

    @staticmethod
    def _column_ref(
        column_idx: Any,
        table_names: list[Any],
        column_names: list[list[Any]],
    ) -> dict[str, Any]:
        if isinstance(column_idx, list | tuple):
            if len(column_idx) >= 2:
                table_idx, column_name = column_idx[0], column_idx[1]
                if isinstance(table_idx, int) and 0 <= table_idx < len(table_names):
                    return {"table": table_names[table_idx], "column": column_name, "index": None}
                return {"table": table_idx, "column": column_name, "index": None}
            return {"table": None, "column": None, "index": None}
        if column_idx < 0 or column_idx >= len(column_names):
            return {"table": None, "column": None, "index": column_idx}
        table_idx, column_name = column_names[column_idx]
        table_name = table_names[table_idx] if 0 <= table_idx < len(table_names) else None
        return {"table": table_name, "column": column_name, "index": column_idx}

    @staticmethod
    def _find_sqlite_path(raw_dir: Path, db_id: str) -> Path | None:
        direct_candidates = [
            raw_dir / "dev_databases" / db_id / f"{db_id}.sqlite",
            raw_dir / "train_databases" / db_id / f"{db_id}.sqlite",
            raw_dir / "test_databases" / db_id / f"{db_id}.sqlite",
            raw_dir / "database" / db_id / f"{db_id}.sqlite",
            raw_dir / "databases" / db_id / f"{db_id}.sqlite",
        ]
        for candidate in direct_candidates:
            if candidate.exists():
                return candidate
        matches = list(raw_dir.rglob(f"{db_id}.sqlite"))
        return matches[0] if matches else None

    @staticmethod
    def _columns_from_schema(schema: DatabaseSchema) -> list[str]:
        columns: list[str] = []
        for table in schema.tables.values():
            for column in table.get("columns", []):
                if isinstance(column, dict):
                    columns.append(str(column.get("name")))
                else:
                    columns.append(str(column))
        return columns

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, Any]]:
        if path.suffix == ".jsonl":
            rows: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [dict(item) for item in raw]
        if isinstance(raw, dict):
            for key in ["data", "examples", "questions"]:
                if isinstance(raw.get(key), list):
                    return [dict(item) for item in raw[key]]
            return [raw]
        return []


@contextmanager
def _external_hf_datasets() -> Iterator[Any]:
    local_package = sys.modules.get("datasets")
    old_path = list(sys.path)
    root = str(PROJECT_ROOT)
    try:
        if local_package is not None:
            sys.modules.pop("datasets", None)
        sys.path = [
            item
            for item in sys.path
            if item not in {"", root, str(PROJECT_ROOT / "datasets")}
        ]
        module = importlib.import_module("datasets")
        yield module
    finally:
        sys.path = old_path
        if local_package is not None:
            sys.modules["datasets"] = local_package


def _stable_suffix(question: str, sql: str, db_id: str) -> str:
    digest = hashlib.sha1(f"{db_id}\n{question}\n{sql}".encode("utf-8")).hexdigest()
    return digest[:12]
