from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import DatabaseSchema, Text2SQLExample
from .schema_normalizer import SchemaNormalizer


class SpiderAdapter:
    dataset_name = "spider"

    def load_schemas(self, raw_dir: Path) -> dict[str, DatabaseSchema]:
        tables_path = raw_dir / "tables.json"
        if not tables_path.exists():
            return {}

        records = json.loads(tables_path.read_text(encoding="utf-8"))
        schemas: dict[str, DatabaseSchema] = {}
        for record in records:
            db_id = record["db_id"]
            table_names = record.get("table_names_original") or record.get("table_names") or []
            column_names = record.get("column_names_original") or record.get("column_names") or []
            column_types = record.get("column_types") or []
            tables: dict[str, dict[str, Any]] = {
                table_name: {"name": table_name, "columns": []}
                for table_name in table_names
            }
            all_columns: list[dict[str, Any]] = []
            for idx, column in enumerate(column_names):
                table_idx, column_name = column
                if table_idx < 0 or column_name == "*":
                    continue
                table_name = table_names[table_idx]
                column_info = {
                    "name": column_name,
                    "type": column_types[idx] if idx < len(column_types) else None,
                    "index": idx,
                }
                tables[table_name]["columns"].append(column_info)
                all_columns.append({"table": table_name, **column_info})

            primary_keys = []
            for column_idx in record.get("primary_keys", []):
                primary_keys.append(self._column_ref(column_idx, table_names, column_names))

            foreign_keys = []
            for left_idx, right_idx in record.get("foreign_keys", []):
                foreign_keys.append(
                    {
                        "from": self._column_ref(left_idx, table_names, column_names),
                        "to": self._column_ref(right_idx, table_names, column_names),
                    }
                )

            db_path = self._find_sqlite_path(raw_dir, db_id)
            schema = DatabaseSchema(
                db_id=db_id,
                dataset_name=self.dataset_name,
                db_path=str(db_path) if db_path else None,
                tables=tables,
                foreign_keys=foreign_keys,
                primary_keys=primary_keys,
            )
            schema.serialized_schema = SchemaNormalizer.serialize_schema(schema)
            schemas[db_id] = schema
        return schemas

    def load_examples(self, raw_dir: Path) -> list[Text2SQLExample]:
        schemas = self.load_schemas(raw_dir)
        examples: list[Text2SQLExample] = []
        files = [
            ("train_spider.json", "train"),
            ("train_others.json", "train"),
            ("dev.json", "validation"),
            ("test.json", "test"),
        ]
        for file_name, split in files:
            path = raw_dir / file_name
            if not path.exists():
                continue
            records = json.loads(path.read_text(encoding="utf-8"))
            for index, record in enumerate(records):
                example = self.normalize_example(record, split, file_name, index)
                schema = schemas.get(example.db_id)
                if schema:
                    example.db_path = schema.db_path
                    example.schema = schema.to_dict()
                    example.tables = list(schema.tables)
                    example.columns = [
                        str(column)
                        for table in schema.tables.values()
                        for column in self._column_names(table)
                    ]
                examples.append(example)
        return examples

    def normalize_example(
        self,
        record: dict[str, Any],
        split: str,
        source_file: str,
        index: int = 0,
    ) -> Text2SQLExample:
        db_id = record.get("db_id") or record.get("database_id") or ""
        source_stem = Path(source_file).stem
        return Text2SQLExample(
            example_id=f"spider:{source_stem}:{index}",
            dataset_name=self.dataset_name,
            db_id=db_id,
            question=record.get("question", ""),
            sql=record.get("query") or record.get("sql") or "",
            split=split,
            difficulty=record.get("difficulty"),
            source_file=source_file,
        )

    @staticmethod
    def _column_ref(
        column_idx: int,
        table_names: list[str],
        column_names: list[list[Any]],
    ) -> dict[str, Any]:
        if column_idx < 0 or column_idx >= len(column_names):
            return {"table": None, "column": None, "index": column_idx}
        table_idx, column_name = column_names[column_idx]
        table_name = table_names[table_idx] if table_idx >= 0 and table_idx < len(table_names) else None
        return {"table": table_name, "column": column_name, "index": column_idx}

    @staticmethod
    def _find_sqlite_path(raw_dir: Path, db_id: str) -> Path | None:
        for folder_name in ["database", "databases"]:
            candidate = raw_dir / folder_name / db_id / f"{db_id}.sqlite"
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _column_names(table_info: dict[str, Any]) -> list[str]:
        columns = table_info.get("columns") or []
        return [str(item.get("name", item)) if isinstance(item, dict) else str(item) for item in columns]
