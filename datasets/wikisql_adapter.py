from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import DatabaseSchema, Text2SQLExample
from .schema_normalizer import SchemaNormalizer


AGGREGATIONS = {
    0: "",
    1: "MAX",
    2: "MIN",
    3: "COUNT",
    4: "SUM",
    5: "AVG",
}

OPERATORS = {
    0: "=",
    1: ">",
    2: "<",
    3: "OP",
}


class WikiSQLAdapter:
    dataset_name = "wikisql"

    def load_schemas(self, raw_dir: Path) -> dict[str, DatabaseSchema]:
        table_map = self._load_table_map(raw_dir)
        schemas: dict[str, DatabaseSchema] = {}
        for table_id, table_schema in table_map.items():
            headers = table_schema.get("header") or []
            schema = DatabaseSchema(
                db_id=table_id,
                dataset_name=self.dataset_name,
                tables={
                    table_id: {
                        "name": table_id,
                        "columns": [{"name": str(header)} for header in headers],
                    }
                },
                primary_keys=[],
                foreign_keys=[],
            )
            schema.serialized_schema = SchemaNormalizer.serialize_schema(schema)
            schemas[table_id] = schema
        return schemas

    def load_examples(self, raw_dir: Path) -> list[Text2SQLExample]:
        table_map = self._load_table_map(raw_dir)
        schemas = self.load_schemas(raw_dir)
        examples: list[Text2SQLExample] = []
        for file_split, split in [("train", "train"), ("dev", "validation"), ("test", "test")]:
            path = raw_dir / f"{file_split}.jsonl"
            if not path.exists():
                continue
            for index, record in enumerate(self._read_jsonl(path)):
                table_id = record.get("table_id", "")
                table_schema = table_map.get(table_id, {"header": []})
                sql = self.convert_wikisql_to_sql(record, table_schema)
                schema = schemas.get(table_id)
                headers = [str(item) for item in table_schema.get("header", [])]
                examples.append(
                    Text2SQLExample(
                        example_id=f"wikisql:{file_split}:{index}",
                        dataset_name=self.dataset_name,
                        db_id=table_id,
                        question=record.get("question", ""),
                        sql=sql,
                        split=split,
                        schema=schema.to_dict() if schema else None,
                        tables=[table_id],
                        columns=headers,
                        difficulty="easy",
                        source_file=f"{file_split}.jsonl",
                    )
                )
        return examples

    def convert_wikisql_to_sql(self, record: dict[str, Any], table_schema: dict[str, Any]) -> str:
        sql_obj = record.get("sql") or {}
        headers = table_schema.get("header") or []
        table_id = record.get("table_id") or table_schema.get("id") or "table"
        sel_idx = int(sql_obj.get("sel", 0))
        agg_idx = int(sql_obj.get("agg", 0))
        selected_column = self._header_at(headers, sel_idx)
        agg = AGGREGATIONS.get(agg_idx, "")
        if agg:
            select_expr = f'{agg}({self._quote_identifier(selected_column)})'
        else:
            select_expr = self._quote_identifier(selected_column)

        clauses = [f"SELECT {select_expr}", f"FROM {self._quote_identifier(str(table_id))}"]
        conditions = []
        for condition in sql_obj.get("conds", []) or []:
            if len(condition) < 3:
                continue
            col_idx, op_idx, value = condition[0], condition[1], condition[2]
            column = self._header_at(headers, int(col_idx))
            operator = OPERATORS.get(int(op_idx), "=")
            conditions.append(
                f"{self._quote_identifier(column)} {operator} {self._literal(value)}"
            )
        if conditions:
            clauses.append("WHERE " + " AND ".join(conditions))
        return "\n".join(clauses)

    @staticmethod
    def _load_table_map(raw_dir: Path) -> dict[str, dict[str, Any]]:
        tables: dict[str, dict[str, Any]] = {}
        for split in ["train", "dev", "test"]:
            path = raw_dir / f"{split}.tables.jsonl"
            if not path.exists():
                continue
            for record in WikiSQLAdapter._read_jsonl(path):
                table_id = record.get("id")
                if table_id:
                    tables[table_id] = record
        return tables

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _header_at(headers: list[Any], index: int) -> str:
        if 0 <= index < len(headers):
            return str(headers[index])
        return f"column_{index}"

    @staticmethod
    def _quote_identifier(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'

    @staticmethod
    def _literal(value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int | float):
            return str(value)
        return "'" + str(value).replace("'", "''") + "'"
