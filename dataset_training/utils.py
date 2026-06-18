from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def normalize_dataset_name(value: str) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def model_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def schema_tokens(schema: dict[str, Any] | None) -> set[str]:
    if not schema:
        return set()
    raw_tables = schema.get("tables", schema)
    tokens: set[str] = set()
    if not isinstance(raw_tables, dict):
        return tokens
    for table, info in raw_tables.items():
        tokens.update(_identifier_tokens(str(table)))
        columns = info.get("columns", info) if isinstance(info, dict) else {}
        if isinstance(columns, dict):
            names = columns.keys()
        else:
            names = [item.get("name", item) if isinstance(item, dict) else item for item in columns]
        for column in names:
            tokens.update(_identifier_tokens(str(column)))
    semantic_profile = schema.get("semantic_profile") if isinstance(schema, dict) else None
    if isinstance(semantic_profile, dict):
        for table_info in (semantic_profile.get("tables") or {}).values():
            for alias in table_info.get("aliases") or []:
                tokens.update(_identifier_tokens(str(alias)))
            for column in table_info.get("columns") or []:
                for alias in column.get("aliases") or []:
                    tokens.update(_identifier_tokens(str(alias)))
    return {token for token in tokens if token}


def query_ir_tables(query_ir: dict[str, Any] | None) -> set[str]:
    ir = query_ir or {}
    tables = {str(item) for item in ir.get("required_tables") or [] if item}
    if ir.get("base_table"):
        tables.add(str(ir["base_table"]))
    for section in ["metrics", "dimensions", "filters"]:
        for item in ir.get(section) or []:
            if item.get("table"):
                tables.add(str(item["table"]))
    for item in ir.get("date_filters") or []:
        if item.get("date_table"):
            tables.add(str(item["date_table"]))
    for join in ir.get("joins") or []:
        for key in ["left_table", "right_table"]:
            if join.get(key):
                tables.add(str(join[key]))
    return tables


def _identifier_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9]+|_", value.lower()) if token}
