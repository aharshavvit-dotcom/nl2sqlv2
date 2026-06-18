from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import re
from typing import Any

try:  # pragma: no cover - optional in minimal installs
    from nl2sql_v1.schema import SchemaGraph
except Exception:  # pragma: no cover
    SchemaGraph = None  # type: ignore


SENSITIVE_MARKERS = {
    "password",
    "passwd",
    "pwd",
    "token",
    "secret",
    "ssn",
    "email",
    "phone",
    "address",
    "dob",
    "birth",
    "credit_card",
    "api_key",
    "auth",
}
NUMERIC_TYPES = ("int", "real", "float", "double", "numeric", "decimal", "number", "money")
TEXT_TYPES = ("char", "text", "string", "varchar", "uuid")
DATE_TYPES = ("date", "time", "timestamp")
DATE_MARKERS = ("date", "time", "created", "updated", "timestamp", "started", "ended")
DIMENSION_MARKERS = {"name", "code", "status", "type", "category", "role", "department", "location", "class"}
FILTER_MARKERS = {"status", "type", "category", "role", "department", "location", "class", "state"}
ENTITY_TABLES = {"users", "customers", "employees", "vessels", "berths", "products", "ports"}
TRANSACTION_MARKERS = ("transaction", "order", "invoice", "sale", "payment", "event", "log", "movement")
BRIDGE_MARKERS = ("assignment", "assignments", "mapping", "map", "bridge")


class SchemaProfiler:
    def profile(self, schema: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_schema(schema)
        relationships = normalized["relationships"]
        tables: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []

        for table, info in normalized["tables"].items():
            columns = info.get("columns", [])
            primary_keys = [column["name"] for column in columns if column.get("is_primary_key")]
            sensitive = [column["name"] for column in columns if is_sensitive_column(column["name"])]
            safe_columns = [column["name"] for column in columns if column["name"] not in sensitive]
            numeric_measures = [column["name"] for column in columns if is_metric_column(column)]
            dimensions = [column["name"] for column in columns if is_dimension_column(column)]
            dates = [column["name"] for column in columns if is_date_column(column)]
            filters = [column for column in dimensions if _identifier(column) in FILTER_MARKERS or "status" in _identifier(column)]
            fk_count = sum(1 for rel in relationships if rel.get("from_table") == table)
            table_type = classify_table(table, columns, fk_count, numeric_measures)
            table_aliases = table_aliases_for(table)

            tables[table] = {
                "table_type": table_type,
                "aliases": table_aliases,
                "columns": [
                    {
                        **column,
                        "aliases": column_aliases_for(column["name"]),
                        "semantic_role": semantic_role(column),
                        "is_sensitive": column["name"] in sensitive,
                    }
                    for column in columns
                ],
                "primary_keys": primary_keys,
                "foreign_keys": [rel for rel in relationships if rel.get("from_table") == table],
                "safe_columns": safe_columns,
                "sensitive_columns": sensitive,
                "likely_metrics": numeric_measures,
                "likely_dimensions": dimensions,
                "likely_dates": dates,
                "likely_filters": filters,
            }
            if not safe_columns:
                warnings.append(f"Table {table} has no non-sensitive columns.")

        return {
            "dialect": normalized.get("dialect", "sqlite"),
            "database": normalized.get("database"),
            "schema_name": normalized.get("schema_name"),
            "tables": tables,
            "relationships": relationships,
            "warnings": warnings,
            "schema_fingerprint": schema_fingerprint(normalized),
        }


def normalize_schema(schema: Any) -> dict[str, Any]:
    if SchemaGraph is not None and isinstance(schema, SchemaGraph):
        tables: dict[str, dict[str, Any]] = {}
        relationships: list[dict[str, str]] = []
        for table_name, table in schema.tables.items():
            columns = []
            for column_name, column in table.columns.items():
                columns.append(_column_payload(column_name, str(column.type), bool(column.primary_key)))
            tables[table_name] = {"columns": columns}
            for fk in table.foreign_keys:
                relationships.append(
                    {
                        "from_table": fk.table,
                        "from_column": fk.constrained_column,
                        "to_table": fk.referred_table,
                        "to_column": fk.referred_column,
                    }
                )
        return {
            "dialect": getattr(schema, "dialect", "sqlite") or "sqlite",
            "database": None,
            "schema_name": None,
            "tables": tables,
            "relationships": _dedupe_relationships(relationships),
        }

    payload = schema if isinstance(schema, dict) else {}
    raw_tables = payload.get("tables", payload)
    relationships = _extract_relationships(payload)
    tables: dict[str, dict[str, Any]] = {}
    for table_name, table_info in (raw_tables or {}).items():
        if not isinstance(table_info, dict):
            continue
        raw_columns = table_info.get("columns", table_info)
        primary_keys = set(table_info.get("primary_keys", []))
        columns = []
        if isinstance(raw_columns, dict):
            iterable = raw_columns.items()
        else:
            iterable = [(str(item.get("name", item)), item) for item in raw_columns or []]
        for column_name, raw in iterable:
            raw_dict = asdict(raw) if is_dataclass(raw) else (raw if isinstance(raw, dict) else {})
            columns.append(
                _column_payload(
                    str(column_name),
                    str(raw_dict.get("type", "")),
                    bool(raw_dict.get("primary_key") or raw_dict.get("is_primary_key") or column_name in primary_keys),
                    raw_dict,
                )
            )
        tables[str(table_name)] = {"columns": columns}
        for fk in table_info.get("foreign_keys", []) or []:
            rel = _normalize_relationship(fk, str(table_name))
            if rel:
                relationships.append(rel)

    relationships.extend(_infer_relationships(tables, relationships))
    return {
        "dialect": str(payload.get("dialect") or "sqlite").lower(),
        "database": payload.get("database"),
        "schema_name": payload.get("schema_name") or payload.get("schema"),
        "tables": tables,
        "relationships": _dedupe_relationships(relationships),
    }


def schema_fingerprint(schema: Any) -> str:
    normalized = normalize_schema(schema)
    safe = {
        "dialect": normalized.get("dialect"),
        "schema_name": normalized.get("schema_name"),
        "tables": {
            table: [
                {"name": column.get("name"), "type": column.get("type"), "is_primary_key": column.get("is_primary_key")}
                for column in info.get("columns", [])
            ]
            for table, info in sorted((normalized.get("tables") or {}).items())
        },
        "relationships": sorted(
            [
                {
                    "from_table": rel.get("from_table"),
                    "from_column": rel.get("from_column"),
                    "to_table": rel.get("to_table"),
                    "to_column": rel.get("to_column"),
                }
                for rel in normalized.get("relationships", [])
            ],
            key=lambda item: json.dumps(item, sort_keys=True),
        ),
    }
    return hashlib.sha256(json.dumps(safe, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def classify_table(table: str, columns: list[dict[str, Any]], fk_count: int, metrics: list[str]) -> str:
    name = _identifier(table)
    if any(marker in name for marker in BRIDGE_MARKERS) or (fk_count >= 2 and len(columns) <= 8):
        return "bridge"
    if name.endswith("_master") or name.endswith("_masters") or name.endswith("_lookup") or name.endswith("_types") or name.endswith("_categories"):
        return "lookup"
    if name in ENTITY_TABLES or any(name == singularize(item) for item in ENTITY_TABLES):
        return "entity"
    if any(marker in name for marker in TRANSACTION_MARKERS) or (metrics and fk_count >= 1):
        return "transaction"
    if any(_identifier(column["name"]) in {"name", "title", "code"} for column in columns):
        return "entity"
    return "unknown"


def table_aliases_for(table: str) -> list[str]:
    tokens = tokenize(table)
    singular_tokens = [singularize(token) for token in tokens]
    aliases = {table.lower(), table.replace("_", " ").lower(), " ".join(tokens), " ".join(singular_tokens)}
    if tokens:
        aliases.update(tokens)
        aliases.update(singular_tokens)
        aliases.update(pluralize(token) for token in singular_tokens)
        aliases.add("_".join(singular_tokens))
    if len(tokens) > 1:
        aliases.add(tokens[-1])
        aliases.add(singular_tokens[-1])
        aliases.add(pluralize(singular_tokens[-1]))
    return sorted(alias for alias in aliases if alias)


def column_aliases_for(column: str) -> list[str]:
    tokens = tokenize(column)
    base = " ".join(tokens)
    aliases = {column.lower(), column.replace("_", " ").lower(), base}
    if tokens:
        aliases.update(tokens)
    ident = _identifier(column)
    if ident.endswith("_code") and len(tokens) > 1:
        aliases.add("code")
        aliases.add(f"{' '.join(tokens[:-1])} code")
    if ident in {"created_at", "created_date"}:
        aliases.update({"created date", "created at", "creation date"})
    if ident in {"updated_at", "updated_date"}:
        aliases.update({"updated date", "updated at", "update date"})
    if ident.endswith("_date") and len(tokens) > 1:
        aliases.add(f"{tokens[0]} date")
        if tokens[0].endswith("ed"):
            aliases.add(f"{tokens[0][:-2]}ment date")
    return sorted(alias for alias in aliases if alias)


def is_sensitive_column(column_name: str) -> bool:
    ident = _identifier(column_name)
    return any(marker in ident for marker in SENSITIVE_MARKERS)


def is_id_column(column_name: str) -> bool:
    ident = _identifier(column_name)
    return ident == "id" or ident.endswith("_id") or ident.endswith("_key") or ident in {"created_by", "updated_by"}


def is_metric_column(column: dict[str, Any]) -> bool:
    if is_sensitive_column(column["name"]) or is_id_column(column["name"]):
        return False
    return bool(column.get("is_numeric"))


def is_dimension_column(column: dict[str, Any]) -> bool:
    ident = _identifier(column["name"])
    if is_sensitive_column(column["name"]) or is_id_column(column["name"]) or is_date_column(column):
        return False
    return bool(column.get("is_text") or ident in DIMENSION_MARKERS or any(marker in ident for marker in DIMENSION_MARKERS))


def is_date_column(column: dict[str, Any]) -> bool:
    ident = _identifier(column["name"])
    typ = str(column.get("type", "")).lower()
    return any(marker in ident for marker in DATE_MARKERS) or any(marker in typ for marker in DATE_TYPES)


def semantic_role(column: dict[str, Any]) -> str:
    if is_sensitive_column(column["name"]):
        return "sensitive"
    if is_date_column(column):
        return "date"
    if is_metric_column(column):
        return "metric"
    if is_dimension_column(column):
        return "dimension"
    if is_id_column(column["name"]):
        return "identifier"
    return "attribute"


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").replace("_", " ").replace("-", " ").lower())


def singularize(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def pluralize(token: str) -> str:
    if not token or token.endswith("s"):
        return token
    if token.endswith("y") and len(token) > 3:
        return token[:-1] + "ies"
    return token + "s"


def _identifier(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower())).strip("_")


def _column_payload(column_name: str, column_type: str, primary_key: bool, raw: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = raw or {}
    name = _identifier(column_name)
    typ = column_type.lower()
    return {
        **raw,
        "name": column_name,
        "type": column_type,
        "is_primary_key": primary_key,
        "primary_key": primary_key,
        "is_numeric": any(marker in typ for marker in NUMERIC_TYPES) or primary_key,
        "is_text": any(marker in typ for marker in TEXT_TYPES),
        "is_date": any(marker in name for marker in DATE_MARKERS) or any(marker in typ for marker in DATE_TYPES),
        "is_id": is_id_column(column_name),
        "is_sensitive": is_sensitive_column(column_name),
    }


def _extract_relationships(payload: dict[str, Any]) -> list[dict[str, str]]:
    relationships: list[dict[str, str]] = []
    for key in ("foreign_keys", "relationships"):
        for raw in payload.get(key, []) or []:
            rel = _normalize_relationship(raw)
            if rel:
                relationships.append(rel)
    return relationships


def _normalize_relationship(raw: dict[str, Any], default_from_table: str | None = None) -> dict[str, str] | None:
    from_table = raw.get("from_table") or raw.get("constrained_table") or default_from_table
    from_column = raw.get("from_column") or raw.get("constrained_column") or raw.get("column")
    to_table = raw.get("to_table") or raw.get("referred_table") or raw.get("references_table")
    to_column = raw.get("to_column") or raw.get("referred_column") or raw.get("references_column")
    if not all([from_table, from_column, to_table, to_column]):
        return None
    return {"from_table": str(from_table), "from_column": str(from_column), "to_table": str(to_table), "to_column": str(to_column)}


def _infer_relationships(tables: dict[str, dict[str, Any]], existing: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = {(rel["from_table"], rel["from_column"], rel["to_table"], rel["to_column"]) for rel in existing}
    inferred: list[dict[str, str]] = []
    table_names = set(tables)
    for table, info in tables.items():
        for column in info.get("columns", []):
            name = _identifier(column["name"])
            if not name.endswith("_id") or name == "id":
                continue
            target_stem = name[:-3]
            targets = [candidate for candidate in table_names if singularize(candidate.lower()) == target_stem or candidate.lower() == pluralize(target_stem)]
            for target in targets:
                target_columns = {col["name"] for col in tables[target].get("columns", [])}
                if "id" not in target_columns:
                    continue
                key = (table, column["name"], target, "id")
                if key not in seen:
                    inferred.append({"from_table": table, "from_column": column["name"], "to_table": target, "to_column": "id", "inferred": "true"})
    return inferred


def _dedupe_relationships(relationships: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped = []
    seen = set()
    for rel in relationships:
        key = (rel.get("from_table"), rel.get("from_column"), rel.get("to_table"), rel.get("to_column"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rel)
    return deduped
