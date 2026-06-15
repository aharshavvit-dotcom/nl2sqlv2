from __future__ import annotations

from typing import Any

from .schema_linearizer import extract_schema_items
from .tokenizer import tokenize


METRIC_TYPES = {"numeric"}
DIMENSION_HINTS = {"name", "status", "region", "category", "type", "city", "country", "product", "customer"}
FILTER_HINTS = {"status", "region", "category", "type", "city", "country"}
BUSINESS_SYNONYMS = {
    "amount": ["sales", "revenue", "total"],
    "sales": ["amount", "revenue"],
    "revenue": ["sales", "amount"],
    "price": ["revenue", "sales"],
    "quantity": ["units", "volume"],
    "customer": ["client", "buyer"],
    "customers": ["client", "buyer"],
    "product": ["item", "sku"],
    "products": ["item", "sku"],
    "category": ["segment"],
    "region": ["territory", "area"],
    "status": ["state"],
    "date": ["time", "month", "year"],
    "order": ["sale", "transaction"],
    "orders": ["sales", "transactions"],
}


class SchemaCandidateBuilder:
    def build_candidates(self, schema: dict | Any, question: str | None = None) -> dict[str, Any]:
        schema_items = extract_schema_items(schema)
        tables = [
            {
                "index": idx,
                "table": table,
                "display": table,
                "tokens": _candidate_tokens(_identifier_tokens(table), ["table"]),
                "type": "table",
            }
            for idx, table in enumerate(schema_items.get("tables", []))
        ]

        columns = []
        for idx, item in enumerate(schema_items.get("columns", [])):
            column_type = _refined_type(item)
            candidate = {
                "index": idx,
                "table": item["table"],
                "column": item["column"],
                "display": f"{item['table']}.{item['column']}",
                "tokens": _candidate_tokens(
                    _identifier_tokens(item["table"]) + _identifier_tokens(item["column"]),
                    [column_type, _semantic_role(item["column"], column_type)],
                ),
                "type": column_type,
                "semantic_role": _semantic_role(item["column"], column_type),
            }
            columns.append(candidate)

        metric_candidates = [item for item in columns if _is_metric_candidate(item)]
        dimension_candidates = [item for item in columns if _is_dimension_candidate(item)]
        date_candidates = [item for item in columns if item["type"] == "date"]
        filter_candidates = [item for item in columns if _is_filter_candidate(item)]

        return {
            "tables": tables,
            "columns": columns,
            "numeric_columns": [item for item in columns if item["type"] == "numeric"],
            "text_columns": [item for item in columns if item["type"] == "text"],
            "date_columns": date_candidates,
            "id_columns": [item for item in columns if item["type"] == "id"],
            "metric_candidates": metric_candidates,
            "dimension_candidates": dimension_candidates,
            "date_candidates": date_candidates,
            "filter_candidates": filter_candidates,
            "warnings": [],
        }


def build_candidate_masks(candidates: dict[str, Any], max_tables: int, max_columns: int) -> dict[str, list[float]]:
    warnings: list[str] = []
    table_mask = _mask([item["index"] for item in candidates.get("tables", [])], max_tables)
    column_mask = _mask([item["index"] for item in candidates.get("columns", [])], max_columns)
    metric_mask = _role_mask(candidates, "metric_candidates", max_columns, column_mask, warnings)
    dimension_mask = _role_mask(candidates, "dimension_candidates", max_columns, column_mask, warnings)
    date_mask = _role_mask(candidates, "date_candidates", max_columns, column_mask, warnings)
    filter_mask = _role_mask(candidates, "filter_candidates", max_columns, column_mask, warnings)
    return {
        "table_candidate_mask": table_mask,
        "column_candidate_mask": column_mask,
        "metric_column_mask": metric_mask,
        "dimension_column_mask": dimension_mask,
        "date_column_mask": date_mask,
        "filter_column_mask": filter_mask,
        "candidate_warnings": warnings,
    }


def schema_link_score_vector(link_result: dict[str, Any], max_columns: int) -> list[float]:
    scores = [0.0] * max_columns
    for item in link_result.get("top_columns", []):
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < max_columns:
            scores[idx] = max(scores[idx], float(item.get("score") or 0.0))
    return scores


def _role_mask(
    candidates: dict[str, Any],
    role_key: str,
    max_columns: int,
    fallback: list[float],
    warnings: list[str],
) -> list[float]:
    role_indexes = [item["index"] for item in candidates.get(role_key, [])]
    role_mask = _mask(role_indexes, max_columns)
    if any(role_mask):
        return role_mask
    warnings.append(f"{role_key} was empty; falling back to all valid columns")
    return list(fallback)


def _mask(indexes: list[int], size: int) -> list[float]:
    values = [0.0] * size
    for index in indexes:
        if 0 <= int(index) < size:
            values[int(index)] = 1.0
    return values


def _identifier_tokens(value: str) -> list[str]:
    tokens = tokenize(str(value).replace("_", " "))
    expanded = []
    for token in tokens:
        expanded.append(_singular(token))
    return list(dict.fromkeys(tokens + expanded))


def _candidate_tokens(base_tokens: list[str], role_tokens: list[str]) -> list[str]:
    expanded = []
    for token in [*base_tokens, *role_tokens]:
        if not token:
            continue
        normalized = str(token).replace("_", " ").lower()
        for part in tokenize(normalized):
            expanded.append(part)
            expanded.append(_singular(part))
            expanded.extend(BUSINESS_SYNONYMS.get(part, []))
            expanded.extend(BUSINESS_SYNONYMS.get(_singular(part), []))
    if "numeric" in expanded:
        expanded.append("metric")
    if "text" in expanded:
        expanded.append("dimension")
    if "date" in expanded:
        expanded.extend(["month", "year"])
    return list(dict.fromkeys(item for item in expanded if item))


def _refined_type(item: dict[str, Any]) -> str:
    name = str(item.get("column") or "").lower()
    parts = set(_identifier_tokens(name))
    if name == "id" or name.endswith("_id"):
        return "id"
    if "date" in parts or name.endswith("_date") or name.endswith("_at") or {"month", "year"} & parts:
        return "date"
    if {"amount", "revenue", "sales", "price", "quantity", "total", "cost", "fare", "count"} & parts:
        return "numeric"
    if DIMENSION_HINTS & parts:
        return "text"
    return str(item.get("type") or "text")


def _semantic_role(column: str, column_type: str) -> str:
    if column_type == "date":
        return "date_candidate"
    if column_type == "numeric" and not (column == "id" or column.endswith("_id")):
        return "metric_candidate"
    if _has_hint(column, DIMENSION_HINTS):
        return "dimension_candidate"
    if column_type == "text":
        return "dimension_candidate"
    return "filter_candidate" if _has_hint(column, FILTER_HINTS) else "column_candidate"


def _is_metric_candidate(item: dict[str, Any]) -> bool:
    column = str(item.get("column") or "")
    return item.get("type") in METRIC_TYPES and column != "id" and not column.endswith("_id")


def _is_dimension_candidate(item: dict[str, Any]) -> bool:
    return item.get("type") == "text" or _has_hint(str(item.get("column") or ""), DIMENSION_HINTS)


def _is_filter_candidate(item: dict[str, Any]) -> bool:
    return item.get("type") in {"text", "date"} or _has_hint(str(item.get("column") or ""), FILTER_HINTS)


def _has_hint(column: str, hints: set[str]) -> bool:
    return bool(set(_identifier_tokens(column)) & hints)


def _singular(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token
