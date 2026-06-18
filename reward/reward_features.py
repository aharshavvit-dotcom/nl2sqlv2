from __future__ import annotations

import re
from typing import Any

from dataset_training.utils import query_ir_tables


JOIN_WORDS = {"join", "with", "including", "along", "by"}
UNSAFE_WORDS = {"insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke", "copy"}


def normalized_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9]+|_", str(value).lower()) if token}


def schema_table_names(schema: dict[str, Any] | None) -> list[str]:
    if not schema:
        return []
    tables = schema.get("tables", schema)
    return sorted(str(table) for table in tables) if isinstance(tables, dict) else []


def requested_base_table(question: str, schema: dict[str, Any] | None) -> str | None:
    question_tokens = normalized_tokens(question)
    best_table = None
    best_score = 0
    for table in schema_table_names(schema):
        table_tokens = normalized_tokens(table)
        score = len(question_tokens & table_tokens)
        if table.lower() in question.lower():
            score += 3
        if score > best_score:
            best_table = table
            best_score = score
    return best_table if best_score > 0 else None


def asks_for_join(question: str) -> bool:
    tokens = normalized_tokens(question)
    return bool(tokens & JOIN_WORDS)


def asks_for_filter(question: str) -> bool:
    return bool(re.search(r"\b(where|with|status|role|after|before|equals|=)\b", question.lower()))


def asks_for_date_filter(question: str) -> bool:
    return bool(re.search(r"\b(last|month|year|date|after|before|between|today|yesterday)\b", question.lower()))


def asks_for_metric(question: str) -> bool:
    return _infer_pattern(question) in {"metric_summary", "metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension", "count_records", "count_by_dimension"}


def asks_for_dimension(question: str) -> bool:
    return bool(re.search(r"\bby\s+\w+", question.lower()))


def has_unsafe_sql(sql: str | None) -> bool:
    if not sql:
        return False
    upper = sql.upper()
    return any(re.search(rf"\b{word.upper()}\b", upper) for word in UNSAFE_WORDS)


def has_select_star(sql: str | None, query_ir: dict[str, Any]) -> bool:
    if sql and re.search(r"SELECT\s+\*", sql, flags=re.IGNORECASE):
        return True
    for metric in query_ir.get("metrics") or []:
        if metric.get("expression") == "*" and str(metric.get("aggregation", "")).upper() != "COUNT":
            return True
    return False


def unnecessary_join(question: str, query_ir: dict[str, Any]) -> bool:
    return bool(query_ir.get("joins")) and _infer_pattern(question) in {"show_records", "count_records", "simple_filter"} and not asks_for_join(question)


def candidate_tables(query_ir: dict[str, Any]) -> set[str]:
    return query_ir_tables(query_ir)


def _infer_pattern(question: str) -> str:
    text = str(question or "").lower()
    if any(word in text for word in ["top", "highest", "best", "most"]):
        return "top_n_metric_by_dimension"
    if any(word in text for word in ["bottom", "lowest", "least", "worst"]):
        return "bottom_n_metric_by_dimension"
    if "count" in text or "how many" in text or "number of" in text:
        return "count_by_dimension" if " by " in text else "count_records"
    if " by " in text:
        return "metric_by_dimension"
    if any(word in text for word in ["sales", "revenue", "sum", "average", "avg"]):
        return "metric_summary"
    return "simple_filter" if asks_for_filter(text) else "show_records"
