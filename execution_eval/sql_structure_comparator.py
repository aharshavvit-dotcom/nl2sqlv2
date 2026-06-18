from __future__ import annotations

import re
from typing import Any

from .sql_canonicalizer import SQLCanonicalizer


class SQLStructureComparator:
    def __init__(self):
        self.canonicalizer = SQLCanonicalizer()

    def compare(self, predicted_sql: str, gold_sql: str, schema: dict[str, Any] | None = None, dialect: str = "sqlite") -> dict[str, Any]:
        pred = self.canonicalizer.canonicalize(predicted_sql or "", dialect=dialect)
        gold = self.canonicalizer.canonicalize(gold_sql or "", dialect=dialect)
        errors: list[str] = []
        warnings: list[str] = [*pred.get("parse_warnings", []), *gold.get("parse_warnings", [])]
        component_scores: dict[str, float] = {}

        comparisons = {
            "tables": _set_score(pred["tables"], gold["tables"]),
            "base_table": 1.0 if _base_table(pred) == _base_table(gold) else 0.0,
            "selected_columns": _set_score(_normalize_columns(pred["columns"]), _normalize_columns(gold["columns"])),
            "aggregations": _set_score(pred["aggregations"], gold["aggregations"]),
            "joins": _join_score(pred["joins"], gold["joins"]),
            "join_count": 1.0 if len(pred["joins"]) == len(gold["joins"]) else 0.0,
            "filters": _set_score(pred["filters"], gold["filters"]),
            "group_by": _set_score(pred["group_by"], gold["group_by"]),
            "order_by": _set_score(pred["order_by"], gold["order_by"]),
            "limit": _limit_score(pred.get("limit"), gold.get("limit")),
        }
        component_scores.update(comparisons)

        if _base_table(pred) != _base_table(gold):
            errors.append("wrong_base_table")
        if pred["joins"] and not gold["joins"]:
            errors.append("unnecessary_join")
        if gold["joins"] and not pred["joins"]:
            errors.append("missing_join")
        if pred["joins"] and gold["joins"] and component_scores["joins"] < 1.0:
            errors.append("wrong_join")
        if component_scores["filters"] < 1.0:
            if pred["filters"] and not gold["filters"]:
                errors.append("extra_filter")
            elif gold["filters"] and not pred["filters"]:
                errors.append("missing_filter")
            else:
                errors.append("wrong_filter")
        if component_scores["limit"] < 1.0:
            errors.append("wrong_limit")

        weights = {
            "tables": 0.18,
            "base_table": 0.18,
            "selected_columns": 0.10,
            "aggregations": 0.10,
            "joins": 0.12,
            "join_count": 0.08,
            "filters": 0.10,
            "group_by": 0.06,
            "order_by": 0.04,
            "limit": 0.04,
        }
        score = sum(component_scores[key] * weights[key] for key in weights)
        return {
            "structure_score": round(max(0.0, min(1.0, score)), 6),
            "component_scores": component_scores,
            "errors": list(dict.fromkeys(errors)),
            "warnings": warnings,
            "predicted": pred,
            "gold": gold,
        }


def _base_table(canonical: dict[str, Any]) -> str | None:
    if canonical.get("base_table"):
        return str(canonical["base_table"])
    return canonical.get("tables", [None])[0] if canonical.get("tables") else None


def _set_score(left: list[Any], right: list[Any]) -> float:
    left_set = {_normalize_expr(item) for item in left or []}
    right_set = {_normalize_expr(item) for item in right or []}
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _join_score(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    left_set = {_join_key(item) for item in left or []}
    right_set = {_join_key(item) for item in right or []}
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _join_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item.get("table", "")).lower(), str(item.get("on", "")).lower())


def _normalize_columns(columns: list[str]) -> list[str]:
    values = []
    for column in columns:
        text = str(column).replace('"', "").lower()
        if text == "*":
            continue
        values.append(text.split(".")[-1].replace(" as ", " "))
    return values


def _normalize_expr(value: Any) -> str:
    text = str(value).replace('"', "").lower()
    text = re.sub(r"(?<![a-z0-9_])[a-z0-9_-]+\.", "", text)
    text = re.sub(r"\s+as\s+\w+", "", text)
    return " ".join(text.split())


def _limit_score(predicted: int | None, gold: int | None) -> float:
    if predicted == gold:
        return 1.0
    if gold is None and predicted is not None and predicted <= 1000:
        return 1.0
    return 0.0
