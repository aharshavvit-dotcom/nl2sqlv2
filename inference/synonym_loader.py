from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNONYMS_PATH = ROOT / "data" / "synonyms.yaml"

SAFE_DEFAULTS: dict[str, Any] = {
    "metrics": {
        "sales": {"synonyms": ["sales", "revenue", "amount"], "candidate_columns": ["amount", "sales", "revenue"], "aggregation": "SUM"},
        "order_count": {"synonyms": ["orders", "order count", "transactions"], "candidate_columns": ["order_id", "id"], "aggregation": "COUNT"},
    },
    "dimensions": {
        "customer": {"synonyms": ["customer", "customers"], "candidate_columns": ["customer_name", "customer"]},
        "product": {"synonyms": ["product", "products"], "candidate_columns": ["product_name", "product"]},
        "status": {"synonyms": ["status", "condition"], "candidate_columns": ["status"]},
    },
    "dates": {
        "order_date": {"synonyms": ["date", "order date"], "candidate_columns": ["order_date", "date"]},
    },
    "entities": {
        "orders": {"synonyms": ["orders", "sales"], "candidate_tables": ["orders", "sales"]},
    },
}


def load_synonym_config(path: str | Path = DEFAULT_SYNONYMS_PATH) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return SAFE_DEFAULTS
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return {**SAFE_DEFAULTS, **raw}


def load_metric_dimension_maps(path: str | Path = DEFAULT_SYNONYMS_PATH) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    config = load_synonym_config(path)
    return normalize_section(config.get("metrics") or {}), normalize_section(config.get("dimensions") or {})


def normalize_section(section: dict[str, Any]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for key, value in section.items():
        if not isinstance(value, dict):
            raw_values = value or []
        else:
            raw_values = [
                *(value.get("aliases") or []),
                *(value.get("synonyms") or []),
                *(value.get("candidate_columns") or []),
                *(value.get("preferred_tables") or []),
                *(value.get("candidate_tables") or []),
            ]
        values = [str(key), str(key).replace("_", " "), *[str(item) for item in raw_values]]
        normalized[str(key)] = list(dict.fromkeys(values))
    return normalized
