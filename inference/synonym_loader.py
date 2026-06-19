from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNONYMS_PATH = ROOT / "data" / "synonyms.yaml"

SAFE_DEFAULTS: dict[str, Any] = {
    "metrics": {},
    "dimensions": {},
    "dates": {"date": {"synonyms": ["date", "time"], "candidate_columns": ["date", "created_at", "updated_at"]}},
    "entities": {},
}


def load_synonym_config(path: str | Path = DEFAULT_SYNONYMS_PATH) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return SAFE_DEFAULTS
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if "business_terms" in raw or "sample_retail_physical_mappings" in raw:
        raw = _flatten_nested_config(raw)
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


def _flatten_nested_config(raw: dict[str, Any]) -> dict[str, Any]:
    business = raw.get("business_terms") or {}
    physical = raw.get("sample_retail_physical_mappings") or {}
    rules = raw.get("schema_mapping_rules") or {}
    flattened: dict[str, Any] = {}
    for section in ["metrics", "dimensions", "entities", "dates", "filters"]:
        merged = {}
        for source in [business.get(section), physical.get(section), rules.get(section), raw.get(section)]:
            if isinstance(source, dict):
                for key, value in source.items():
                    existing = merged.get(key, {})
                    if isinstance(existing, dict) and isinstance(value, dict):
                        merged[key] = {**existing, **value}
                    else:
                        merged[key] = value
        if merged:
            flattened[section] = merged
    return flattened
