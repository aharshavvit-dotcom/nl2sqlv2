from __future__ import annotations

from typing import Any

from .date_column_detector import DateColumnDetector
from .dimension_detector import DimensionDetector
from .entity_detector import EntityDetector
from .glossary_generator import GlossaryGenerator
from .metric_detector import MetricDetector
from .schema_profiler import SchemaProfiler, schema_fingerprint
from .semantic_mapper import SemanticMapper
from .semantic_profile_store import SemanticProfileStore


def build_semantic_profile(schema: dict[str, Any]) -> dict[str, Any]:
    profiler = SchemaProfiler()
    profile = profiler.profile(schema)
    glossary = GlossaryGenerator().generate(schema, profile)
    profile["glossary"] = glossary
    profile.update(MetricDetector().detect(schema, profile))
    profile.update(DimensionDetector().detect(schema, profile))
    profile.update(DateColumnDetector().detect(schema, profile))
    profile.update(EntityDetector().detect(schema, profile))
    profile["filters"] = _filters_from_profile(profile)
    return profile


def _filters_from_profile(profile: dict[str, Any]) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    for table, info in (profile.get("tables") or {}).items():
        for column in info.get("likely_filters", []):
            key = f"{table}.{column}"
            filters[key] = {
                "table": table,
                "column": column,
                "aliases": (profile.get("glossary", {}).get("columns") or {}).get(key, [column.replace("_", " ")]),
                "confidence": 0.85,
            }
    return filters


__all__ = [
    "DateColumnDetector",
    "DimensionDetector",
    "EntityDetector",
    "GlossaryGenerator",
    "MetricDetector",
    "SchemaProfiler",
    "SemanticMapper",
    "SemanticProfileStore",
    "build_semantic_profile",
    "schema_fingerprint",
]
