from __future__ import annotations

from semantic_layer.glossary_generator import GlossaryGenerator
from semantic_layer.schema_profiler import SchemaProfiler
from tests.test_60_schema_profiler import generic_schema


def test_glossary_generates_schema_specific_aliases_without_retail_terms() -> None:
    schema = generic_schema()
    profile = SchemaProfiler().profile(schema)
    glossary = GlossaryGenerator().generate(schema, profile)

    assert "berth" in glossary["tables"]["berth_masters"]
    assert "berths" in glossary["tables"]["berth_masters"]
    assert "berth code" in glossary["columns"]["berth_masters.berth_code"]
    assert "code" in glossary["columns"]["berth_masters.berth_code"]
    assert "created date" in glossary["columns"]["users.created_at"]
    assert "revenue" not in glossary["tables"]["users"]
