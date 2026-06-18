from __future__ import annotations

from clarification.ambiguity_detector import AmbiguityDetector
from semantic_layer import build_semantic_profile
from semantic_layer.semantic_mapper import SemanticMapper
from tests.test_60_schema_profiler import generic_schema


def test_ambiguity_detector_builds_column_mapping_options() -> None:
    schema = generic_schema()
    mapper = SemanticMapper(build_semantic_profile(schema))
    mapping = mapper.map_column("status")

    ambiguity = AmbiguityDetector().detect("show status", mapping, schema)

    assert ambiguity["ambiguous"] is True
    assert ambiguity["ambiguity_type"] == "column_mapping"
    assert {option["value"] for option in ambiguity["options"]} >= {"users.status", "assignments.status", "berth_masters.status"}
