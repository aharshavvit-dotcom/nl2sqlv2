from __future__ import annotations

from semantic_layer import build_semantic_profile
from semantic_layer.semantic_mapper import SemanticMapper
from tests.test_60_schema_profiler import generic_schema


def test_semantic_mapper_maps_aliases_and_detects_ambiguous_columns() -> None:
    mapper = SemanticMapper(build_semantic_profile(generic_schema()))

    table = mapper.map_table("berth")
    assert table["matched"] is True
    assert table["target"] == "berth_masters"

    role = mapper.map_column("role", table="users")
    assert role["matched"] is True
    assert role["target"] == "users.role"

    status = mapper.map_column("status")
    assert status["matched"] is False
    assert status["ambiguous"] is True
    assert status["requires_clarification"] is True
    assert {item["target"] for item in status["alternatives"]} >= {"users.status", "assignments.status", "berth_masters.status"}


def test_semantic_mapper_does_not_apply_retail_mappings_to_generic_schema() -> None:
    mapper = SemanticMapper(build_semantic_profile(generic_schema()))

    revenue = mapper.map_metric("revenue")

    assert revenue["matched"] is False
    assert revenue["requires_clarification"] is True
    assert all("orders.amount" not in item["target"] for item in revenue["alternatives"])
