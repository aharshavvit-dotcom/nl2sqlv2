from __future__ import annotations

from pathlib import Path

import pytest

from inference.runtime_schema_context import RuntimeSchemaContext
from inference.slot_resolver import SlotResolver
from nl2sql_v1.schema import read_sqlite_schema
from scripts.create_sample_db import build_database


@pytest.fixture()
def runtime_context(tmp_path: Path) -> RuntimeSchemaContext:
    db_path = tmp_path / "retail.db"
    build_database(db_path)
    return RuntimeSchemaContext(read_sqlite_schema(db_path))


def test_slot_resolver_uses_yaml_style_metric_and_dimension_synonyms(runtime_context: RuntimeSchemaContext) -> None:
    synonyms = {
        "metrics": {"sales": {"aliases": ["gross takings"]}},
        "dimensions": {
            "customer": {"aliases": ["client", "clients"]},
            "product": {"aliases": ["item", "items"]},
            "status": {"aliases": ["status"]},
            "region": {"aliases": ["region"]},
        },
    }

    slots = SlotResolver().resolve_slots(
        "Top 5 clients by gross takings",
        {"template_id": "top_n_metric_by_dimension", "confidence": 0.8},
        [],
        runtime_context,
        synonyms,
    )["slots"]

    assert slots["metric"]["value"] == "sales"
    assert slots["dimension"]["value"] == "customer"


def test_slot_resolver_uses_yaml_style_filter_synonyms(runtime_context: RuntimeSchemaContext) -> None:
    synonyms = {
        "metrics": {"sales": {"aliases": ["gross takings"]}},
        "dimensions": {
            "status": {"aliases": ["status"]},
            "region": {"aliases": ["region"]},
            "category": {"aliases": ["category"]},
        },
    }

    status_slots = SlotResolver().resolve_slots(
        "Orders where status is completed",
        {"template_id": "simple_filter", "confidence": 0.8},
        [],
        runtime_context,
        synonyms,
    )["slots"]
    region_slots = SlotResolver().resolve_slots(
        "Show gross takings for region west",
        {"template_id": "metric_summary", "confidence": 0.8},
        [],
        runtime_context,
        synonyms,
    )["slots"]

    assert status_slots["filter_column"]["value"] == "status"
    assert status_slots["filter_value"]["value"] == "completed"
    assert region_slots["metric"]["value"] == "sales"
    assert region_slots["filter_column"]["value"] == "region"
    assert region_slots["filter_value"]["value"] == "west"

