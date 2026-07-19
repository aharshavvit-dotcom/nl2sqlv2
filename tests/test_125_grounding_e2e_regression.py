"""
Purpose: Protects model regression behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import pytest
from inference.runtime_schema_context import RuntimeSchemaContext
from inference.prediction_models import RetrievedCandidate
from inference.slot_resolver import SlotResolver
from inference.schema_aware_mapper import SchemaAwareMapper


def test_grounding_e2e_flow():
    schema = {
        "tables": {
            "orders": {
                "columns": {
                    "id": {"type": "INTEGER", "primary_key": True},
                    "status": {"type": "TEXT", "sample_values": ["completed", "pending", "failed"]},
                    "amount": {"type": "REAL"},
                }
            },
            "customers": {
                "columns": {
                    "id": {"type": "INTEGER", "primary_key": True},
                    "name": {"type": "TEXT", "sample_values": ["Alice", "Bob", "Charlie"]},
                    "region": {"type": "TEXT", "sample_values": ["North", "South", "East", "West"]},
                }
            },
        },
        "foreign_keys": [
            {"child_table": "orders", "child_column": "customer_id", "parent_table": "customers", "parent_column": "id"}
        ],
    }

    ctx = RuntimeSchemaContext(schema)
    resolver = SlotResolver()

    res = resolver.resolve_slots(
        question="show details of customers named Alice",
        selected_template={"template_id": "simple_filter"},
        candidates=[],
        schema_context=ctx,
    )

    slots = res["slots"]
    assert slots["filter_value"]["value"] == "Alice"
    assert slots["filter_column"]["value"] == "name"

    mapper = SchemaAwareMapper()
    mapping = mapper.map_slots_to_schema(slots, ctx, template_id="simple_filter")
    assert mapping.filter_table == "customers"
    assert mapping.filter_column == "name"
