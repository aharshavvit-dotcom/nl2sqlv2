"""Full runtime checks for generic PostgreSQL-shaped schemas."""

from __future__ import annotations

import pytest

from generic_planner import SchemaProfile, TableIntentResolver
from inference.prediction_orchestrator import PredictionOrchestrator
from inference.runtime_join_planner import RuntimeJoinPlanner
from inference.runtime_schema_context import RuntimeSchemaContext
from inference.schema_aware_mapper import SchemaAwareMapper
from inference.slot_resolver import SlotResolver
from tests.test_10_generic_table_intent import GENERIC_POSTGRES_SCHEMA


NON_RETAIL_SCHEMA = {
    "dialect": "postgres",
    "tables": {
        **GENERIC_POSTGRES_SCHEMA["tables"],
        "berths": {"columns": [{"name": "id", "type": "integer", "is_primary_key": True}, {"name": "berth_name", "type": "text"}]},
        "vessels": {"columns": [{"name": "id", "type": "integer", "is_primary_key": True}, {"name": "vessel_name", "type": "text"}]},
        "terminals": {"columns": [{"name": "id", "type": "integer", "is_primary_key": True}, {"name": "terminal_name", "type": "text"}]},
        "service_orders": {
            "columns": [
                {"name": "id", "type": "integer", "is_primary_key": True},
                {"name": "vessel_id", "type": "integer"},
                {"name": "terminal_id", "type": "integer"},
                {"name": "status", "type": "text"},
                {"name": "cost", "type": "numeric"},
                {"name": "created_at", "type": "timestamp"},
            ]
        },
    },
}


class ExplodingRetriever:
    def query(self, *_args, **_kwargs):
        raise AssertionError("retriever should be bypassed for direct schema-safe queries")


@pytest.mark.parametrize(
    ("question", "table", "extra_sql"),
    [
        ("list all users", "users", None),
        ("list all berth_masters", "berth_masters", None),
        ("list assignments", "assignments", None),
        ("count users", "users", "COUNT(*)"),
        ("show users where role is admin", "users", 'WHERE "users"."role" = \'admin\''),
    ],
)
def test_runtime_bypasses_models_for_generic_single_table_queries(
    question: str,
    table: str,
    extra_sql: str | None,
) -> None:
    result = PredictionOrchestrator().predict(
        question,
        schema=GENERIC_POSTGRES_SCHEMA,
        retriever=ExplodingRetriever(),
    )

    assert result.source_model == "generic_direct_planner"
    assert result.sql is not None
    assert result.validation["is_valid"], result.validation
    assert f'FROM "{table}"' in result.sql
    assert "JOIN" not in result.sql.upper()
    assert "password_hash" not in result.sql
    assert result.query_ir["base_table"] == table
    assert result.query_ir["required_tables"] == [table]
    assert result.query_ir["joins"] == []
    if extra_sql:
        assert extra_sql in result.sql


def test_explicit_join_language_is_not_directly_handled() -> None:
    direct = TableIntentResolver(SchemaProfile(GENERIC_POSTGRES_SCHEMA)).resolve("show assignments with user names")

    assert direct.handled is False
    assert "join" in (direct.reason or "")


def test_service_orders_does_not_enable_sample_retail_mapping() -> None:
    context = RuntimeSchemaContext(NON_RETAIL_SCHEMA)
    mapper = SchemaAwareMapper()

    assert mapper._deterministic_metric("sales", context) is None
    assert mapper._deterministic_metric("revenue", context) is None
    assert mapper._deterministic_filter("status", context) is None

    slots = SlotResolver().resolve_slots(
        "show total cost by terminal",
        {"template_id": "metric_by_dimension"},
        [],
        context,
    )["slots"]
    mapping = mapper.map_slots_to_schema(slots, context, template_id="metric_by_dimension")
    assert slots["metric"]["value"] == "cost"
    assert mapping.metric_table == "service_orders"
    assert mapping.metric_column == "cost"
    assert not mapping.mapping_reasons


def test_generic_join_base_table_has_no_retail_priority() -> None:
    assert RuntimeJoinPlanner.choose_base_table(None, "terminals", ["service_orders", "terminals"]) == "terminals"
