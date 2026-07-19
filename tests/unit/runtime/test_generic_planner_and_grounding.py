"""
Purpose: Verifies runtime unit behaviour consolidated from fragmented test files.
Required because: Schema profiling, generic planning, clarification and grounding form one runtime grounding responsibility.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# Source: tests/test_10_generic_table_intent.py
"""Generic schema-first table intent planning."""


import pytest

from generic_planner import SchemaProfile, TableIntentResolver
from ir.ir_to_sql_renderer import IRToSQLRenderer
from validation.sql_validator import SQLValidator


GENERIC_POSTGRES_SCHEMA = {
    "dialect": "postgres",
    "database": "test_db",
    "schema_name": "public",
    "tables": {
        "users": {
            "columns": [
                {"name": "id", "type": "integer", "is_primary_key": True},
                {"name": "name", "type": "text"},
                {"name": "role", "type": "text"},
                {"name": "created_at", "type": "timestamp"},
                {"name": "password_hash", "type": "text"},
            ],
            "primary_keys": ["id"],
            "foreign_keys": [],
        },
        "berth_masters": {
            "columns": [
                {"name": "id", "type": "integer", "is_primary_key": True},
                {"name": "berth_name", "type": "text"},
                {"name": "berth_code", "type": "text"},
            ],
            "primary_keys": ["id"],
            "foreign_keys": [],
        },
        "assignments": {
            "columns": [
                {"name": "id", "type": "integer", "is_primary_key": True},
                {"name": "user_id", "type": "integer", "is_foreign_key": True},
                {"name": "berth_id", "type": "integer", "is_foreign_key": True},
                {"name": "assigned_date", "type": "date"},
                {"name": "status", "type": "text"},
            ],
            "primary_keys": ["id"],
            "foreign_keys": [
                {"column": "user_id", "references_table": "users", "references_column": "id"},
                {"column": "berth_id", "references_table": "berth_masters", "references_column": "id"},
            ],
        },
    },
    "relationships": [
        {"from_table": "assignments", "from_column": "user_id", "to_table": "users", "to_column": "id"},
        {"from_table": "assignments", "from_column": "berth_id", "to_table": "berth_masters", "to_column": "id"},
    ],
}


@pytest.mark.parametrize(
    ("question", "table"),
    [
        ("list all users", "users"),
        ("show users", "users"),
        ("display users", "users"),
        ("list all berth_masters", "berth_masters"),
        ("list all berth masters", "berth_masters"),
        ("show berth masters", "berth_masters"),
        ("list assignments", "assignments"),
    ],
)
def test_simple_table_questions_build_direct_queryir(question: str, table: str) -> None:
    result = TableIntentResolver(SchemaProfile(GENERIC_POSTGRES_SCHEMA)).resolve(question)

    assert result.handled is True
    assert result.intent == "show_records"
    query_ir = result.query_ir
    assert query_ir.base_table == table
    assert query_ir.required_tables == [table]
    assert query_ir.joins == []
    assert query_ir.metrics == []
    assert query_ir.metadata["source"] == "generic_direct_planner"

    sql = IRToSQLRenderer().render(query_ir, dialect="postgres")
    validation = SQLValidator().validate(sql, schema=GENERIC_POSTGRES_SCHEMA, dialect="postgres")

    assert validation["is_valid"], validation
    assert f'FROM "{table}"' in sql
    assert "JOIN" not in sql.upper()
    assert "LIMIT" in sql.upper()
    assert "SELECT *" not in sql.upper()
    assert "password_hash" not in sql


def test_users_default_projection_is_bounded_and_excludes_audit_columns() -> None:
    result = TableIntentResolver(SchemaProfile(GENERIC_POSTGRES_SCHEMA)).resolve("list all users")

    projected = [item.column for item in result.query_ir.dimensions]
    assert projected == ["id", "name", "role"]
    assert "created_at" not in projected
    assert "password_hash" not in projected
    assert result.query_ir.metadata["projection_mode"] == "list_all_records"
    assert result.query_ir.metadata["default_projection_used"] is True


def test_show_active_users_is_a_simple_filter_without_join() -> None:
    schema = {**GENERIC_POSTGRES_SCHEMA, "tables": {**GENERIC_POSTGRES_SCHEMA["tables"]}}
    schema["tables"]["users"] = {
        **GENERIC_POSTGRES_SCHEMA["tables"]["users"],
        "columns": [*GENERIC_POSTGRES_SCHEMA["tables"]["users"]["columns"], {"name": "status", "type": "text"}],
    }
    result = TableIntentResolver(SchemaProfile(schema)).resolve("show active users")

    assert result.handled is True
    assert result.intent == "simple_filter"
    assert result.query_ir.base_table == "users"
    assert result.query_ir.joins == []
    assert result.query_ir.filters[0].column == "status"
    assert result.query_ir.filters[0].value == "active"


# Source: tests/test_11_generic_join_policy.py
"""Generic join-policy enforcement."""


from generic_planner import JoinPolicy, infer_join_policy
from inference.runtime_join_planner import RuntimeJoinPlanner
from inference.runtime_schema_context import RuntimeSchemaContext
from tests.fixtures.generic_schema import GENERIC_POSTGRES_SCHEMA


def test_join_policy_inference_for_generic_questions() -> None:
    assert infer_join_policy("list all users", "show_records") == JoinPolicy.NONE
    assert infer_join_policy("list all berth_masters", "show_records") == JoinPolicy.NONE
    assert infer_join_policy("show assignments with user names", "show_records") == JoinPolicy.EXPLICIT_ONLY
    assert infer_join_policy("assignments by user", "metric_by_dimension") == JoinPolicy.EXPLICIT_ONLY


def test_join_planner_returns_no_joins_for_none_policy() -> None:
    context = RuntimeSchemaContext(GENERIC_POSTGRES_SCHEMA)
    plan = RuntimeJoinPlanner().plan_joins(
        context,
        base_table="users",
        required_tables=["users", "assignments"],
        join_policy=JoinPolicy.NONE,
    )

    assert plan.join_policy == JoinPolicy.NONE.value
    assert plan.required_tables == ["users"]
    assert plan.join_steps == []
    assert plan.join_clause == ""


# Source: tests/test_60_schema_profiler.py
from semantic_layer.schema_profiler import SchemaProfiler


def generic_schema() -> dict:
    return {
        "dialect": "postgres",
        "tables": {
            "users": {
                "columns": {
                    "id": {"type": "integer", "primary_key": True},
                    "name": {"type": "text"},
                    "role": {"type": "text"},
                    "status": {"type": "text"},
                    "password_hash": {"type": "text"},
                    "created_at": {"type": "timestamp"},
                }
            },
            "berth_masters": {
                "columns": {
                    "id": {"type": "integer", "primary_key": True},
                    "berth_code": {"type": "text"},
                    "berth_name": {"type": "text"},
                    "status": {"type": "text"},
                }
            },
            "assignments": {
                "columns": {
                    "id": {"type": "integer", "primary_key": True},
                    "user_id": {"type": "integer"},
                    "berth_id": {"type": "integer"},
                    "status": {"type": "text"},
                    "assigned_date": {"type": "date"},
                }
            },
        },
        "relationships": [
            {"from_table": "assignments", "from_column": "user_id", "to_table": "users", "to_column": "id"},
            {"from_table": "assignments", "from_column": "berth_id", "to_table": "berth_masters", "to_column": "id"},
        ],
    }


def test_schema_profiler_detects_table_roles_and_sensitive_columns() -> None:
    profile = SchemaProfiler().profile(generic_schema())

    assert profile["tables"]["users"]["table_type"] == "entity"
    assert profile["tables"]["berth_masters"]["table_type"] == "lookup"
    assert profile["tables"]["assignments"]["table_type"] == "bridge"
    assert "password_hash" in profile["tables"]["users"]["sensitive_columns"]
    assert "password_hash" not in profile["tables"]["users"]["safe_columns"]
    assert "role" in profile["tables"]["users"]["likely_filters"]
    assert "created_at" in profile["tables"]["users"]["likely_dates"]


# Source: tests/test_61_glossary_generator.py
from semantic_layer.glossary_generator import GlossaryGenerator
from semantic_layer.schema_profiler import SchemaProfiler
from tests.fixtures.generic_schema import generic_schema


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


# Source: tests/test_62_semantic_mapper.py
from semantic_layer import build_semantic_profile
from semantic_layer.semantic_mapper import SemanticMapper
from tests.fixtures.generic_schema import generic_schema


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


# Source: tests/test_63_ambiguity_detector.py
from clarification.ambiguity_detector import AmbiguityDetector
from semantic_layer import build_semantic_profile
from semantic_layer.semantic_mapper import SemanticMapper
from tests.fixtures.generic_schema import generic_schema


def test_ambiguity_detector_builds_column_mapping_options() -> None:
    schema = generic_schema()
    mapper = SemanticMapper(build_semantic_profile(schema))
    mapping = mapper.map_column("status")

    ambiguity = AmbiguityDetector().detect("show status", mapping, schema)

    assert ambiguity["ambiguous"] is True
    assert ambiguity["ambiguity_type"] == "column_mapping"
    assert {option["value"] for option in ambiguity["options"]} >= {"users.status", "assignments.status", "berth_masters.status"}


# Source: tests/test_64_clarification_runtime.py
from inference.prediction_orchestrator import PredictionOrchestrator
from tests.fixtures.generic_schema import generic_schema


class DummyRetriever:
    def query(self, text: str, top_k: int = 3) -> list:
        return []

    def query_with_schema(self, text: str, schema: dict, top_k: int = 3) -> list:
        return []


def test_runtime_clarification_blocks_sql_for_ambiguous_status() -> None:
    result = PredictionOrchestrator(use_neural_ir_fallback=False).predict("show status", generic_schema(), DummyRetriever())

    assert result.needs_clarification is True
    assert result.sql is None
    assert result.query_ir is None
    assert "users.status" in result.clarification["options"]


def test_direct_table_query_uses_direct_planner_with_no_join_and_safe_columns() -> None:
    result = PredictionOrchestrator(use_neural_ir_fallback=False).predict("list all users", generic_schema(), DummyRetriever())

    assert result.needs_clarification is False
    assert result.source_model == "generic_direct_planner"
    assert "JOIN" not in (result.sql or "").upper()
    assert '"password_hash"' not in (result.sql or "")


# Source: tests/test_120_schema_value_index.py
from unittest.mock import MagicMock
import pytest
from inference.runtime_schema_context import RuntimeSchemaContext
from inference.grounding.schema_value_index import SchemaValueIndex, ValueIndexMode


def test_sensitive_column_explicitly_excluded():
    ctx = MagicMock(spec=RuntimeSchemaContext)
    ctx.get_tables.return_value = ["users"]
    ctx.get_table_columns.return_value = ["password", "username"]
    ctx.get_columns.return_value = ["users.password", "users.username"]

    def col_info(table, column):
        if column == "password":
            return {"is_sensitive": True, "sample_values": ["secret123"]}
        return {"is_sensitive": False, "sample_values": ["john_doe"]}
    ctx.column_info.side_effect = col_info

    idx = SchemaValueIndex(ctx, mode=ValueIndexMode.APPROVED_DOMAIN_VALUES)
    assert "secret123" not in idx.index
    assert "john doe" in idx.index


def test_sensitive_column_inferred_from_name():
    ctx = MagicMock(spec=RuntimeSchemaContext)
    ctx.get_tables.return_value = ["users"]
    ctx.get_table_columns.return_value = ["ssn", "name"]
    ctx.get_columns.return_value = ["users.ssn", "users.name"]

    def col_info(table, column):
        if column == "ssn":
            return {"is_sensitive": False, "sample_values": ["123-456-7890"]}
        return {"is_sensitive": False, "sample_values": ["Jane"]}
    ctx.column_info.side_effect = col_info

    idx = SchemaValueIndex(ctx, mode=ValueIndexMode.APPROVED_DOMAIN_VALUES)
    assert "123 456 7890" not in idx.index
    assert "jane" in idx.index


def test_disabled_mode_performs_no_lookup():
    ctx = MagicMock(spec=RuntimeSchemaContext)
    ctx.get_tables.return_value = ["users"]
    ctx.get_columns.return_value = ["users.name"]
    ctx.column_info.return_value = {"is_sensitive": False, "sample_values": ["Jane"]}

    idx = SchemaValueIndex(ctx, mode=ValueIndexMode.DISABLED)
    assert not idx.index
    assert idx.lookup_value("Jane") == []


# Source: tests/test_121_filter_value_extractor.py
from datetime import datetime
from unittest.mock import MagicMock
import pytest
from inference.grounding.filter_value_contract import QueryTimeContext
from inference.grounding.filter_value_extractor import FilterValueExtractor
from inference.grounding.schema_value_index import SchemaValueIndex


def test_negative_number_extraction():
    idx = MagicMock(spec=SchemaValueIndex)
    extractor = FilterValueExtractor(idx)

    contract = extractor.extract_literals("temperature was below -15 degrees")
    literals = {l.normalized_value for l in contract.extracted_literals}
    assert -15.0 in literals or -15 in literals


def test_relative_date_with_time_context():
    idx = MagicMock(spec=SchemaValueIndex)
    extractor = FilterValueExtractor(idx)

    time_ctx = QueryTimeContext(current_datetime=datetime(2026, 7, 9))
    contract = extractor.extract_literals("orders placed yesterday", time_context=time_ctx)
    yesterday_lit = [l for l in contract.extracted_literals if l.raw_text == "yesterday"]
    assert len(yesterday_lit) == 1
    assert yesterday_lit[0].normalized_value == "2026-07-08"


def test_list_extraction():
    idx = MagicMock(spec=SchemaValueIndex)
    extractor = FilterValueExtractor(idx)

    contract = extractor.extract_literals("customers in India, Japan, or Singapore")
    list_lits = [l for l in contract.extracted_literals if l.value_type == "list"]
    assert len(list_lits) == 1
    assert list_lits[0].normalized_value == ["India", "Japan", "Singapore"]


# Source: tests/test_122_filter_grounding.py
from unittest.mock import MagicMock
import pytest
from inference.grounding.filter_value_contract import ExtractedLiteral
from inference.grounding.filter_value_extractor import FilterValueExtractionContract
from inference.grounding.filter_grounding_service import FilterGroundingService
from inference.grounding.schema_value_index import SchemaValueIndex
from inference.runtime_schema_context import RuntimeSchemaContext


def test_join_graph_relevance_resolves_conflict():
    val_idx = MagicMock(spec=SchemaValueIndex)
    val_idx.lookup_value.return_value = [
        {"table": "customers", "column": "customers.region", "score": 0.94, "signals": {"exact_value_match": 1.0}},
        {"table": "suppliers", "column": "suppliers.region", "score": 0.94, "signals": {"exact_value_match": 1.0}},
    ]

    ctx = MagicMock(spec=RuntimeSchemaContext)
    ctx.foreign_keys = [{"child_table": "orders", "parent_table": "customers"}]
    ctx.get_columns.return_value = ["customers.region", "suppliers.region"]
    ctx.column_info.return_value = {"is_sensitive": False}

    service = FilterGroundingService(val_idx, ctx)

    contract = FilterValueExtractionContract(
        raw_question="show orders from customers in the west",
        extracted_literals=[
            ExtractedLiteral(
                literal_id="lit_0",
                raw_text="west",
                normalized_value="west",
                value_type="string",
                span_start=34,
                span_end=38,
                extraction_method="quoted_string",
                extraction_confidence=0.9,
            )
        ],
    )

    res = service.ground_filters("show orders from customers in the west", contract, entity_table="orders")

    assert res[0].selected_candidate is not None
    assert res[0].selected_candidate.table_name == "customers"
    assert res[0].selected_candidate.column_name == "region"


def test_ambiguity_requires_clarification():
    val_idx = MagicMock(spec=SchemaValueIndex)
    val_idx.lookup_value.return_value = [
        {"table": "employees", "column": "employees.account_status", "score": 0.85, "signals": {"exact_value_match": 1.0}},
        {"table": "employees", "column": "employees.employment_status", "score": 0.84, "signals": {"exact_value_match": 1.0}},
    ]

    ctx = MagicMock(spec=RuntimeSchemaContext)
    ctx.foreign_keys = []
    ctx.get_columns.return_value = ["employees.account_status", "employees.employment_status"]
    ctx.column_info.return_value = {"is_sensitive": False}

    service = FilterGroundingService(val_idx, ctx)

    contract = FilterValueExtractionContract(
        raw_question="show active employees",
        extracted_literals=[
            ExtractedLiteral(
                literal_id="lit_0",
                raw_text="active",
                normalized_value="active",
                value_type="string",
                span_start=5,
                span_end=11,
                extraction_method="quoted_string",
                extraction_confidence=0.9,
            )
        ],
    )

    res = service.ground_filters("show active employees", contract, entity_table="employees")
    assert res[0].requires_clarification is True
    assert "Does 'active' refer to" in res[0].clarification_question


# Source: tests/test_123_projection_resolution.py
from unittest.mock import MagicMock
import pytest
from inference.runtime_schema_context import RuntimeSchemaContext
from inference.grounding.projection_resolver import ProjectionResolver


def test_count_projection_only():
    ctx = MagicMock(spec=RuntimeSchemaContext)
    resolver = ProjectionResolver(ctx)
    res = resolver.resolve_projection("count the number of orders", entity_table="orders")
    assert res.projection_mode == "count-only"
    assert res.selected_columns == ["*"]


def test_specific_projection_only():
    ctx = MagicMock(spec=RuntimeSchemaContext)
    ctx.get_table_columns.return_value = ["id", "customer_name", "secret_key"]

    def col_info(table, col):
        return {"is_sensitive": col == "secret_key"}
    ctx.column_info.side_effect = col_info
    ctx.foreign_keys = []

    resolver = ProjectionResolver(ctx)
    res = resolver.resolve_projection("show customer name of orders", entity_table="orders")
    assert res.projection_mode == "specific-column"
    assert "orders.customer_name" in res.selected_columns
    assert "orders.secret_key" not in res.selected_columns


# Source: tests/test_124_dimension_resolution.py
from unittest.mock import MagicMock
import pytest
from inference.runtime_schema_context import RuntimeSchemaContext
from inference.grounding.dimension_resolver import DimensionResolver


def test_grouping_dimension():
    ctx = MagicMock(spec=RuntimeSchemaContext)
    ctx.get_columns.return_value = ["orders.status", "orders.id"]
    ctx.column_info.return_value = {"is_sensitive": False}

    resolver = DimensionResolver(ctx)
    res = resolver.resolve_dimension("count orders grouped by status", "status", active_table="orders")
    assert res["role"] == "grouping"
    assert res["column"] == "status"
    assert res["table"] == "orders"


def test_dimension_unreachable_penalty():
    ctx = MagicMock(spec=RuntimeSchemaContext)
    ctx.get_columns.return_value = ["customers.region", "suppliers.region"]
    ctx.column_info.return_value = {"is_sensitive": False}
    ctx.foreign_keys = []

    resolver = DimensionResolver(ctx)
    res = resolver.resolve_dimension("sales by region", "region", active_table="customers")
    assert res["table"] == "customers"
    assert res["column"] == "region"
