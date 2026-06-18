"""Full runtime checks for generic PostgreSQL-shaped schemas."""

from __future__ import annotations

import pytest

from generic_planner import SchemaProfile, TableIntentResolver
from inference.prediction_orchestrator import PredictionOrchestrator
from tests.test_10_generic_table_intent import GENERIC_POSTGRES_SCHEMA


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
