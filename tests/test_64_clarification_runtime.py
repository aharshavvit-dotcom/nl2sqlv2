from __future__ import annotations

from inference.prediction_orchestrator import PredictionOrchestrator
from tests.test_60_schema_profiler import generic_schema


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
