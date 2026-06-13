from __future__ import annotations

from pathlib import Path

import pytest

from inference.candidate_generator import CandidateGenerator
from inference.candidate_reranker import CandidateReranker
from inference.prediction_confidence import PredictionConfidenceCalculator
from inference.prediction_models import RetrievedCandidate
from inference.prediction_orchestrator import PredictionOrchestrator
from inference.runtime_join_planner import RuntimeJoinPlanner
from inference.runtime_schema_context import RuntimeSchemaContext
from inference.schema_aware_mapper import SchemaAwareMapper
from inference.slot_resolver import SlotResolver
from inference.template_selector import TemplateSelector
from nl2sql_v1.retriever import RetrievalResult
from nl2sql_v1.retriever import TfidfRetriever
from nl2sql_v1.schema import read_sqlite_schema
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from scripts.create_sample_db import build_database


class FakeRetriever:
    def __init__(self, rows: list[dict[str, object]]):
        self.rows = rows

    def query(self, text: str, top_k: int = 10) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                example_id=str(row["id"]),
                question=str(row["question"]),
                score=float(row["score"]),
                template_id=str(row["template_id"]),
                example=row,
            )
            for row in self.rows[:top_k]
        ]


@pytest.fixture()
def runtime_context(tmp_path: Path) -> RuntimeSchemaContext:
    db_path = tmp_path / "retail.db"
    build_database(db_path)
    return RuntimeSchemaContext(read_sqlite_schema(db_path))


@pytest.fixture()
def fake_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "ex-count-status",
            "question": "Count orders by status",
            "template_id": "count_dimension",
            "metric": "order_count",
            "dimension": "status",
            "limit": 100,
            "score": 0.84,
        },
        {
            "id": "ex-top-customer",
            "question": "Top 5 customers by sales",
            "template_id": "rank_dimension",
            "metric": "sales",
            "dimension": "customer",
            "limit": 5,
            "score": 0.72,
        },
        {
            "id": "ex-product-sales",
            "question": "Top 5 products by sales",
            "template_id": "rank_dimension",
            "metric": "sales",
            "dimension": "product",
            "limit": 5,
            "score": 0.68,
        },
    ]


def test_runtime_schema_context_marks_columns_and_relationships(runtime_context: RuntimeSchemaContext) -> None:
    assert runtime_context.has_column("orders", "status")
    assert "orders.amount" in runtime_context.get_numeric_columns()
    assert "orders.status" in runtime_context.get_text_columns()
    assert "orders.order_date" in runtime_context.get_date_columns()
    assert any(edge["neighbor"] == "customers" for edge in runtime_context.relationships["orders"])


def test_candidate_generation_reranking_and_template_selection(
    runtime_context: RuntimeSchemaContext,
    fake_rows: list[dict[str, object]],
) -> None:
    candidates = CandidateGenerator().generate_candidates("Count orders by status", FakeRetriever(fake_rows), top_k=3)
    assert candidates[0].example_id == "ex-count-status"
    assert candidates[0].slots["dimension"] == "status"

    reranked = CandidateReranker().rerank_candidates("Count orders by status", candidates, runtime_context)
    assert reranked[0].template_id == "count_dimension"
    assert reranked[0].rerank_score is not None

    selected = TemplateSelector().select_template(reranked, "Count orders by status")
    assert selected["template_id"] == "count_by_dimension"
    assert selected["intent"] == "count_by_dimension"


def test_slot_resolver_and_schema_mapper_resolve_status_grouping(
    runtime_context: RuntimeSchemaContext,
    fake_rows: list[dict[str, object]],
) -> None:
    candidates = CandidateGenerator().generate_candidates("Count orders by status", FakeRetriever(fake_rows), top_k=3)
    selected = {"template_id": "count_by_dimension", "confidence": 0.8}

    slots = SlotResolver().resolve_slots("Count orders by status", selected, candidates, runtime_context)["slots"]
    assert slots["metric"]["value"] == "order_count"
    assert slots["dimension"]["value"] == "status"

    mapping = SchemaAwareMapper().map_slots_to_schema(slots, runtime_context)
    assert mapping.metric_table == "orders"
    assert mapping.metric_column == "order_id"
    assert mapping.dimension_table == "orders"
    assert mapping.dimension_column == "status"


def test_runtime_join_planner_bridges_orders_to_products(runtime_context: RuntimeSchemaContext) -> None:
    plan = RuntimeJoinPlanner().plan_joins(runtime_context, base_table="orders", required_tables=["orders", "products"])

    assert plan.confidence == 1.0
    assert "JOIN order_items" in plan.join_clause
    assert "JOIN products" in plan.join_clause
    assert "order_items.order_id = orders.order_id" in plan.join_clause
    assert "order_items.product_id = products.product_id" in plan.join_clause


def test_prediction_confidence_calculator_uses_validation_and_mapping() -> None:
    candidate = RetrievedCandidate(
        rank=1,
        example_id="ex1",
        question="Count orders by status",
        template_id="count_by_dimension",
        slots={"metric": "order_count", "dimension": "status"},
        similarity_score=0.9,
        rerank_score=0.88,
    )
    mapping = type(
        "Mapping",
        (),
        {
            "metric_table": "orders",
            "dimension_table": "orders",
            "match_scores": {"metric": 0.95, "dimension": 1.0, "entity": 1.0, "date": 1.0},
        },
    )()

    confidence = PredictionConfidenceCalculator().calculate(
        {
            "candidates": [candidate],
            "selected_template": {"template_id": "count_by_dimension", "confidence": 0.85},
            "slots": {
                "metric": {"confidence": 0.9},
                "dimension": {"confidence": 0.9},
                "entity": {"confidence": 0.8},
            },
            "schema_mapping": mapping,
            "join_plan": {"warnings": []},
            "ir_validation": {"is_valid": True},
            "validation": {"is_valid": True, "ok": True},
        }
    )

    assert confidence["confidence"] >= 0.8
    assert confidence["confidence_tier"] == "high"


def test_prediction_orchestrator_generates_valid_count_by_status_sql(
    tmp_path: Path,
    fake_rows: list[dict[str, object]],
) -> None:
    db_path = tmp_path / "retail.db"
    build_database(db_path)
    schema = read_sqlite_schema(db_path)

    result = PredictionOrchestrator(top_k=3).predict(
        "Count orders by status",
        schema=schema,
        retriever=FakeRetriever(fake_rows),
    )

    assert result.template_id == "count_by_dimension"
    assert result.validation["ok"]
    assert result.sql is not None
    assert "orders.status AS status" in result.sql
    assert "COUNT(*) AS record_count" in result.sql
    assert "GROUP BY orders.status" in result.sql
    assert result.confidence_tier in {"medium", "high"}


def test_prediction_orchestrator_generates_valid_joined_product_sql(
    tmp_path: Path,
    fake_rows: list[dict[str, object]],
) -> None:
    db_path = tmp_path / "retail.db"
    build_database(db_path)
    schema = read_sqlite_schema(db_path)

    result = PredictionOrchestrator(top_k=3).predict(
        "Top 5 products by sales",
        schema=schema,
        retriever=FakeRetriever(fake_rows),
    )

    assert result.template_id == "top_n_metric_by_dimension"
    assert result.validation["ok"]
    assert result.sql is not None
    assert "products.product_name AS product" in result.sql
    assert "SUM(orders.amount) AS revenue" in result.sql
    assert "JOIN order_items" in result.sql
    assert "JOIN products" in result.sql
    assert "LIMIT 5" in result.sql


def test_orchestrator_with_real_retriever_and_sample_db(tmp_path: Path) -> None:
    db_path = tmp_path / "retail.db"
    build_database(db_path)
    schema = read_sqlite_schema(db_path)
    retriever = TfidfRetriever.train(Path(__file__).resolve().parents[1] / "training_data" / "examples.jsonl")
    metric_synonyms, dimension_synonyms = RetrievalNL2SQLModel._load_synonyms(
        Path(__file__).resolve().parents[1] / "data" / "synonyms.yaml"
    )

    result = PredictionOrchestrator().predict(
        "Top 5 customers by sales",
        schema=schema,
        retriever=retriever,
        metric_synonyms=metric_synonyms,
        dimension_synonyms=dimension_synonyms,
    )

    assert result.sql is not None
    assert result.validation["ok"] is True
