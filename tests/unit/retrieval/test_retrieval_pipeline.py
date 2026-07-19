"""
Purpose: Verifies retrieval unit behaviour consolidated from fragmented test files.
Required because: Retriever runtime, index building and dataset trainer wrappers belong to the retrieval pipeline.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# Source: tests/test_04_retrieval_runtime.py
"""Test 04: Retrieval Runtime — retrieval model, prediction orchestrator, confidence."""


from pathlib import Path

import pytest

from inference.prediction_orchestrator import PredictionOrchestrator
from nl2sql_v1.retriever import TfidfRetriever
from nl2sql_v1.schema import read_sqlite_schema
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from scripts.create_sample_db import build_database
from retrieval.rag_index_builder import RAGIndexBuilder


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def sample_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "sample_retail.db"
    build_database(db_path)
    return db_path


@pytest.fixture()
def schema(sample_db: Path):
    return read_sqlite_schema(sample_db)


@pytest.fixture()
def retriever() -> TfidfRetriever:
    return TfidfRetriever.train(ROOT / "training_data" / "examples.jsonl")


@pytest.fixture()
def model(retriever: TfidfRetriever) -> RetrievalNL2SQLModel:
    metric_synonyms, dimension_synonyms = RetrievalNL2SQLModel._load_synonyms(ROOT / "data" / "synonyms.yaml")
    return RetrievalNL2SQLModel(
        retriever=retriever,
        templates_path=ROOT / "data" / "templates.yaml",
        synonyms_path=ROOT / "data" / "synonyms.yaml",
        metric_synonyms=metric_synonyms,
        dimension_synonyms=dimension_synonyms,
    )


@pytest.fixture()
def compatible_rag_dir(tmp_path: Path) -> Path:
    RAGIndexBuilder().build(
        [{
            "example_id": "users_1",
            "question": "list all users",
            "intent": "show_records",
            "template_id": "show_records",
            "serialized_schema": "users(id, name)",
            "query_ir": {
                "intent": "show_records",
                "template_id": "show_records",
                "base_table": "users",
                "required_tables": ["users"],
                "joins": [],
            },
        }],
        tmp_path,
    )
    return tmp_path


class TestRetrievalNL2SQLModel:
    def test_model_loads(self, compatible_rag_dir: Path) -> None:
        model = RetrievalNL2SQLModel.load(artifact_dir=compatible_rag_dir)
        assert model.retriever is not None

    def test_backward_compat_option_a_param(self, compatible_rag_dir: Path) -> None:
        """The old param names still work."""
        model = RetrievalNL2SQLModel.load(
            artifact_dir=compatible_rag_dir,
            use_option_a_fallback=False,
        )
        assert not model.use_neural_ir_fallback


class TestRetrievalPrediction:
    def test_top_customers_sql(self, model, schema) -> None:
        result = model.predict("Top 5 customers by sales", schema)
        assert result.sql is not None
        assert "SUM" in result.sql
        assert "LIMIT 5" in result.sql
        assert result.validation.get("ok") or result.validation.get("is_valid")

    def test_count_by_status(self, model, schema) -> None:
        result = model.predict("Count orders by status", schema)
        assert result.sql is not None
        assert "COUNT" in result.sql
        assert "status" in result.sql

    def test_date_filter_applied(self, model, schema) -> None:
        result = model.predict("Show revenue last month", schema)
        assert result.sql is not None
        assert "date" in result.sql.lower()

    def test_unknown_question_has_low_confidence(self, model, schema) -> None:
        result = model.predict("xyzzy frobnicate the quux", schema)
        assert result.confidence < 0.50
        assert result.confidence_tier == "low"

    def test_source_model_uses_new_names(self, model, schema) -> None:
        result = model.predict("Top 5 customers by sales", schema)
        assert result.source_model in ("retrieval_ir", "neural_ir", "adaptive_router")

    def test_simple_filter_runtime(self, model, schema) -> None:
        result = model.predict("Orders where status is completed", schema)
        assert result.sql is not None
        assert "status" in result.sql.lower()
        assert "completed" in result.sql.lower()


class TestPredictionOrchestrator:
    def test_backward_compat_init_params(self, tmp_path) -> None:
        orch = PredictionOrchestrator(option_a_model_dir=tmp_path, use_option_a_fallback=False)
        assert orch.neural_ir_model_dir == tmp_path
        assert not orch.use_neural_ir_fallback


class TestConfidenceBreakdown:
    def test_confidence_has_breakdown(self, model, schema) -> None:
        result = model.predict("Top 5 customers by sales", schema)
        breakdown = result.confidence_breakdown
        assert "retrieval" in breakdown or "final" in breakdown

    def test_low_confidence_has_caps(self, model, schema) -> None:
        result = model.predict("xyzzy frobnicate the quux", schema)
        if result.confidence_breakdown.get("caps_applied"):
            assert isinstance(result.confidence_breakdown["caps_applied"], list)


# Source: tests/test_23_retrieval_rag_index.py
from pathlib import Path

from retrieval import ExampleIndex, LocalRAGRetriever, PatternIndex, RetrievalReranker, SchemaIndex
from retrieval.rag_index_builder import RAGIndexBuilder
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from retrieval.artifact_compatibility import validate_sklearn_metadata
import json
import pytest


def _examples() -> list[dict]:
    return [
        {
            "example_id": "show_users",
            "question": "list all users",
            "intent": "show_records",
            "template_id": "show_records",
            "schema": {"tables": {"users": {"columns": {"id": {}, "name": {}}}}},
            "query_ir": {"intent": "show_records", "template_id": "show_records", "base_table": "users", "required_tables": ["users"], "joins": [], "metrics": []},
        },
        {
            "example_id": "top_sales",
            "question": "top customers by sales",
            "intent": "top_n_metric_by_dimension",
            "template_id": "top_n_metric_by_dimension",
            "schema": {"tables": {"customers": {"columns": {"customer_name": {}}}, "orders": {"columns": {"amount": {}}}}},
            "query_ir": {"intent": "top_n_metric_by_dimension", "template_id": "top_n_metric_by_dimension", "base_table": "orders", "required_tables": ["orders", "customers"], "joins": [{"condition": "orders.customer_id = customers.customer_id"}], "metrics": [{"expression": "orders.amount"}]},
        },
    ]


def test_rag_retriever_prioritizes_show_records_for_simple_listing() -> None:
    examples = _examples()
    example_index = ExampleIndex()
    schema_index = SchemaIndex()
    pattern_index = PatternIndex()
    example_index.build(examples)
    schema_index.build(examples)
    pattern_index.build(examples)
    retriever = LocalRAGRetriever(example_index, schema_index, pattern_index, RetrievalReranker())

    result = retriever.retrieve(
        "list all users",
        {"tables": {"users": {"columns": {"id": {}, "name": {}}}, "assignments": {"columns": {"id": {}}}}},
        top_k=2,
    )

    assert result["patterns"][0]["pattern"] == "show_records"
    assert result["reranked"][0]["example_id"] == "show_users"
    assert result["reranked"][0]["intent"] != "top_n_metric_by_dimension"


def test_analytics_question_retrieves_analytics_example() -> None:
    examples = _examples()
    example_index = ExampleIndex()
    schema_index = SchemaIndex()
    pattern_index = PatternIndex()
    example_index.build(examples)
    schema_index.build(examples)
    pattern_index.build(examples)

    result = LocalRAGRetriever(example_index, schema_index, pattern_index, RetrievalReranker()).retrieve(
        "top customers by sales",
        {"tables": {"customers": {"columns": {"customer_name": {}}}, "orders": {"columns": {"amount": {}}}}},
    )

    assert result["reranked"][0]["example_id"] == "top_sales"


def test_runtime_loader_prefers_rag_index_when_present(tmp_path: Path) -> None:
    RAGIndexBuilder().build(_examples(), tmp_path)

    model = RetrievalNL2SQLModel.load(artifact_dir=tmp_path)
    results = model.retriever.query_with_schema(
        "list all users",
        {"tables": {"users": {"columns": {"id": {}, "name": {}}}}},
        top_k=1,
    )

    assert model.metadata["retrieval_backend"] == "local_rag"
    assert results[0].example_id == "show_users"


def test_rag_builder_saves_sklearn_metadata(tmp_path: Path) -> None:
    RAGIndexBuilder().build(_examples(), tmp_path)

    metadata = json.loads((tmp_path / "sklearn_artifact_metadata.json").read_text(encoding="utf-8"))
    assert metadata["sklearn_version"]
    assert metadata["python_version"]
    assert "tfidf_vectorizer" in metadata["artifact_types"]


def test_sklearn_version_mismatch_rebuilds_in_training_and_fails_runtime(tmp_path: Path) -> None:
    RAGIndexBuilder().build(_examples(), tmp_path)
    path = tmp_path / "sklearn_artifact_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["sklearn_version"] = "0.0.incompatible"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    training_check = validate_sklearn_metadata(tmp_path, mode="training")
    assert training_check["rebuild_required"] is True
    with pytest.raises(RuntimeError, match="Incompatible sklearn artifact version"):
        validate_sklearn_metadata(tmp_path, mode="runtime")


# Source: tests/test_train_retriever_from_datasets.py
"""Compatibility helpers for legacy tests."""


from tests.legacy.test_train_retriever_from_datasets import FakeCorpusBuilder

__test__ = False

__all__ = ["FakeCorpusBuilder"]
