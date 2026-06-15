"""Test 04: Retrieval Runtime — retrieval model, prediction orchestrator, confidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference.prediction_orchestrator import PredictionOrchestrator
from nl2sql_v1.retriever import TfidfRetriever
from nl2sql_v1.schema import read_sqlite_schema
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from scripts.create_sample_db import build_database


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


class TestRetrievalNL2SQLModel:
    def test_model_loads(self) -> None:
        model = RetrievalNL2SQLModel.load()
        assert model.retriever is not None

    def test_backward_compat_option_a_param(self) -> None:
        """The old param names still work."""
        model = RetrievalNL2SQLModel.load(use_option_a_fallback=False)
        assert not model.use_neural_ir_fallback


class TestRetrievalPrediction:
    def test_top_customers_sql(self, model, schema) -> None:
        result = model.predict("Top 5 customers by sales", schema)
        assert result.sql is not None
        assert "customers" in result.sql.lower()
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
