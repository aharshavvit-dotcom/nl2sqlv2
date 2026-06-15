from __future__ import annotations

from pathlib import Path

import pytest

from execution.query_executor import execute_select
from nl2sql_v1.retriever import TfidfRetriever
from nl2sql_v1.schema import SchemaGraph, read_sqlite_schema
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from scripts.create_sample_db import build_database


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def sample_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "sample_retail.db"
    build_database(db_path)
    return db_path


@pytest.fixture()
def schema(sample_db: Path) -> SchemaGraph:
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


def test_top_customers_sql(model: RetrievalNL2SQLModel, schema: SchemaGraph) -> None:
    result = model.predict("Top 5 customers by sales", schema)

    assert result.sql is not None
    assert "customers" in result.sql.lower()
    assert "SUM" in result.sql
    assert "LIMIT 5" in result.sql
    assert result.validation["ok"] is True


def test_count_by_status(model: RetrievalNL2SQLModel, schema: SchemaGraph) -> None:
    result = model.predict("Count orders by status", schema)

    assert result.sql is not None
    assert "COUNT" in result.sql
    assert "status" in result.sql
    assert result.validation["ok"] is True


def test_date_filter_applied(model: RetrievalNL2SQLModel, schema: SchemaGraph) -> None:
    result = model.predict("Show revenue last month", schema)

    assert result.sql is not None
    assert "date" in result.sql.lower()
    assert result.validation["ok"] is True


def test_unknown_question_has_low_confidence(model: RetrievalNL2SQLModel, schema: SchemaGraph) -> None:
    result = model.predict("xyzzy frobnicate the quux", schema)

    assert result.confidence < 0.50
    assert result.confidence_tier == "low"


def test_end_to_end_executes_on_sample_db(model: RetrievalNL2SQLModel, schema: SchemaGraph, sample_db: Path) -> None:
    result = model.predict("Top 5 customers by sales", schema)

    assert result.sql is not None
    df = execute_select(sample_db, result.sql, validation_result=result.validation)
    assert len(df) == 5
