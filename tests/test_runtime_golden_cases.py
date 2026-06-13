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
def model() -> RetrievalNL2SQLModel:
    retriever = TfidfRetriever.train(ROOT / "training_data" / "examples.jsonl")
    metric_synonyms, dimension_synonyms = RetrievalNL2SQLModel._load_synonyms(ROOT / "data" / "synonyms.yaml")
    return RetrievalNL2SQLModel(
        retriever=retriever,
        templates_path=ROOT / "data" / "templates.yaml",
        synonyms_path=ROOT / "data" / "synonyms.yaml",
        metric_synonyms=metric_synonyms,
        dimension_synonyms=dimension_synonyms,
    )


@pytest.mark.parametrize(
    ("question", "contains"),
    [
        (
            "Top 5 customers by sales",
            [
                "customers.customer_name",
                "SUM(orders.amount)",
                "JOIN customers",
                "GROUP BY customers.customer_name",
                "ORDER BY revenue DESC",
                "LIMIT 5",
            ],
        ),
        (
            "Show revenue by region",
            ["customers.region", "SUM(orders.amount)", "JOIN customers", "GROUP BY customers.region"],
        ),
        (
            "Count orders by status",
            ["orders.status", "COUNT(*)", "GROUP BY orders.status"],
        ),
        (
            "Show sales by month",
            ["strftime('%Y-%m', orders.order_date)", "SUM(orders.amount)", "GROUP BY", "ORDER BY"],
        ),
        (
            "Top 5 products by revenue",
            ["products.product_name", "SUM", "JOIN order_items", "JOIN products", "GROUP BY products.product_name", "LIMIT 5"],
        ),
        (
            "Sales last month",
            ["orders.order_date >=", "orders.order_date <", "SUM(orders.amount)"],
        ),
        (
            "Orders where status is completed",
            ["orders.status = 'completed'"],
        ),
    ],
)
def test_runtime_golden_sql_shapes(
    model: RetrievalNL2SQLModel,
    schema: SchemaGraph,
    sample_db: Path,
    question: str,
    contains: list[str],
) -> None:
    result = model.predict(question, schema)

    assert result.query_ir is not None
    assert result.ir_validation and result.ir_validation["is_valid"]
    assert result.validation["is_valid"]
    assert result.sql is not None
    for fragment in contains:
        assert fragment in result.sql

    df = execute_select(sample_db, result.sql, validation_result=result.validation)
    assert len(df) >= 0
    assert len(df.columns) >= 1
