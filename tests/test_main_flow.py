from __future__ import annotations

import json
import py_compile
from pathlib import Path

import pytest
import yaml

from execution.query_executor import execute_select
from nl2sql_v1.feedback import append_feedback
from nl2sql_v1.retriever import TfidfRetriever, load_examples
from nl2sql_v1.schema import read_sqlite_schema
from nl2sql_v1.schema_matcher import SchemaMatcher
from nl2sql_v1.slot_extractor import SlotExtractor
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from scripts.create_sample_db import build_database
from validation.sql_validator import SQLValidator


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def sample_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "retail.db"
    build_database(db_path)
    return db_path


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


def test_examples_file_has_80_plus_rows() -> None:
    examples = load_examples(ROOT / "training_data" / "examples.jsonl")
    assert len(examples) >= 80


def test_streamlit_app_path_compiles() -> None:
    py_compile.compile(str(ROOT / "app" / "streamlit_app.py"), doraise=True)


def test_templates_file_has_8_templates() -> None:
    with (ROOT / "data" / "templates.yaml").open("r", encoding="utf-8") as fh:
        templates = yaml.safe_load(fh)["templates"]
    canonical = {
        "top_n_metric_by_dimension",
        "bottom_n_metric_by_dimension",
        "metric_by_dimension",
        "metric_summary",
        "count_records",
        "count_by_dimension",
        "trend_by_date",
        "simple_filter",
        "show_records",
    }
    assert canonical.issubset(templates)


def test_main_question_generates_expected_sql_shape(sample_db: Path, model: RetrievalNL2SQLModel) -> None:
    schema = read_sqlite_schema(sample_db)
    result = model.predict("Top 5 customers by sales", schema)

    assert result.query_ir is not None
    assert result.ir_validation and result.ir_validation["is_valid"]
    assert result.validation["is_valid"]
    assert "customers.customer_name" in result.sql
    assert "SUM(orders.amount) AS revenue" in result.sql
    assert "JOIN customers" in result.sql
    assert "orders.customer_id = customers.customer_id" in result.sql
    assert "GROUP BY customers.customer_name" in result.sql
    assert "ORDER BY revenue DESC" in result.sql
    assert "LIMIT 5" in result.sql

    df = execute_select(sample_db, result.sql, validation_result=result.validation)
    assert list(df.columns) == ["customer", "revenue"]
    assert len(df) == 5


def test_join_resolver_can_bridge_orders_to_products(sample_db: Path, model: RetrievalNL2SQLModel) -> None:
    schema = read_sqlite_schema(sample_db)
    result = model.predict("Top 5 products by sales", schema)

    assert "SUM(order_items.quantity * order_items.price) AS revenue" in result.sql
    assert "SUM(orders.amount)" not in result.sql
    assert "JOIN products" in result.sql
    assert "order_items.product_id = products.product_id" in result.sql
    assert result.validation["is_valid"]


def test_public_corpus_fallback_metric_star_is_ignored() -> None:
    matcher = SchemaMatcher.from_yaml(ROOT / "data" / "synonyms.yaml")
    extractor = SlotExtractor(matcher.catalog)

    slots = extractor.extract(
        "Top 5 customers by sales",
        fallback={
            "template_id": "count_records",
            "metric": "*",
            "dimension": "CustomerID",
            "filters": {"unknown_filter": "x"},
        },
    )

    assert slots.metric == "sales"
    assert slots.dimension == "customer"
    assert slots.filters == {}


def test_sql_validator_rejects_mutation(sample_db: Path) -> None:
    schema = read_sqlite_schema(sample_db)
    result = SQLValidator().validate("DELETE FROM customers", schema=schema)
    assert not result["is_valid"]


def test_executor_rejects_non_select(sample_db: Path) -> None:
    with pytest.raises(ValueError):
        execute_select(sample_db, "DROP TABLE customers")


def test_feedback_is_saved(tmp_path: Path) -> None:
    target = tmp_path / "feedback.jsonl"
    append_feedback(target, {"question": "Top 5 customers by sales", "rating": "thumbs_up"})
    line = target.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["question"] == "Top 5 customers by sales"
    assert payload["rating"] == "thumbs_up"
    assert "created_at" in payload
