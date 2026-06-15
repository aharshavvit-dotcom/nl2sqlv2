from __future__ import annotations

from pathlib import Path

from nl2sql_v1.retriever import TfidfRetriever
from nl2sql_v1.schema import read_sqlite_schema
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from scripts.create_sample_db import build_database


ROOT = Path(__file__).resolve().parents[1]


def test_top_products_revenue_uses_item_level_expression(tmp_path: Path) -> None:
    db_path = tmp_path / "retail.db"
    build_database(db_path)
    schema = read_sqlite_schema(db_path)
    retriever = TfidfRetriever.train(ROOT / "training_data" / "examples.jsonl")
    metric_synonyms, dimension_synonyms = RetrievalNL2SQLModel._load_synonyms(ROOT / "data" / "synonyms.yaml")
    model = RetrievalNL2SQLModel(
        retriever=retriever,
        templates_path=ROOT / "data" / "templates.yaml",
        synonyms_path=ROOT / "data" / "synonyms.yaml",
        metric_synonyms=metric_synonyms,
        dimension_synonyms=dimension_synonyms,
    )

    result = model.predict("Top 5 products by revenue", schema)

    assert result.query_ir is not None
    assert result.query_ir["base_table"] == "order_items"
    assert result.ir_validation and result.ir_validation["is_valid"]
    assert result.validation["is_valid"]
    assert result.sql is not None
    assert "products.product_name AS product" in result.sql
    assert "SUM(order_items.quantity * order_items.price) AS revenue" in result.sql
    assert "order_items.product_id = products.product_id" in result.sql
    assert "SUM(orders.amount)" not in result.sql
