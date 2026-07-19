"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from pathlib import Path

from execution.query_executor import execute_select
from nl2sql_v1.schema import read_sqlite_schema
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from scripts.create_sample_db import build_database


ROOT = Path(__file__).resolve().parents[1]


def test_last_month_filter_is_represented_in_query_ir_and_sql(tmp_path: Path) -> None:
    db_path = tmp_path / "retail.db"
    build_database(db_path)
    schema = read_sqlite_schema(db_path)
    model = RetrievalNL2SQLModel.load(
        artifact_dir=tmp_path / "missing_artifact",
        sample_model_path=tmp_path / "sample_model.joblib",
        sample_examples_path=ROOT / "training_data" / "examples.jsonl",
        templates_path=ROOT / "data" / "templates.yaml",
        synonyms_path=ROOT / "data" / "synonyms.yaml",
    )

    result = model.predict("Sales last month", schema)

    assert result.query_ir is not None
    assert result.ir_validation and result.ir_validation["is_valid"]
    assert result.validation["is_valid"]
    assert result.query_ir["date_filters"]
    assert result.query_ir["date_filters"][0]["date_expression"] == "orders.order_date"
    assert result.sql is not None
    assert "orders.order_date >=" in result.sql
    assert "orders.order_date <" in result.sql
    execute_select(db_path, result.sql, validation_result=result.validation)

