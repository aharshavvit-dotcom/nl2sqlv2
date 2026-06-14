from __future__ import annotations

from pathlib import Path

import pytest

from execution.query_executor import execute_select
from nl2sql_v1.schema import read_sqlite_schema
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from scripts.create_sample_db import build_database


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def runtime(tmp_path: Path) -> tuple[Path, object, RetrievalNL2SQLModel]:
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
    return db_path, schema, model


@pytest.mark.parametrize(
    ("question", "filter_expression", "sql_fragment"),
    [
        ("Orders where status is completed", "orders.status", "orders.status = 'completed'"),
        ("Show sales for region west", "customers.region", "customers.region = 'west'"),
        ("Show products in category electronics", "products.category", "products.category = 'electronics'"),
    ],
)
def test_simple_filter_patterns_generate_safe_executable_sql(
    runtime: tuple[Path, object, RetrievalNL2SQLModel],
    question: str,
    filter_expression: str,
    sql_fragment: str,
) -> None:
    db_path, schema, model = runtime

    result = model.predict(question, schema)

    assert result.query_ir is not None
    assert result.ir_validation and result.ir_validation["is_valid"]
    assert result.validation["is_valid"]
    assert result.sql is not None
    assert "SELECT *" not in result.sql
    assert any(item["expression"] == filter_expression for item in result.query_ir["filters"])
    assert sql_fragment in result.sql
    execute_select(db_path, result.sql, validation_result=result.validation)

