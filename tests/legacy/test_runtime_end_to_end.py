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


def test_runtime_end_to_end_executes_top_customers(tmp_path: Path) -> None:
    db_path = tmp_path / "sample_retail.db"
    model_path = tmp_path / "sample_model.joblib"
    build_database(db_path)
    schema = read_sqlite_schema(db_path)
    root = Path(__file__).resolve().parents[1]
    model = RetrievalNL2SQLModel.load(
        artifact_dir=tmp_path / "missing_artifact",
        sample_model_path=model_path,
        sample_examples_path=root / "training_data" / "examples.jsonl",
        templates_path=root / "data" / "templates.yaml",
        synonyms_path=root / "data" / "synonyms.yaml",
    )

    result = model.predict("Top 5 customers by sales", schema)

    assert result.query_ir is not None
    assert result.ir_validation and result.ir_validation["is_valid"]
    assert result.validation["is_valid"]
    df = execute_select(db_path, result.sql or "", validation_result=result.validation)
    assert len(df) == 5
