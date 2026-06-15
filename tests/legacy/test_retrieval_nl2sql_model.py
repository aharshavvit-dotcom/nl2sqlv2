from __future__ import annotations

from pathlib import Path

from nl2sql_v1.schema import read_sqlite_schema
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from scripts.create_sample_db import build_database


ROOT = Path(__file__).resolve().parents[1]


def test_retrieval_model_predicts_with_sample_fallback(tmp_path: Path) -> None:
    db_path = tmp_path / "retail.db"
    build_database(db_path)

    model = RetrievalNL2SQLModel.load(
        artifact_dir=tmp_path / "missing_artifact",
        sample_model_path=tmp_path / "sample_model.joblib",
        sample_examples_path=ROOT / "training_data" / "examples.jsonl",
        templates_path=ROOT / "data" / "templates.yaml",
        synonyms_path=ROOT / "data" / "synonyms.yaml",
    )
    result = model.predict("Top 5 customers by sales", read_sqlite_schema(db_path))

    assert result.sql is not None
    assert "SUM(orders.amount) AS revenue" in result.sql
    assert result.template_id == "top_n_metric_by_dimension"
    assert result.validation["ok"]
