"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from pathlib import Path

from inference.prediction_orchestrator import PredictionOrchestrator
from nl2sql_v1.retriever import TfidfRetriever
from nl2sql_v1.schema import read_sqlite_schema
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from scripts.create_sample_db import build_database


ROOT = Path(__file__).resolve().parents[1]


def test_prediction_orchestrator_uses_query_ir_pipeline(tmp_path: Path) -> None:
    db_path = tmp_path / "retail.db"
    build_database(db_path)
    schema = read_sqlite_schema(db_path)
    retriever = TfidfRetriever.train(ROOT / "training_data" / "examples.jsonl")
    metric_synonyms, dimension_synonyms = RetrievalNL2SQLModel._load_synonyms(ROOT / "data" / "synonyms.yaml")

    result = PredictionOrchestrator().predict(
        "Top 5 customers by sales",
        schema=schema,
        retriever=retriever,
        metric_synonyms=metric_synonyms,
        dimension_synonyms=dimension_synonyms,
    )

    assert result.query_ir is not None
    assert result.query_ir["template_id"] == "top_n_metric_by_dimension"
    assert result.ir_validation and result.ir_validation["is_valid"]
    assert result.validation["is_valid"]
    assert "SUM(orders.amount) AS revenue" in (result.sql or "")
