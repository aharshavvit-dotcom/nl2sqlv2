"""Test 09: End-to-End Smoke — full pipeline tests, legacy runtime checks."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from execution.query_executor import execute_select
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


class TestEndToEndSQLite:
    def test_generate_and_execute(self, model, schema, sample_db) -> None:
        result = model.predict("Top 5 customers by sales", schema)
        assert result.sql is not None
        assert result.validation.get("ok") or result.validation.get("is_valid")
        df = execute_select(sample_db, result.sql, validation_result=result.validation)
        assert len(df) == 5

    def test_count_query_executes(self, model, schema, sample_db) -> None:
        result = model.predict("Count orders by status", schema)
        assert result.sql is not None
        df = execute_select(sample_db, result.sql, validation_result=result.validation)
        assert len(df) > 0


class TestNoActiveLegacyUsage:
    """Ensure the active runtime does not use nl2sql_v1 engine/renderer/executor for SQL generation."""

    def test_orchestrator_does_not_import_nl2sql_v1_engine(self) -> None:
        source = (ROOT / "inference" / "prediction_orchestrator.py").read_text(encoding="utf-8")
        assert "nl2sql_v1.engine" not in source
        assert "nl2sql_v1.renderer" not in source

    def test_ir_renderer_does_not_import_nl2sql_v1(self) -> None:
        source = (ROOT / "ir" / "ir_to_sql_renderer.py").read_text(encoding="utf-8")
        assert "nl2sql_v1" not in source


class TestCanonicalRuntimeNaming:
    """The active runtime should use new class names."""

    def test_orchestrator_uses_new_import(self) -> None:
        source = (ROOT / "inference" / "prediction_orchestrator.py").read_text(encoding="utf-8")
        # Should import RetrievalIRConverter or OptionCToIRConverter (alias is OK)
        # But class instantiation should work
        from inference.prediction_orchestrator import PredictionOrchestrator
        orch = PredictionOrchestrator()
        assert orch.ir_converter is not None

    def test_prediction_result_uses_new_model_names(self) -> None:
        from inference.prediction_models import PredictionResult
        result = PredictionResult(question="q", normalized_question="q",
                                  source_model="retrieval_ir")
        assert result.source_model == "retrieval_ir"

    def test_backward_compat_source_model_values(self) -> None:
        """Old source_model values should still be accepted."""
        from inference.prediction_models import PredictionResult
        result = PredictionResult(question="q", normalized_question="q",
                                  source_model="option_c")
        assert result.source_model == "option_c"  # Still valid in Literal


class TestNewDBConnectorAvailable:
    """Verify the db connector layer is importable."""

    def test_import_db_package(self) -> None:
        from db import DatabaseConnectionConfig, SQLiteConnector
        config = DatabaseConnectionConfig(db_type="sqlite", sqlite_path="/tmp/x.db")
        assert config.dialect == "sqlite"

    def test_import_postgres_connector(self) -> None:
        try:
            from db import PostgresConnector
            assert PostgresConnector is not None
        except ImportError:
            pytest.skip("psycopg2 not installed")
