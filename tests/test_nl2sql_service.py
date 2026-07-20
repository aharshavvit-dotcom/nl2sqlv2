"""Tests for NL2SQLService facade."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from inference.nl2sql_service import NL2SQLResult, NL2SQLService


class TestNL2SQLResult:
    """Test the unified result dataclass."""

    def test_to_dict_roundtrip(self):
        result = NL2SQLResult(
            question="show sales by region",
            sql="SELECT region, SUM(sales) FROM orders GROUP BY region",
            query_ir={"intent": "metric_by_dimension"},
            confidence=0.85,
            source_model="retrieval",
        )
        d = result.to_dict()
        assert d["question"] == "show sales by region"
        assert d["confidence"] == 0.85
        assert d["source_model"] == "retrieval"
        assert d["abstained"] is False

    def test_abstention_fields(self):
        result = NL2SQLResult(
            question="something unclear",
            sql="",
            query_ir={},
            confidence=0.0,
            abstained=True,
            abstention_reason="No model loaded",
        )
        assert result.abstained is True
        assert result.abstention_reason == "No model loaded"


class TestNL2SQLService:
    """Test the service facade."""

    def test_service_no_models(self):
        """Service with no models returns abstention."""
        service = NL2SQLService()
        assert service.is_ready is False
        assert service.loaded_models == []
        result = service.predict("show sales by region")
        assert result.abstained is True
        assert result.abstention_reason == "No model loaded"

    def test_service_with_retrieval_model(self):
        """Service with retrieval model returns prediction."""
        mock_model = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.model_dump.return_value = {
            "sql": "SELECT * FROM orders",
            "query_ir": {"intent": "show_records"},
            "confidence": 0.75,
        }
        mock_model.predict.return_value = mock_prediction

        service = NL2SQLService(retrieval_model=mock_model)
        assert service.is_ready is True
        assert "retrieval" in service.loaded_models
        result = service.predict("show all orders")
        assert result.sql == "SELECT * FROM orders"
        assert result.confidence == 0.75
        assert result.source_model == "retrieval"

    def test_service_with_orchestrator(self):
        """Service with orchestrator returns prediction."""
        mock_orch = MagicMock()
        mock_orch.predict.return_value = {
            "prediction": {
                "sql": "SELECT region, SUM(sales) FROM orders GROUP BY region",
                "query_ir": {"intent": "metric_by_dimension"},
                "confidence": 0.92,
                "source_model": "neural",
                "confidence_breakdown": {"slot": 0.95, "intent": 0.89},
            },
            "route": "neural",
            "is_safe": True,
            "diagnostics": {},
        }

        service = NL2SQLService(prediction_orchestrator=mock_orch)
        result = service.predict("show total sales by region")
        assert "SUM(sales)" in result.sql
        assert result.confidence == 0.92
        assert result.route == "neural"
        assert result.abstained is False

    def test_abstention_on_low_confidence(self):
        """Service abstains on low confidence."""
        mock_orch = MagicMock()
        mock_orch.predict.return_value = {
            "prediction": {
                "sql": "SELECT * FROM unknown_table",
                "query_ir": {},
                "confidence": 0.10,
                "source_model": "retrieval",
            },
            "route": "retrieval",
            "is_safe": True,
        }
        service = NL2SQLService(
            prediction_orchestrator=mock_orch,
            config={"abstention_threshold": 0.20},
        )
        result = service.predict("what is the meaning of life?")
        assert result.abstained is True
        assert result.abstention_reason == "Low confidence"

    def test_error_handling(self):
        """Service handles errors gracefully."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Model crash")

        service = NL2SQLService(retrieval_model=mock_model)
        result = service.predict("show orders")
        assert result.abstained is True
        assert "Prediction error" in result.abstention_reason
        assert result.latency_ms > 0

    def test_custom_abstention_threshold(self):
        """Custom abstention threshold works."""
        service = NL2SQLService(config={"abstention_threshold": 0.50})
        assert service._abstention_threshold == 0.50
