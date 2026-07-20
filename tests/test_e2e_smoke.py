"""End-to-end smoke tests for the NL2SQL system.

Tests the full pipeline from question → NL2SQLService → SQL output,
ensuring all layers integrate correctly. Uses mock models to avoid
requiring a trained model bundle.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestNL2SQLServiceE2E:
    """End-to-end tests for the NL2SQLService facade."""

    def test_service_creates_without_bundle(self):
        """Service instantiates even without a bundle directory."""
        from inference.nl2sql_service import NL2SQLService
        service = NL2SQLService()
        assert service.is_ready is False
        result = service.predict("show sales")
        assert result.abstained is True
        assert result.latency_ms >= 0

    def test_service_with_mock_retrieval(self):
        """Service works end-to-end with a mock retrieval model."""
        from inference.nl2sql_service import NL2SQLService

        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "sql": "SELECT region, SUM(sales) FROM orders GROUP BY region LIMIT 100",
            "query_ir": {"intent": "metric_by_dimension", "base_table": "orders"},
            "confidence": 0.85,
        }
        mock_model.predict.return_value = mock_result

        service = NL2SQLService(retrieval_model=mock_model)
        result = service.predict("show total sales by region")

        assert result.sql == "SELECT region, SUM(sales) FROM orders GROUP BY region LIMIT 100"
        assert result.confidence == 0.85
        assert result.abstained is False
        assert result.latency_ms > 0

    def test_result_serializes_to_dict(self):
        """NL2SQLResult.to_dict() produces valid dict for API responses."""
        from inference.nl2sql_service import NL2SQLResult

        result = NL2SQLResult(
            question="test",
            sql="SELECT 1",
            query_ir={"intent": "show_records"},
            confidence=0.9,
            source_model="retrieval",
            model_version="neural_queryir",
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["sql"] == "SELECT 1"
        assert d["model_version"] == "neural_queryir"
        assert d["abstained"] is False

    def test_low_confidence_triggers_abstention(self):
        """Service abstains when confidence is below threshold."""
        from inference.nl2sql_service import NL2SQLService

        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "sql": "SELECT * FROM unknown",
            "query_ir": {},
            "confidence": 0.05,
        }
        mock_model.predict.return_value = mock_result

        service = NL2SQLService(
            retrieval_model=mock_model,
            config={"abstention_threshold": 0.20}
        )
        result = service.predict("wfjweifjweif gibberish")
        assert result.abstained is True
        assert result.abstention_reason == "Low confidence"

    def test_exception_returns_abstention(self):
        """Runtime errors produce abstention, not crashes."""
        from inference.nl2sql_service import NL2SQLService

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Tokenizer OOM")

        service = NL2SQLService(retrieval_model=mock_model)
        result = service.predict("show sales")
        assert result.abstained is True
        assert "Tokenizer OOM" in result.abstention_reason


class TestCanonicalImports:
    """Verify canonical import paths resolve correctly."""

    def test_schema_graph_import(self):
        from db.schema_graph import SchemaGraph, TableInfo, ColumnInfo, ForeignKeyInfo
        assert SchemaGraph is not None

    def test_tfidf_retriever_import(self):
        from retrieval.tfidf_retriever import TfidfRetriever, RetrievalResult
        assert TfidfRetriever is not None

    def test_compat_query_ir_loader(self):
        from compat.legacy_query_ir_loader import QueryIR, get_query_ir_class
        assert QueryIR is not None
        v2_cls = get_query_ir_class("v2")
        assert v2_cls is not None
        assert v2_cls.__name__ == "QueryNode"

    def test_model_registry_has_canonical_version(self):
        from neural_ir.model_registry import SCHEMA_AWARE_MODEL_VERSIONS
        assert "neural_queryir" in SCHEMA_AWARE_MODEL_VERSIONS
        # Backward compat
        assert "schema_aware_queryir_v1" in SCHEMA_AWARE_MODEL_VERSIONS


class TestProvenanceCapture:
    """Test training provenance recording."""

    def test_provenance_captures_environment(self):
        from training.provenance import TrainingProvenance
        prov = TrainingProvenance.capture(
            config={"training": {"seed": 42}},
            train_path="nonexistent.jsonl",
        )
        assert prov.python_version != ""
        assert prov.started_at != ""
        assert prov.seed == 42

    def test_provenance_mark_completed(self):
        from training.provenance import TrainingProvenance
        prov = TrainingProvenance()
        assert prov.completed_at == ""
        prov.mark_completed()
        assert prov.completed_at != ""

    def test_provenance_to_dict(self):
        from training.provenance import TrainingProvenance
        prov = TrainingProvenance.capture(config={})
        d = prov.to_dict()
        assert isinstance(d, dict)
        assert "python_version" in d
        assert "git_commit" in d


class TestSemanticConsistencyLoss:
    """Test semantic consistency loss module."""

    def test_loss_disabled(self):
        import torch
        from neural_optimization.semantic_consistency_loss import SemanticConsistencyLoss
        loss = SemanticConsistencyLoss(enabled=False)
        outputs = {"intent_logits": torch.randn(2, 5)}
        labels = {"intent_label": torch.tensor([0, 1])}
        result = loss(outputs, labels)
        assert result["semantic_total"].item() == 0.0

    def test_loss_returns_all_keys(self):
        import torch
        from neural_optimization.semantic_consistency_loss import SemanticConsistencyLoss
        loss = SemanticConsistencyLoss(enabled=True)
        outputs = {
            "intent_logits": torch.randn(4, 8),
            "metric_aggregation_logits": torch.randn(4, 5),
            "metric_column_logits": torch.randn(4, 10),
            "filter_operator_logits": torch.randn(4, 6),
            "filter_column_logits": torch.randn(4, 10),
            "dimension_column_logits": torch.randn(4, 10),
        }
        labels = {"intent_label": torch.tensor([0, 1, 2, 3])}
        col_types = torch.tensor([
            [0, 1, 2, 1, 3, 2, 1, 0, 1, 2],
            [1, 1, 2, 2, 3, 1, 0, 1, 2, 1],
            [2, 0, 1, 1, 1, 2, 3, 0, 1, 2],
            [1, 2, 1, 0, 2, 1, 1, 3, 0, 1],
        ])
        result = loss(outputs, labels, column_datatypes=col_types)
        assert "semantic_total" in result
        assert "metric_agg_consistency" in result
        assert "filter_op_consistency" in result
        assert "groupby_consistency" in result
        # All should be finite
        for key, val in result.items():
            assert torch.isfinite(val), f"{key} is not finite"


class TestIdentifierDecomposer:
    """Test deterministic identifier decomposition."""

    def test_underscore_split(self):
        from neural_ir.identifier_decomposer import decompose_identifier
        assert decompose_identifier("order_items") == ["order", "items"]

    def test_camel_case_split(self):
        from neural_ir.identifier_decomposer import decompose_identifier
        assert decompose_identifier("totalRevenue") == ["total", "revenue"]

    def test_all_caps(self):
        from neural_ir.identifier_decomposer import decompose_identifier
        assert decompose_identifier("CUSTOMER_ID") == ["customer", "id"]

    def test_mixed_case_underscore(self):
        from neural_ir.identifier_decomposer import decompose_identifier
        assert decompose_identifier("first_name") == ["first", "name"]

    def test_empty_string(self):
        from neural_ir.identifier_decomposer import decompose_identifier
        assert decompose_identifier("") == []

    def test_max_tokens(self):
        from neural_ir.identifier_decomposer import decompose_and_flatten
        result = decompose_and_flatten("a_very_long_identifier_name_here", max_tokens=3)
        assert len(result) <= 3

    def test_schema_decomposition(self):
        from neural_ir.identifier_decomposer import decompose_schema_identifiers
        tables = {
            "order_items": {"columns": {"item_price": {}, "productId": {}}},
        }
        result = decompose_schema_identifiers(tables)
        assert "order_items" in result
        assert result["order_items"]["table_tokens"] == ["order", "items"]
        assert result["order_items"]["columns"]["item_price"] == ["item", "price"]
        assert result["order_items"]["columns"]["productId"] == ["product", "id"]


class TestCalibrationEvaluator:
    """Test calibration evaluation."""

    def test_perfect_calibration(self):
        from neural_ir.calibration_evaluator import CalibrationEvaluator
        evaluator = CalibrationEvaluator(n_bins=5)
        # Perfectly calibrated: confidence = accuracy
        confidences = [0.9] * 9 + [0.1]
        correct = [True] * 9 + [False]
        results = evaluator.evaluate(confidences, correct)
        assert results["n_samples"] == 10
        assert results["ece"] < 0.2  # Should be reasonably calibrated
        assert results["brier_score"] < 0.1

    def test_empty_input(self):
        from neural_ir.calibration_evaluator import CalibrationEvaluator
        evaluator = CalibrationEvaluator()
        results = evaluator.evaluate([], [])
        assert results["n_samples"] == 0
        assert results["ece"] == 0.0

    def test_coverage_risk_curve(self):
        from neural_ir.calibration_evaluator import CalibrationEvaluator
        evaluator = CalibrationEvaluator()
        confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
        correct = [False, False, True, True, True]
        results = evaluator.evaluate(confidences, correct)
        curve = results["coverage_risk_curve"]
        assert len(curve) > 0
        # At threshold 0.0, coverage should be 1.0
        assert curve[0]["coverage"] == 1.0

    def test_optimal_threshold(self):
        from neural_ir.calibration_evaluator import compute_optimal_threshold
        confidences = [0.1, 0.2, 0.3, 0.8, 0.9, 0.95]
        correct = [False, False, True, True, True, True]
        result = compute_optimal_threshold(confidences, correct, target_precision=0.90)
        assert result["precision"] >= 0.90 or result["coverage"] == 0.0


class TestSafetyDataset:
    """Test safety dataset builder."""

    def test_build_produces_rows(self):
        from dataset_training.safety_dataset import SafetyDatasetBuilder
        builder = SafetyDatasetBuilder(augmentation_factor=1)
        dataset = builder.build()
        assert len(dataset) > 0

    def test_safe_rows_have_correct_masks(self):
        from dataset_training.safety_dataset import SafetyDatasetBuilder
        builder = SafetyDatasetBuilder(augmentation_factor=1)
        dataset = builder.build()
        safe_rows = [r for r in dataset if r.get("is_safe")]
        assert len(safe_rows) > 0
        for row in safe_rows:
            assert row["task_masks"]["safety"] == 1.0
            assert row["safety_label"] == "safe"

    def test_unsafe_rows_have_categories(self):
        from dataset_training.safety_dataset import SafetyDatasetBuilder
        builder = SafetyDatasetBuilder(augmentation_factor=1)
        dataset = builder.build()
        unsafe_rows = [r for r in dataset if not r.get("is_safe")]
        assert len(unsafe_rows) > 0
        categories = {r["safety_category"] for r in unsafe_rows}
        assert "unsafe_ddl" in categories
        assert "unsafe_dml" in categories
        assert "unsafe_injection" in categories

    def test_balance(self):
        from dataset_training.safety_dataset import SafetyDatasetBuilder
        builder = SafetyDatasetBuilder(augmentation_factor=3)
        dataset = builder.build()
        safe = sum(1 for r in dataset if r.get("is_safe"))
        unsafe = len(dataset) - safe
        # Both categories should be represented
        assert safe > 0
        assert unsafe > 0
