"""Test 05: Neural Runtime — tokenizer, vocab, schema linearizer/linker, model, predictor, converter."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class TestNeuralIRTokenizer:
    def test_tokenize_basic(self) -> None:
        from neural_ir.tokenizer import tokenize
        tokens = tokenize("Top 5 customers by sales")
        assert len(tokens) >= 3

class TestNeuralIRVocab:
    def test_vocab_build(self) -> None:
        from neural_ir.vocab import Vocabulary
        vocab = Vocabulary()
        vocab.build([["hello", "world", "hello"]])
        assert vocab.token_to_id.get("hello") is not None
        assert vocab.token_to_id.get("world") is not None
        # Unknown token should map to unk_id
        assert vocab.token_to_id.get("nonexistent") is None


class TestSchemaLinearizer:
    def test_linearize_schema(self) -> None:
        from neural_ir.schema_linearizer import SchemaLinearizer
        schema = {"tables": {"orders": {"columns": {"order_id": {}, "amount": {}}}}}
        linearizer = SchemaLinearizer()
        result = linearizer.linearize(schema)
        assert isinstance(result, str)
        assert "orders" in result

    def test_extract_schema_items(self) -> None:
        from neural_ir.schema_linearizer import SchemaLinearizer
        schema = {"tables": {"orders": {"columns": {"order_id": {}, "amount": {}}}}}
        items = SchemaLinearizer().extract_schema_items(schema)
        assert isinstance(items, dict)
        assert len(items) > 0


class TestSchemaLinker:
    def test_link_question_to_schema(self) -> None:
        from neural_ir.schema_linker import SchemaLinker
        from neural_ir.candidate_builder import SchemaCandidateBuilder
        schema = {"tables": {"orders": {"columns": {"order_id": {}, "amount": {}, "customer_name": {}}}}}
        candidates = SchemaCandidateBuilder().build_candidates(schema)
        linker = SchemaLinker()
        result = linker.link("customers by sales", candidates)
        assert isinstance(result, dict)
        assert "top_columns" in result


class TestCandidateBuilder:
    def test_builds_candidates(self) -> None:
        from neural_ir.candidate_builder import SchemaCandidateBuilder
        schema = {"tables": {"orders": {"columns": {"order_id": {}, "amount": {}}}}}
        builder = SchemaCandidateBuilder()
        candidates = builder.build_candidates(schema)
        assert isinstance(candidates, dict)


class TestIRLabelEncoder:
    def test_encode(self) -> None:
        from neural_ir.ir_label_encoder import IRLabelEncoder
        from neural_ir.schema_linearizer import SchemaLinearizer
        encoder = IRLabelEncoder()
        schema = {"tables": {"orders": {"columns": {"order_id": {}, "amount": {}}}}}
        schema_items = SchemaLinearizer().extract_schema_items(schema)
        query_ir = {"intent": "metric_summary", "template_id": "metric_summary",
                    "base_table": "orders", "metrics": [{"column": "amount", "table": "orders"}],
                    "dimensions": [], "filters": [], "date_filters": []}
        encoded = encoder.encode(query_ir, schema_items)
        assert isinstance(encoded, dict)


class TestNeuralIRPredictor:
    def test_backward_alias(self) -> None:
        from neural_ir.predictor import NeuralIRPredictor, OptionAIRPredictor
        assert OptionAIRPredictor is NeuralIRPredictor


class TestNeuralIRToIRConverter:
    def test_backward_alias(self) -> None:
        from neural_ir.option_a_to_ir import NeuralIRToIRConverter, OptionAToIRConverter
        assert OptionAToIRConverter is NeuralIRToIRConverter


class TestNeuralIRRepairer:
    def test_backward_alias(self) -> None:
        from neural_ir.ir_repair import NeuralIRRepairer, OptionAIRRepairer
        assert OptionAIRRepairer is NeuralIRRepairer


class TestNeuralIRConfidenceCalibrator:
    def test_backward_alias(self) -> None:
        from neural_ir.confidence_calibrator import NeuralIRConfidenceCalibrator, OptionAConfidenceCalibrator
        assert OptionAConfidenceCalibrator is NeuralIRConfidenceCalibrator

    def test_calibrator_fit(self) -> None:
        from neural_ir.confidence_calibrator import NeuralIRConfidenceCalibrator
        calibrator = NeuralIRConfidenceCalibrator()
        result = calibrator.fit([])
        assert result["fitted_rows"] == 0

    def test_calibrator_calibrate(self) -> None:
        from neural_ir.confidence_calibrator import NeuralIRConfidenceCalibrator
        calibrator = NeuralIRConfidenceCalibrator()
        score = calibrator.calibrate(0.7, {"ir_validation": {"is_valid": True}, "sql_validation": {"is_valid": True}}, {})
        assert 0.0 <= score <= 1.0
