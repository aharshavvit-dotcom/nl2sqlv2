"""Test 07: Training Data Pipeline — adapters, corpus builder, IR training data, hard negatives."""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


class TestDatasetModels:
    def test_import_models(self) -> None:
        from datasets.models import Text2SQLExample
        ex = Text2SQLExample(example_id="test_1", question="q", sql="SELECT 1",
                             db_id="test", dataset_name="test", split="test")
        assert ex.question == "q"


class TestBirdAdapter:
    def test_adapter_loads(self) -> None:
        from datasets.bird_adapter import BirdAdapter
        adapter = BirdAdapter()
        assert adapter is not None

    @pytest.mark.skipif(not (ROOT / "datasets" / "bird-mini").exists(), reason="BIRD dataset not downloaded")
    def test_bird_produces_examples(self) -> None:
        from datasets.bird_adapter import BirdAdapter
        adapter = BirdAdapter()
        examples = list(adapter.load(str(ROOT / "datasets" / "bird-mini")))
        assert len(examples) > 0


class TestSpiderAdapter:
    def test_adapter_loads(self) -> None:
        from datasets.spider_adapter import SpiderAdapter
        adapter = SpiderAdapter()
        assert adapter is not None


class TestWikiSQLAdapter:
    def test_adapter_loads(self) -> None:
        from datasets.wikisql_adapter import WikiSQLAdapter
        adapter = WikiSQLAdapter()
        assert adapter is not None


class TestCorpusBuilder:
    def test_builder_exists(self) -> None:
        from datasets.corpus_builder import CorpusBuilder
        builder = CorpusBuilder()
        assert builder is not None


class TestBuildIRTrainingData:
    def test_import(self) -> None:
        from training_ir.build_ir_training_data import build_ir_training_data
        assert callable(build_ir_training_data)


class TestValidateIRCorpus:
    def test_import(self) -> None:
        from training_ir.validate_ir_corpus import validate_ir_corpus
        assert callable(validate_ir_corpus)


class TestHardNegativeBuilder:
    def test_import(self) -> None:
        from training_ir.build_hard_negative_data import HardNegativeBuilder
        assert HardNegativeBuilder is not None


class TestSQLFeatureExtractor:
    def test_extract_features(self) -> None:
        from datasets.sql_feature_extractor import SQLFeatureExtractor
        features = SQLFeatureExtractor().extract("SELECT SUM(amount) FROM orders GROUP BY customer_id LIMIT 10")
        assert isinstance(features, dict)
        assert "aggregations" in features


class TestSQLPatternClassifier:
    def test_classify_metric_summary(self) -> None:
        from datasets.sql_feature_extractor import SQLFeatureExtractor
        from datasets.sql_pattern_classifier import SQLPatternClassifier
        features = SQLFeatureExtractor().extract("SELECT SUM(amount) FROM orders LIMIT 100")
        classifier = SQLPatternClassifier()
        result = classifier.classify("SELECT SUM(amount) FROM orders LIMIT 100", features)
        assert isinstance(result, dict)
        assert result["template_id"] == "metric_summary"
