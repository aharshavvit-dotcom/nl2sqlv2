"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from datasets.models import Text2SQLExample, TrainingCorpusStats
from training import train_retriever_from_datasets as trainer


class FakeCorpusBuilder:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def build_corpus(self, dataset_names, max_examples=None, include_schema_text=False):
        train_example = Text2SQLExample(
            example_id="ex1",
            dataset_name="mock",
            db_id="db",
            question="total sales",
            sql="SELECT SUM(amount) FROM orders",
            split="train",
            template_id="metric_summary",
            intent="metric_summary",
            extracted_slots={"metric": "amount", "limit": 10},
            is_supported=True,
        )
        validation_example = Text2SQLExample(
            example_id="ex2",
            dataset_name="mock",
            db_id="db",
            question="sum sales",
            sql="SELECT SUM(amount) FROM orders",
            split="validation",
            template_id="metric_summary",
            intent="metric_summary",
            extracted_slots={"metric": "amount", "limit": 10},
            is_supported=True,
        )
        stats = TrainingCorpusStats(
            total_examples=2,
            supported_examples=2,
            unsupported_examples=0,
            by_dataset={"mock": 2},
            by_template={"metric_summary": 2},
            by_split={"train": 1, "validation": 1},
        )
        return {"examples": [train_example, validation_example], "schemas": {}, "schema_registry": [], "stats": stats}

    def save_outputs(self, processed_payload, output_dir=None):
        (output_dir or self.output_dir).mkdir(parents=True, exist_ok=True)


def test_train_retriever_from_datasets_saves_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(trainer, "CorpusBuilder", FakeCorpusBuilder)
    report = trainer.train_from_datasets(["mock"], artifact_dir=tmp_path / "artifacts", output_dir=tmp_path / "processed")

    artifact_dir = tmp_path / "artifacts"
    assert (artifact_dir / "tfidf_vectorizer.pkl").exists()
    assert (artifact_dir / "tfidf_matrix.pkl").exists()
    assert (artifact_dir / "training_examples.jsonl").exists()
    assert (artifact_dir / "train_examples.jsonl").exists()
    assert (artifact_dir / "validation_examples.jsonl").exists()
    assert report["training_report"]["supported_examples"] == 1
    assert report["training_report"]["supported_examples_all_splits"] == 2
