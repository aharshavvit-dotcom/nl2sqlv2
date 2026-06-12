from __future__ import annotations

from training import evaluate_retriever
from training import train_retriever_from_datasets as trainer
from tests.test_train_retriever_from_datasets import FakeCorpusBuilder


def test_evaluate_retriever_computes_metrics(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(trainer, "CorpusBuilder", FakeCorpusBuilder)
    trainer.train_from_datasets(["mock"], artifact_dir=tmp_path, output_dir=tmp_path / "processed")

    report = evaluate_retriever.evaluate_retriever(tmp_path)

    assert report["top_1_template_accuracy"] == 1.0
    assert report["top_5_template_accuracy"] == 1.0
    assert report["evaluation_splits"] == ["validation"]
    assert (tmp_path / "evaluation_report.json").exists()
