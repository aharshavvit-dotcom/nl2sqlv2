"""Tests for neural_optimization.neural_candidate_ranker."""

from __future__ import annotations

import json
import torch
from pathlib import Path

from neural_optimization.neural_candidate_ranker import (
    NeuralCandidateRanker,
    DEFAULT_RANKER_FEATURES,
    save_ranker,
    load_ranker,
)
from neural_optimization.ranker_dataset_builder import (
    RankerDataset,
    build_ranker_dataset,
)


class TestNeuralCandidateRanker:
    def test_forward_pass(self):
        ranker = NeuralCandidateRanker(input_dim=15, hidden_dim=32)
        x = torch.randn(4, 15)
        scores = ranker(x)
        assert scores.shape == (4,)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_sigmoid_output(self):
        ranker = NeuralCandidateRanker(input_dim=5, hidden_dim=16)
        x = torch.randn(10, 5)
        scores = ranker(x)
        assert scores.min().item() >= 0.0
        assert scores.max().item() <= 1.0

    def test_training_step(self):
        ranker = NeuralCandidateRanker(input_dim=5, hidden_dim=16)
        opt = torch.optim.Adam(ranker.parameters())
        loss_fn = torch.nn.BCELoss()
        x = torch.randn(4, 5)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        ranker.train()
        opt.zero_grad()
        scores = ranker(x)
        loss = loss_fn(scores, labels)
        loss.backward()
        opt.step()
        assert loss.item() > 0

    def test_save_and_load(self, tmp_path):
        ranker = NeuralCandidateRanker(input_dim=15, hidden_dim=32, activation="relu")
        config = {
            "input_dim": 15,
            "hidden_dim": 32,
            "activation": "relu",
        }
        save_ranker(ranker, tmp_path, config, {"status": "test"})
        assert (tmp_path / "ranker.pt").exists()
        assert (tmp_path / "ranker_config.json").exists()
        assert (tmp_path / "ranker_training_report.json").exists()

        loaded = load_ranker(tmp_path)
        x = torch.randn(2, 15)
        scores = loaded(x)
        assert scores.shape == (2,)

    def test_default_features_length(self):
        assert len(DEFAULT_RANKER_FEATURES) == 15


class TestRankerDataset:
    def test_build_and_getitem(self):
        examples = [
            {"retrieval_score": 0.8, "neural_confidence": 0.7, "label": 1.0},
            {"retrieval_score": 0.3, "neural_confidence": 0.2, "label": 0.0},
        ]
        ds = build_ranker_dataset(examples)
        assert len(ds) == 2
        item = ds[0]
        assert item["features"].shape == (15,)
        assert item["label"].item() == 1.0
