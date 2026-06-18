"""Tests for neural_optimization.early_stopping."""

from __future__ import annotations

from neural_optimization.early_stopping import EarlyStopping


class TestEarlyStopping:
    def test_does_not_stop_before_patience(self):
        es = EarlyStopping(patience=3, metric_name="accuracy", mode="max")
        assert es.step({"accuracy": 0.8}) is False
        assert es.step({"accuracy": 0.7}) is False  # worse
        assert es.step({"accuracy": 0.6}) is False  # worse
        # counter is 2, not yet >= 3

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3, metric_name="accuracy", mode="max")
        es.step({"accuracy": 0.8})
        es.step({"accuracy": 0.7})
        es.step({"accuracy": 0.6})
        assert es.step({"accuracy": 0.5}) is True  # 3 non-improvements

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=2, metric_name="accuracy", mode="max")
        es.step({"accuracy": 0.8})
        es.step({"accuracy": 0.7})  # worse, counter=1
        es.step({"accuracy": 0.9})  # better, counter=0
        assert es.step({"accuracy": 0.85}) is False  # worse, counter=1
        assert es.counter == 1

    def test_min_mode(self):
        es = EarlyStopping(patience=2, metric_name="loss", mode="min")
        es.step({"loss": 0.5})
        es.step({"loss": 0.6})  # worse
        assert es.step({"loss": 0.7}) is True  # 2 non-improvements

    def test_min_delta(self):
        es = EarlyStopping(patience=2, metric_name="accuracy", mode="max", min_delta=0.01)
        es.step({"accuracy": 0.80})
        # Improvement smaller than min_delta counts as no improvement
        es.step({"accuracy": 0.805})
        assert es.counter == 1

    def test_best_value_tracked(self):
        es = EarlyStopping(patience=3, metric_name="accuracy", mode="max")
        es.step({"accuracy": 0.7})
        es.step({"accuracy": 0.9})
        es.step({"accuracy": 0.8})
        assert es.best_value == 0.9

    def test_fallback_metric(self):
        """When metric_name is missing, falls back to overall_slot_accuracy."""
        es = EarlyStopping(patience=2, metric_name="nonexistent", mode="max")
        es.step({"overall_slot_accuracy": 0.5})
        es.step({"overall_slot_accuracy": 0.4})
        assert es.step({"overall_slot_accuracy": 0.3}) is True
