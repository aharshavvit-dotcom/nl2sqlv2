"""Tests for self_training.improvement_tracker."""

from __future__ import annotations

import json

import pytest

from self_training.improvement_tracker import ImprovementReport, ImprovementTracker


@pytest.fixture
def tracker(tmp_path):
    return ImprovementTracker(tmp_path)


class TestRecordIteration:
    def test_record_single(self, tracker):
        tracker.record_iteration(0, {"overall_slot_accuracy": 0.50, "exact_match_rate": 0.20})
        assert tracker.iteration_count == 1
        assert tracker.history[0]["iteration"] == 0
        assert tracker.history[0]["overall_slot_accuracy"] == 0.50

    def test_record_multiple(self, tracker):
        tracker.record_iteration(0, {"overall_slot_accuracy": 0.50})
        tracker.record_iteration(1, {"overall_slot_accuracy": 0.60})
        tracker.record_iteration(2, {"overall_slot_accuracy": 0.65})
        assert tracker.iteration_count == 3

    def test_persistence(self, tmp_path):
        tracker1 = ImprovementTracker(tmp_path)
        tracker1.record_iteration(0, {"overall_slot_accuracy": 0.50})
        # Reload from disk
        tracker2 = ImprovementTracker(tmp_path)
        assert tracker2.iteration_count == 1
        assert tracker2.history[0]["overall_slot_accuracy"] == 0.50

    def test_extra_metrics_stored(self, tracker):
        tracker.record_iteration(0, {"overall_slot_accuracy": 0.50, "custom_metric": 0.99})
        assert tracker.history[0]["custom_metric"] == 0.99


class TestGetImprovement:
    def test_improvement_calculation(self, tracker):
        tracker.record_iteration(0, {"overall_slot_accuracy": 0.50})
        tracker.record_iteration(1, {"overall_slot_accuracy": 0.60})
        imp = tracker.get_improvement("overall_slot_accuracy")
        assert abs(imp - 0.20) < 0.001  # 20% improvement

    def test_no_iterations(self, tracker):
        assert tracker.get_improvement("overall_slot_accuracy") == 0.0

    def test_single_iteration(self, tracker):
        tracker.record_iteration(0, {"overall_slot_accuracy": 0.50})
        assert tracker.get_improvement("overall_slot_accuracy") == 0.0

    def test_zero_baseline(self, tracker):
        tracker.record_iteration(0, {"overall_slot_accuracy": 0.0})
        tracker.record_iteration(1, {"overall_slot_accuracy": 0.50})
        imp = tracker.get_improvement("overall_slot_accuracy")
        assert imp == 0.50  # returns raw value when baseline is 0


class TestShouldStop:
    def test_stop_when_no_improvement(self, tracker):
        tracker.record_iteration(0, {"overall_slot_accuracy": 0.60})
        tracker.record_iteration(1, {"overall_slot_accuracy": 0.601})
        assert tracker.should_stop(min_improvement=0.005)

    def test_continue_when_improving(self, tracker):
        tracker.record_iteration(0, {"overall_slot_accuracy": 0.50})
        tracker.record_iteration(1, {"overall_slot_accuracy": 0.60})
        assert not tracker.should_stop(min_improvement=0.005)

    def test_single_iteration_doesnt_stop(self, tracker):
        tracker.record_iteration(0, {"overall_slot_accuracy": 0.50})
        assert not tracker.should_stop()


class TestGenerateReport:
    def test_report_structure(self, tracker):
        tracker.record_iteration(0, {"overall_slot_accuracy": 0.50, "exact_match_rate": 0.20})
        tracker.record_iteration(1, {"overall_slot_accuracy": 0.60, "exact_match_rate": 0.30})
        report = tracker.generate_report()
        assert isinstance(report, ImprovementReport)
        assert report.best_iteration == 1
        assert report.best_metrics["overall_slot_accuracy"] == 0.60
        assert "overall_slot_accuracy" in report.total_improvement
        assert report.total_improvement["overall_slot_accuracy"] == pytest.approx(0.10, abs=1e-5)

    def test_empty_report(self, tracker):
        report = tracker.generate_report()
        assert report.convergence_reason == "no_iterations_recorded"

    def test_to_dict(self, tracker):
        tracker.record_iteration(0, {"overall_slot_accuracy": 0.50})
        report = tracker.generate_report()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "iterations" in d
        assert "best_iteration" in d

    def test_best_iteration_selected(self, tracker):
        tracker.record_iteration(0, {"overall_slot_accuracy": 0.50})
        tracker.record_iteration(1, {"overall_slot_accuracy": 0.70})
        tracker.record_iteration(2, {"overall_slot_accuracy": 0.65})
        report = tracker.generate_report()
        assert report.best_iteration == 1
