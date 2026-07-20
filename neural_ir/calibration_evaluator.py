"""Calibration evaluator for NL2SQL confidence scores.

Computes standard calibration metrics on the dedicated calibration split:
- Expected Calibration Error (ECE)
- Brier Score
- Coverage-Risk curve
- Reliability diagram data

Usage::

    evaluator = CalibrationEvaluator()
    results = evaluator.evaluate(predictions, ground_truth)
    evaluator.save(results, Path("artifacts/evaluation/calibration_report.json"))
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class CalibrationEvaluator:
    """Evaluate calibration quality of confidence predictions."""

    def __init__(self, n_bins: int = 10):
        """Initialize evaluator.

        Parameters
        ----------
        n_bins : int
            Number of bins for ECE and reliability diagram.
        """
        self.n_bins = n_bins

    def evaluate(
        self,
        confidences: list[float],
        correct: list[bool],
    ) -> dict[str, Any]:
        """Compute calibration metrics.

        Parameters
        ----------
        confidences : list[float]
            Predicted confidence scores (0.0 to 1.0).
        correct : list[bool]
            Whether each prediction was correct.

        Returns
        -------
        dict with 'ece', 'brier_score', 'reliability_diagram',
        'coverage_risk_curve', 'n_samples'.
        """
        if not confidences or not correct:
            return {
                "ece": 0.0,
                "brier_score": 0.0,
                "reliability_diagram": [],
                "coverage_risk_curve": [],
                "n_samples": 0,
            }

        conf = np.array(confidences, dtype=np.float64)
        corr = np.array(correct, dtype=np.float64)
        n = len(conf)

        ece = self._expected_calibration_error(conf, corr)
        brier = self._brier_score(conf, corr)
        reliability = self._reliability_diagram(conf, corr)
        coverage_risk = self._coverage_risk_curve(conf, corr)

        return {
            "ece": float(ece),
            "brier_score": float(brier),
            "mean_confidence": float(np.mean(conf)),
            "mean_accuracy": float(np.mean(corr)),
            "overconfidence": float(np.mean(conf) - np.mean(corr)),
            "reliability_diagram": reliability,
            "coverage_risk_curve": coverage_risk,
            "n_samples": n,
        }

    def _expected_calibration_error(
        self, conf: np.ndarray, corr: np.ndarray
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        ECE = Σ (|B_m| / n) × |acc(B_m) - conf(B_m)|

        where B_m is the set of predictions in bin m.
        """
        n = len(conf)
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            mask = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
            if i == 0:
                mask = (conf >= bin_edges[i]) & (conf <= bin_edges[i + 1])
            bin_size = mask.sum()
            if bin_size == 0:
                continue
            bin_acc = corr[mask].mean()
            bin_conf = conf[mask].mean()
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

        return ece

    def _brier_score(self, conf: np.ndarray, corr: np.ndarray) -> float:
        """Compute Brier Score: mean((confidence - correct)^2)."""
        return float(np.mean((conf - corr) ** 2))

    def _reliability_diagram(
        self, conf: np.ndarray, corr: np.ndarray
    ) -> list[dict[str, float]]:
        """Compute reliability diagram data points.

        Returns list of {bin_midpoint, accuracy, confidence, count}.
        """
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        diagram: list[dict[str, float]] = []

        for i in range(self.n_bins):
            mask = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
            if i == 0:
                mask = (conf >= bin_edges[i]) & (conf <= bin_edges[i + 1])
            bin_size = int(mask.sum())
            if bin_size == 0:
                continue
            diagram.append({
                "bin_midpoint": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                "accuracy": float(corr[mask].mean()),
                "confidence": float(conf[mask].mean()),
                "count": bin_size,
            })

        return diagram

    def _coverage_risk_curve(
        self, conf: np.ndarray, corr: np.ndarray
    ) -> list[dict[str, float]]:
        """Compute coverage-risk curve.

        At each threshold, compute:
        - coverage: fraction of examples above threshold
        - risk: error rate among covered examples
        """
        thresholds = np.linspace(0, 1, 21)
        curve: list[dict[str, float]] = []

        for threshold in thresholds:
            mask = conf >= threshold
            coverage = float(mask.sum() / len(conf)) if len(conf) > 0 else 0.0
            if mask.sum() == 0:
                risk = 0.0
            else:
                risk = float(1.0 - corr[mask].mean())
            curve.append({
                "threshold": float(threshold),
                "coverage": coverage,
                "risk": risk,
                "count": int(mask.sum()),
            })

        return curve

    def save(self, results: dict[str, Any], path: Path) -> None:
        """Save calibration report to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(results, indent=2, default=str),
            encoding="utf-8",
        )


def compute_optimal_threshold(
    confidences: list[float],
    correct: list[bool],
    target_precision: float = 0.90,
) -> dict[str, float]:
    """Find the confidence threshold that achieves target precision.

    Parameters
    ----------
    confidences : list[float]
        Predicted confidence scores.
    correct : list[bool]
        Whether each prediction was correct.
    target_precision : float
        Desired precision level (default 0.90).

    Returns
    -------
    dict with 'threshold', 'precision', 'coverage'.
    """
    if not confidences or not correct:
        return {"threshold": 0.5, "precision": 0.0, "coverage": 0.0}

    conf = np.array(confidences)
    corr = np.array(correct)
    n = len(conf)

    best = {"threshold": 0.5, "precision": 0.0, "coverage": 0.0}
    for threshold in np.linspace(0.05, 0.95, 19):
        mask = conf >= threshold
        if mask.sum() == 0:
            continue
        precision = float(corr[mask].mean())
        coverage = float(mask.sum() / n)
        if precision >= target_precision and coverage > best["coverage"]:
            best = {
                "threshold": float(threshold),
                "precision": precision,
                "coverage": coverage,
            }

    return best
