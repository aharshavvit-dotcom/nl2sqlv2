"""Dataset-driven self-improvement training loop.

This package implements the core self-improvement pipeline:
  Gold Dataset → Train → Predict → Compare → Classify Errors →
  Generate Corrections + Hard Negatives → Retrain → Evaluate → Report

No external LLM APIs, no GPU required.  All training uses the local
PyTorch-based Neural QueryIR model on CPU.
"""

from .gold_comparator import BatchComparisonReport, ComparisonResult, GoldComparator, SQLComparisonResult
from .error_classifier import ErrorCategory, ErrorClassification, ErrorClassifier, ErrorReport
from .hard_negative_generator import PredictionHardNegativeGenerator
from .correction_generator import CorrectionExampleGenerator
from .improvement_tracker import ImprovementReport, ImprovementTracker
from .prediction_runner import PredictionRunner
from .model_selector import ModelSelector
from .self_improvement_loop import SelfImprovementLoop

__all__ = [
    "BatchComparisonReport",
    "ComparisonResult",
    "CorrectionExampleGenerator",
    "ErrorCategory",
    "ErrorClassification",
    "ErrorClassifier",
    "ErrorReport",
    "GoldComparator",
    "ImprovementReport",
    "ImprovementTracker",
    "ModelSelector",
    "PredictionHardNegativeGenerator",
    "PredictionRunner",
    "SelfImprovementLoop",
    "SQLComparisonResult",
]
