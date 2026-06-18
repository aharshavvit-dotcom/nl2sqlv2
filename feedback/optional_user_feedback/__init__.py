"""Optional user feedback — legacy manual feedback modules.

This sub-package preserves the original user-feedback-first learning
modules for optional human-in-the-loop correction workflows.

The primary training loop now uses dataset-driven self-improvement.
See the ``self_training`` package for the active pipeline.

All original classes and functions are re-exported here for backward
compatibility.
"""

from feedback.feedback_models import ALLOWED_FEEDBACK_TAGS, ALLOWED_RATINGS, QueryFeedback
from feedback.feedback_store import FeedbackStore, append_feedback
from feedback.feedback_quality import FeedbackQualityFilter
from feedback.correction_parser import parse_correction
from feedback.feedback_to_ir_examples import build_feedback_training_examples

__all__ = [
    "ALLOWED_FEEDBACK_TAGS",
    "ALLOWED_RATINGS",
    "FeedbackQualityFilter",
    "FeedbackStore",
    "QueryFeedback",
    "append_feedback",
    "build_feedback_training_examples",
    "parse_correction",
]
