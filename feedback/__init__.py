"""Feedback capture and feedback-to-training-data utilities.

.. deprecated::
    Manual user feedback is now **optional**.  The primary training loop
    uses dataset-driven self-improvement (see ``self_training`` package).
    This module is preserved for backward compatibility and optional
    human-in-the-loop correction workflows.
"""

# Feedback mode: "optional" means the primary loop is dataset-driven.
# Set to "active" only if you want the UI to prominently surface feedback.
FEEDBACK_MODE = "optional"

from .feedback_models import ALLOWED_FEEDBACK_TAGS, ALLOWED_RATINGS, QueryFeedback
from .feedback_store import FeedbackStore

__all__ = [
    "ALLOWED_FEEDBACK_TAGS",
    "ALLOWED_RATINGS",
    "FEEDBACK_MODE",
    "FeedbackStore",
    "QueryFeedback",
]
