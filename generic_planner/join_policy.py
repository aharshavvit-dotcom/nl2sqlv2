from __future__ import annotations

from enum import Enum
import re


class JoinPolicy(str, Enum):
    NONE = "none"
    EXPLICIT_ONLY = "explicit_only"
    INFERRED_ALLOWED = "inferred_allowed"


EXPLICIT_JOIN_PATTERNS = (
    r"\bwith\s+\w+",
    r"\balong with\b",
    r"\band their\b",
    r"\bincluding\b",
    r"\bjoined with\b",
)

ANALYTIC_JOIN_PATTERNS = (
    r"\bby\s+\w+",
    r"\btop\b",
    r"\bbottom\b",
    r"\baverage\b",
    r"\bsum\b",
    r"\bgroup\b",
)


def infer_join_policy(question: str, intent: str) -> JoinPolicy:
    normalized = re.sub(r"\s+", " ", str(question or "").lower()).strip()
    if intent in {"show_records", "count_records", "simple_filter"}:
        if any(re.search(pattern, normalized) for pattern in EXPLICIT_JOIN_PATTERNS):
            return JoinPolicy.EXPLICIT_ONLY
        return JoinPolicy.NONE
    if any(re.search(pattern, normalized) for pattern in (*EXPLICIT_JOIN_PATTERNS, *ANALYTIC_JOIN_PATTERNS)):
        return JoinPolicy.EXPLICIT_ONLY
    return JoinPolicy.EXPLICIT_ONLY
