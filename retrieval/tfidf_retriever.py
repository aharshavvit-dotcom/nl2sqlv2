"""Canonical TF-IDF retriever location.

Re-exports TfidfRetriever and RetrievalResult from nl2sql_v1.retriever.
All new code should import from retrieval.tfidf_retriever.

Migration deadline: 2026-09-01
"""
from __future__ import annotations

from nl2sql_v1.retriever import (  # noqa: F401
    RetrievalResult,
    TfidfRetriever,
)

__all__ = [
    "TfidfRetriever",
    "RetrievalResult",
]
