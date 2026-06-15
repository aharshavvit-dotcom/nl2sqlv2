"""SQL dialect constants and helpers."""

from __future__ import annotations

SUPPORTED_DIALECTS = ("sqlite", "postgres")


def get_sqlglot_dialect(dialect: str) -> str:
    """Map a dialect name to a sqlglot-compatible dialect string."""
    normalized = (dialect or "sqlite").lower()
    if normalized in ("postgresql", "pg"):
        return "postgres"
    if normalized in SUPPORTED_DIALECTS:
        return normalized
    return "sqlite"
