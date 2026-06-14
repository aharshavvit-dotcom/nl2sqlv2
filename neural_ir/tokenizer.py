from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str | None) -> list[str]:
    """Lowercase and split text on punctuation/whitespace."""
    if not text:
        return []
    return TOKEN_RE.findall(str(text).lower())
