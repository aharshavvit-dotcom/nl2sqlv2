from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenericPlannerResult:
    handled: bool
    intent: str | None = None
    query_ir: Any | None = None
    confidence: float = 0.0
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
