from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClarificationState:
    question: str
    ambiguity_type: str
    options: list[str]
    selected_option: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.selected_option is not None
