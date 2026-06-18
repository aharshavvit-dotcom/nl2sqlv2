from __future__ import annotations

from typing import Any


class ClarificationResolver:
    def resolve(self, clarification: dict[str, Any], selected_option: str) -> dict[str, Any]:
        options = clarification.get("options") or []
        if selected_option not in options:
            return {"resolved": False, "error": "selected option is not available", "selected_option": selected_option}
        return {
            "resolved": True,
            "selected_option": selected_option,
            "ambiguity_type": clarification.get("ambiguity_type"),
            "resolved_mapping": selected_option,
        }
