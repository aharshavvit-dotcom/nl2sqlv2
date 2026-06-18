from __future__ import annotations

from typing import Any


class ClarificationQuestionBuilder:
    def build(self, ambiguity: dict[str, Any]) -> dict[str, Any]:
        options = [str(option.get("label") or option.get("value")) for option in ambiguity.get("options", [])]
        ambiguity_type = str(ambiguity.get("ambiguity_type") or "schema_mapping")
        subject = _subject_for(ambiguity_type)
        return {
            "question": f"Which {subject} do you want to use?",
            "options": options,
            "default_option": None,
            "ambiguity_type": ambiguity_type,
            "candidate_mappings": ambiguity.get("candidate_mappings") or ambiguity.get("options", []),
            "scores": ambiguity.get("scores", []),
            "reason": ambiguity.get("reason") or ambiguity.get("message"),
        }


def _subject_for(ambiguity_type: str) -> str:
    if "column" in ambiguity_type:
        return "column"
    if "metric" in ambiguity_type:
        return "metric"
    if "dimension" in ambiguity_type:
        return "dimension"
    if "date" in ambiguity_type:
        return "date column"
    if "join" in ambiguity_type:
        return "relationship"
    if "table" in ambiguity_type:
        return "table"
    return "schema mapping"
