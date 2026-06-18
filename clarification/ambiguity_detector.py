from __future__ import annotations

import re
from typing import Any

from semantic_layer import build_semantic_profile
from semantic_layer.semantic_mapper import SemanticMapper


class AmbiguityDetector:
    def detect(self, question: str, mapping_result: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        if mapping_result.get("requires_clarification") or mapping_result.get("ambiguous"):
            return _from_mapping_result(mapping_result)

        profile = build_semantic_profile(schema)
        mapper = SemanticMapper(profile)
        normalized = _normalize(question)

        column_phrase = _leading_show_phrase(normalized)
        if column_phrase:
            column_result = mapper.map_column(column_phrase)
            if column_result.get("requires_clarification") and column_result.get("alternatives"):
                return _from_mapping_result({**column_result, "mapping_type": "column_mapping", "phrase": column_phrase})

        by_match = re.search(r"\bby\s+([a-z0-9_ -]+)$", normalized)
        if by_match:
            dimension_result = mapper.map_dimension(by_match.group(1))
            if dimension_result.get("requires_clarification") and dimension_result.get("alternatives"):
                return _from_mapping_result({**dimension_result, "mapping_type": "dimension_mapping", "phrase": by_match.group(1)})

        if re.search(r"\bwith\s+\w+\s+(name|names|details|detail)\b", normalized):
            candidates = []
            for rel in profile.get("relationships", []):
                candidates.append({"label": f"{rel.get('from_table')} with {rel.get('to_table')}", "value": rel})
            if len(candidates) > 1:
                return {
                    "ambiguous": True,
                    "ambiguity_type": "join_requirement",
                    "message": "Multiple relationships could satisfy this join request.",
                    "options": candidates[:5],
                    "reason": "join relationship is not uniquely determined",
                }

        return {"ambiguous": False, "ambiguity_type": None, "message": "", "options": []}


def _from_mapping_result(mapping_result: dict[str, Any]) -> dict[str, Any]:
    alternatives = mapping_result.get("alternatives") or []
    options = [
        {
            "label": str(item.get("target")),
            "value": item.get("target"),
            "score": item.get("score"),
            "match_type": item.get("match_type"),
        }
        for item in alternatives[:5]
    ]
    mapping_type = mapping_result.get("mapping_type") or "schema_mapping"
    phrase = mapping_result.get("phrase")
    label = phrase or mapping_type.replace("_", " ")
    return {
        "ambiguous": True,
        "ambiguity_type": mapping_type,
        "message": f"Multiple schema mappings match '{label}'.",
        "options": options,
        "candidate_mappings": alternatives[:5],
        "scores": [item.get("score") for item in alternatives[:5]],
        "reason": "top semantic candidates are too close or below confidence threshold",
    }


def _leading_show_phrase(question: str) -> str | None:
    match = re.match(r"^(show|list|display|view|get|fetch|select)\s+(all\s+)?(?P<phrase>[a-z0-9_ -]+)$", question)
    if not match:
        return None
    phrase = match.group("phrase").strip()
    if len(phrase.split()) > 4:
        return None
    return phrase


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_ -]+", str(value or "").lower())).strip()
