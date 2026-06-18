from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any

try:  # pragma: no cover - optional
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None

from .semantic_confidence import SemanticConfidence


class SemanticMapper:
    def __init__(self, semantic_profile: dict[str, Any]):
        self.profile = semantic_profile
        self.confidence = SemanticConfidence()

    def map_table(self, phrase: str) -> dict[str, Any]:
        candidates = []
        for table, info in (self.profile.get("tables") or {}).items():
            aliases = info.get("aliases") or [table, table.replace("_", " ")]
            score, match_type = self._best_score(phrase, aliases)
            candidates.append({"target": table, "score": score, "match_type": match_type})
        return self._result(candidates, "table_mapping")

    def map_column(self, phrase: str, table: str | None = None) -> dict[str, Any]:
        candidates = []
        for table_name, info in (self.profile.get("tables") or {}).items():
            if table and table_name != table:
                continue
            for column in info.get("columns", []):
                if column.get("is_sensitive"):
                    continue
                aliases = column.get("aliases") or [column.get("name", "")]
                score, match_type = self._best_score(phrase, aliases)
                candidates.append({"target": f"{table_name}.{column['name']}", "table": table_name, "column": column["name"], "score": score, "match_type": match_type})
        return self._result(candidates, "column_mapping")

    def map_metric(self, phrase: str, table: str | None = None) -> dict[str, Any]:
        return self._map_collection(phrase, self.profile.get("metrics") or {}, "metric_mapping", table)

    def map_dimension(self, phrase: str, table: str | None = None) -> dict[str, Any]:
        return self._map_collection(phrase, self.profile.get("dimensions") or {}, "dimension_mapping", table)

    def map_date(self, phrase: str, table: str | None = None) -> dict[str, Any]:
        return self._map_collection(phrase, self.profile.get("dates") or {}, "date_mapping", table)

    def _map_collection(self, phrase: str, collection: dict[str, Any], mapping_type: str, table: str | None) -> dict[str, Any]:
        candidates = []
        for key, item in collection.items():
            item_table = item.get("base_table") or item.get("table")
            if table and item_table != table:
                continue
            aliases = item.get("aliases") or [key, str(item.get("column") or "")]
            score, match_type = self._best_score(phrase, aliases)
            candidates.append({"target": key, "table": item_table, "column": item.get("column"), "score": score, "match_type": match_type})
        return self._result(candidates, mapping_type)

    def _result(self, candidates: list[dict[str, Any]], mapping_type: str) -> dict[str, Any]:
        ranked = sorted(candidates, key=lambda item: (-float(item["score"]), str(item["target"])))
        alternatives = [{**item, "score": round(float(item["score"]), 4)} for item in ranked[:5] if float(item["score"]) >= 0.45]
        if not alternatives:
            return {
                "matched": False,
                "target": None,
                "score": 0.0,
                "match_type": "no_match",
                "alternatives": [],
                "mapping_type": mapping_type,
                "requires_clarification": True,
                "ambiguous": False,
            }
        top = alternatives[0]
        state = self.confidence.classify(float(top["score"]), alternatives)
        matched = state["high_confidence"] or (not state["ambiguous"] and float(top["score"]) >= self.confidence.minimum_match_threshold)
        return {
            "matched": matched and not state["requires_clarification"],
            "target": top["target"] if matched and not state["requires_clarification"] else None,
            "score": top["score"],
            "match_type": top.get("match_type", "fuzzy"),
            "alternatives": alternatives,
            "ambiguous": state["ambiguous"],
            "requires_clarification": state["requires_clarification"],
            "mapping_type": mapping_type,
        }

    @staticmethod
    def _best_score(phrase: str, aliases: list[str]) -> tuple[float, str]:
        phrase_norm = _normalize_phrase(phrase)
        phrase_ident = phrase_norm.replace(" ", "_")
        best = (0.0, "no_match")
        for alias in aliases:
            alias_norm = _normalize_phrase(alias)
            if not alias_norm:
                continue
            alias_ident = alias_norm.replace(" ", "_")
            if phrase_norm == alias_norm or phrase_ident == alias_ident:
                return 0.98, "alias_exact"
            if re.search(rf"\b{re.escape(alias_norm)}\b", phrase_norm):
                best = max(best, (0.9, "alias_phrase"), key=lambda item: item[0])
            token_overlap = _token_score(phrase_norm, alias_norm)
            if token_overlap:
                best = max(best, (token_overlap, "token_overlap"), key=lambda item: item[0])
            if fuzz is not None:
                fuzzy_score = float(fuzz.WRatio(phrase_norm, alias_norm)) / 100.0
            else:
                fuzzy_score = SequenceMatcher(None, phrase_norm, alias_norm).ratio()
            best = max(best, (fuzzy_score, "fuzzy"), key=lambda item: item[0])
        return round(best[0], 4), best[1]


def _normalize_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _token_score(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    if not overlap:
        return 0.0
    return min(0.88, 0.48 + len(overlap) / max(len(left_tokens), len(right_tokens)) * 0.40)
