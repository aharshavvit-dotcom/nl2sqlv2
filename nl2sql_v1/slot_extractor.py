from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz, process


LIMIT_RE = re.compile(r"\b(?:top|first|best|highest|largest|bottom|lowest|worst)?\s*(\d{1,3})\b", re.I)
YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")


@dataclass
class ExtractedSlots:
    template_id: str | None = None
    metric: str | None = None
    dimension: str | None = None
    limit: int = 10
    order: str = "DESC"
    filters: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "metric": self.metric,
            "dimension": self.dimension,
            "limit": self.limit,
            "order": self.order,
            "filters": self.filters,
        }


class SlotExtractor:
    def __init__(self, catalog: Any):
        self.catalog = catalog

    def extract(self, question: str, fallback: dict[str, Any] | None = None) -> ExtractedSlots:
        fallback = fallback or {}
        q = question.lower()
        fallback_metric = self._catalog_key_or_hit(fallback.get("metric"), self.catalog.metrics)
        fallback_dimension = self._catalog_key_or_hit(fallback.get("dimension"), self.catalog.dimensions)
        slots = ExtractedSlots(
            template_id=fallback.get("template_id"),
            metric=fallback_metric,
            dimension=fallback_dimension,
            limit=int(fallback.get("limit") or 10),
            order=fallback.get("order", "DESC"),
            filters=self._catalog_filters(fallback.get("filters") or {}),
        )

        slots.limit = self._extract_limit(q, slots.limit)
        slots.order = "ASC" if any(word in q for word in ["bottom", "lowest", "least", "worst", "smallest"]) else "DESC"
        slots.metric = self._best_catalog_hit(q, self.catalog.metrics) or slots.metric or "sales"
        slots.dimension = self._extract_dimension(q) or self._best_catalog_hit(q, self.catalog.dimensions) or slots.dimension

        year_match = YEAR_RE.search(q)
        if year_match:
            slots.filters["year"] = year_match.group(1)

        for filter_key, item in self.catalog.filters.items():
            for alias in item.get("aliases", []):
                value = self._extract_after_alias(q, alias)
                if value:
                    slots.filters[filter_key] = value
                    break

        if slots.template_id is None:
            slots.template_id = "rank_dimension" if slots.dimension else "aggregate_metric"
        return slots

    def _catalog_filters(self, filters: dict[str, Any]) -> dict[str, str]:
        valid: dict[str, str] = {}
        for key, value in filters.items():
            catalog_key = self._catalog_key_or_hit(key, self.catalog.filters)
            if catalog_key and value:
                valid[catalog_key] = str(value)
        return valid

    @staticmethod
    def _extract_limit(question: str, default: int) -> int:
        match = LIMIT_RE.search(question)
        if match:
            return max(1, min(100, int(match.group(1))))
        return default

    @staticmethod
    def _extract_dimension(question: str) -> str | None:
        if "customer segment" in question or "customer tier" in question or "segment" in question:
            return "customer_segment"
        if "customers" in question or "customer" in question or "clients" in question or "buyers" in question:
            return "customer"
        if "products" in question or "product" in question or "items" in question:
            return "product"
        if "categories" in question or "category" in question:
            return "category"
        if "brands" in question or "brand" in question:
            return "brand"
        if "stores" in question or "store" in question:
            return "store"
        if "sales reps" in question or "sales rep" in question or "representatives" in question or "salespeople" in question:
            return "sales_rep"
        if "regions" in question or "region" in question:
            return "region"
        if "states" in question or "state" in question:
            return "state"
        if "cities" in question or "city" in question:
            return "city"
        if "monthly" in question or "month" in question:
            return "month"
        if "yearly" in question or "annual" in question or "year" in question:
            return "year"

        patterns = [
            r"\bby\s+([a-z_ ]+?)(?:\s+in\s+|\s+for\s+|\s+where\s+|$)",
            r"\bper\s+([a-z_ ]+?)(?:\s+in\s+|\s+for\s+|\s+where\s+|$)",
            r"\bacross\s+([a-z_ ]+?)(?:\s+in\s+|\s+for\s+|\s+where\s+|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, question, re.I)
            if match:
                value = match.group(1).strip()
                value = re.sub(
                    r"\b(total|avg|average|mean|sales|revenue|orders|order count|quantity|units|profit|discount|amount)\b",
                    "",
                    value,
                ).strip()
                if value and value not in {"total", "average", "avg", "amount"}:
                    return value
        return None

    @staticmethod
    def _extract_after_alias(question: str, alias: str) -> str | None:
        pattern = rf"\b{re.escape(alias.lower())}\s+(?:is\s+|=|equals\s+)?([a-z0-9 _-]+)"
        match = re.search(pattern, question)
        if not match:
            return None
        value = match.group(1).strip()
        value = re.split(r"\s+(?:by|top|limit|order|group|and)\s+", value)[0].strip()
        return value or None

    @staticmethod
    def _best_catalog_hit(question: str, catalog: dict[str, dict[str, Any]]) -> str | None:
        choices: dict[str, str] = {}
        for key, item in catalog.items():
            choices[key] = " ".join([key, *item.get("aliases", [])])
        match = process.extractOne(question, choices, scorer=fuzz.partial_ratio)
        if not match:
            return None
        _, score, key = match
        return str(key) if score >= 76 else None

    @staticmethod
    def _catalog_key_or_hit(value: Any, catalog: dict[str, dict[str, Any]]) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        normalized = _normalize(text)
        for key, item in catalog.items():
            aliases = [key, *item.get("aliases", [])]
            if normalized in {_normalize(alias) for alias in aliases}:
                return key

        choices = {key: " ".join([key, *item.get("aliases", [])]) for key, item in catalog.items()}
        match = process.extractOne(text, choices, scorer=fuzz.WRatio)
        if not match:
            return None
        _, score, key = match
        return str(key) if score >= 76 else None


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
