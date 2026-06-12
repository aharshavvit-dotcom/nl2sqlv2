from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rapidfuzz import fuzz, process

from .schema import SchemaGraph
from .slot_extractor import ExtractedSlots


QUALIFIED_COLUMN = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b")


@dataclass(frozen=True)
class Catalog:
    metrics: dict[str, dict[str, Any]]
    dimensions: dict[str, dict[str, Any]]
    filters: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class MatchedMetric:
    key: str
    expression: str
    aggregate: str
    alias: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class MatchedDimension:
    key: str
    expression: str
    alias: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class MatchedFilter:
    key: str
    column: str
    operator: str
    value: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class MatchResult:
    metric: MatchedMetric
    dimension: MatchedDimension | None
    filters: list[MatchedFilter]


class SchemaMatcher:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SchemaMatcher":
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(
            Catalog(
                metrics=raw.get("metrics", {}),
                dimensions=raw.get("dimensions", {}),
                filters=raw.get("filters", {}),
            )
        )

    def match(self, slots: ExtractedSlots, schema: SchemaGraph) -> MatchResult:
        metric = self.match_metric(slots.metric or "sales", schema)
        dimension = self.match_dimension(slots.dimension, schema) if slots.dimension else None
        filters = [self.match_filter(key, value, schema) for key, value in slots.filters.items()]
        return MatchResult(metric=metric, dimension=dimension, filters=[f for f in filters if f])

    def match_metric(self, text: str, schema: SchemaGraph) -> MatchedMetric:
        key, score = self._best_key(text, self.catalog.metrics)
        item = self.catalog.metrics[key]
        self._assert_expression_in_schema(item["expression"], schema)
        return MatchedMetric(
            key=key,
            expression=item["expression"],
            aggregate=item.get("aggregate", "SUM").upper(),
            alias=item.get("alias", key),
            score=score,
        )

    def match_dimension(self, text: str | None, schema: SchemaGraph) -> MatchedDimension | None:
        if not text:
            return None
        key, score = self._best_key(text, self.catalog.dimensions)
        item = self.catalog.dimensions[key]
        expression = item.get("expression") or item["column"]
        self._assert_expression_in_schema(expression, schema)
        return MatchedDimension(
            key=key,
            expression=expression,
            alias=item.get("alias", key),
            score=score,
        )

    def match_filter(self, key_text: str, value: str, schema: SchemaGraph) -> MatchedFilter | None:
        if not value:
            return None
        key, score = self._best_key(key_text, self.catalog.filters)
        item = self.catalog.filters[key]
        column = item["column"]
        self._assert_expression_in_schema(column, schema)
        return MatchedFilter(
            key=key,
            column=column,
            operator=item.get("operator", "="),
            value=value,
            score=score,
        )

    def _best_key(self, text: str, catalog: dict[str, dict[str, Any]]) -> tuple[str, float]:
        normalized_text = _normalize(text)
        for key, item in catalog.items():
            aliases = [key, *item.get("aliases", [])]
            if normalized_text in {_normalize(alias) for alias in aliases}:
                return key, 100.0

        choices: dict[str, str] = {}
        for key, item in catalog.items():
            choices[key] = " ".join([key, *item.get("aliases", [])])
        match = process.extractOne(text, choices, scorer=fuzz.WRatio)
        if not match:
            raise ValueError(f"No catalog match for {text!r}")
        _, score, key = match
        if score < 55:
            raise ValueError(f"Low confidence catalog match for {text!r}: {score:.1f}")
        return str(key), float(score)

    @staticmethod
    def _assert_expression_in_schema(expression: str, schema: SchemaGraph) -> None:
        for table, column in QUALIFIED_COLUMN.findall(expression):
            if not schema.has_column(table, column):
                raise ValueError(f"Expression references missing column: {table}.{column}")


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
