from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .join_resolver import JoinResolver
from .schema import SchemaGraph
from .schema_matcher import MatchedDimension, MatchedFilter, MatchedMetric, QUALIFIED_COLUMN
from .slot_extractor import ExtractedSlots


@dataclass(frozen=True)
class RenderPlan:
    template_id: str
    template: str
    context: dict[str, Any]


class TemplateAdapter:
    def __init__(self, templates: dict[str, dict[str, Any]]):
        self.templates = templates

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TemplateAdapter":
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(raw["templates"])

    def build_plan(
        self,
        template_id: str,
        schema: SchemaGraph,
        metric: MatchedMetric,
        dimension: MatchedDimension | None,
        filters: list[MatchedFilter],
        slots: ExtractedSlots,
    ) -> RenderPlan:
        if template_id not in self.templates:
            template_id = "rank_dimension" if dimension else "aggregate_metric"
        if dimension is not None and metric.key != "order_count" and template_id in {
            "count_records",
            "aggregate_metric",
            "metric_summary",
        }:
            template_id = "rank_dimension"
        if template_id == "rank_dimension" and dimension is None:
            template_id = "aggregate_metric"

        template = self.templates[template_id]["sql"]
        base_table = self._first_table(metric.expression)
        required_tables = self._required_tables(metric, dimension, filters)
        joins = JoinResolver(schema).resolve(base_table, required_tables)
        where_clause = self._where_clause(filters)

        context = {
            "base_table": base_table,
            "joins": "\n".join(step.sql for step in joins),
            "where_clause": where_clause,
            "group_by": dimension.expression if dimension else "",
            "order": slots.order,
            "limit": slots.limit,
            "metric": {
                "expression": metric.expression,
                "aggregate": metric.aggregate,
                "alias": metric.alias,
            },
            "dimension": {
                "expression": dimension.expression if dimension else "",
                "alias": dimension.alias if dimension else "",
            },
        }
        return RenderPlan(template_id=template_id, template=template, context=context)

    @staticmethod
    def _required_tables(
        metric: MatchedMetric,
        dimension: MatchedDimension | None,
        filters: list[MatchedFilter],
    ) -> set[str]:
        expressions = [metric.expression]
        if dimension:
            expressions.append(dimension.expression)
        expressions.extend(item.column for item in filters)
        return {table for expression in expressions for table, _ in QUALIFIED_COLUMN.findall(expression)}

    @staticmethod
    def _first_table(expression: str) -> str:
        match = QUALIFIED_COLUMN.search(expression)
        if not match:
            raise ValueError(f"Cannot find source table in expression: {expression}")
        return match.group(1)

    @staticmethod
    def _where_clause(filters: list[MatchedFilter]) -> str:
        if not filters:
            return ""
        clauses = []
        for item in filters:
            value = item.value.replace("'", "''")
            if item.key == "year":
                clauses.append(f"strftime('%Y', {item.column}) = '{value}'")
            else:
                clauses.append(f"{item.column} {item.operator} '{value}'")
        return "WHERE " + " AND ".join(clauses)
