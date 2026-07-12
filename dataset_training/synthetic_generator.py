"""Synthetic query generator with provenance tracking and safeguards.

Gate 2: Data Readiness — generates synthetic training examples from schema
definitions, with full provenance tracking and capability coverage guarantees.

Design:
- Every synthetic example has a provenance record (schema source, template, seed)
- Schema renaming augmenter creates surface-form variants
- Coverage targets are per-capability, not aggregate
- Generated examples are validated through QueryIR v2 before acceptance
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ir.query_ir_v2_models import (
    AggregationExpression,
    BooleanPredicate,
    ColumnExpression,
    ComparisonPredicate,
    CTEDefinition,
    FromItem,
    InLiteralPredicate,
    JoinNode,
    LiteralExpression,
    LiteralValueType,
    NullPredicate,
    OrderByItem,
    QueryNode,
    SelectItem,
    WindowExpression,
    WindowSpecification,
)
from ir.query_ir_v2_validation import QueryIRV2Validator


class SyntheticProvenance:
    """Tracks the origin and generation parameters of every synthetic example."""

    def __init__(
        self,
        schema_source: str,
        template_id: str,
        seed: int,
        capability_tags: list[str],
        augmentation_chain: list[str] | None = None,
    ) -> None:
        self.schema_source = schema_source
        self.template_id = template_id
        self.seed = seed
        self.capability_tags = list(capability_tags)
        self.augmentation_chain = list(augmentation_chain or [])
        self.fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        blob = json.dumps({
            "schema_source": self.schema_source,
            "template_id": self.template_id,
            "seed": self.seed,
            "capability_tags": sorted(self.capability_tags),
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_source": self.schema_source,
            "template_id": self.template_id,
            "seed": self.seed,
            "capability_tags": self.capability_tags,
            "augmentation_chain": self.augmentation_chain,
            "fingerprint": self.fingerprint,
        }


@dataclass
class SyntheticExample:
    """A single synthetic training example."""
    question: str
    query_ir: QueryNode
    sql: str
    provenance: SyntheticProvenance
    is_valid: bool = True
    validation_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "query_ir": self.query_ir.model_dump(mode="json"),
            "sql": self.sql,
            "provenance": self.provenance.to_dict(),
            "is_valid": self.is_valid,
            "validation_issues": self.validation_issues,
        }


@dataclass
class SchemaDefinition:
    """Minimal schema definition for synthetic generation."""
    name: str
    tables: dict[str, list[str]]  # table_name -> [column_names]
    primary_keys: dict[str, str] = field(default_factory=dict)  # table -> pk_column
    foreign_keys: list[tuple[str, str, str, str]] = field(default_factory=list)  # (src_table, src_col, tgt_table, tgt_col)

    @property
    def all_columns(self) -> list[tuple[str, str]]:
        return [
            (table, column)
            for table, columns in self.tables.items()
            for column in columns
        ]


class CapabilityTag(str, Enum):
    """Standardized capability tags for coverage tracking."""
    SIMPLE_SELECT = "SIMPLE_SELECT"
    WHERE_FILTER = "WHERE_FILTER"
    AND_FILTER = "AND_FILTER"
    OR_FILTER = "OR_FILTER"
    IN_FILTER = "IN_FILTER"
    NULL_CHECK = "NULL_CHECK"
    LIKE_FILTER = "LIKE_FILTER"
    JOIN = "JOIN"
    GROUP_BY = "GROUP_BY"
    HAVING = "HAVING"
    ORDER_BY = "ORDER_BY"
    AGGREGATION = "AGGREGATION"
    DISTINCT = "DISTINCT"
    SUBQUERY = "SUBQUERY"
    CTE = "CTE"
    WINDOW_FUNCTION = "WINDOW_FUNCTION"
    CASE_EXPRESSION = "CASE_EXPRESSION"
    BETWEEN = "BETWEEN"
    LIMIT = "LIMIT"
    MULTI_TABLE = "MULTI_TABLE"


@dataclass
class CoverageTarget:
    """Per-capability coverage target."""
    capability: CapabilityTag
    min_examples: int
    current_count: int = 0

    @property
    def satisfied(self) -> bool:
        return self.current_count >= self.min_examples

    @property
    def deficit(self) -> int:
        return max(0, self.min_examples - self.current_count)


# ── Query Templates ──────────────────────────────────────────────────

class QueryTemplate:
    """Generates QueryIR v2 instances from schema definitions."""

    def __init__(self, template_id: str, capability_tags: list[CapabilityTag]):
        self.template_id = template_id
        self.capability_tags = capability_tags

    def generate(
        self,
        schema: SchemaDefinition,
        rng: random.Random,
    ) -> tuple[str, QueryNode, str] | None:
        """Returns (question, query_ir, sql) or None if template can't apply."""
        raise NotImplementedError


class SimpleSelectTemplate(QueryTemplate):
    def __init__(self):
        super().__init__("simple_select", [CapabilityTag.SIMPLE_SELECT, CapabilityTag.LIMIT])

    def generate(self, schema: SchemaDefinition, rng: random.Random):
        table = rng.choice(list(schema.tables.keys()))
        columns = schema.tables[table]
        selected = rng.sample(columns, min(rng.randint(2, 4), len(columns)))
        limit = rng.choice([10, 25, 50, 100])

        question = f"Show the {', '.join(selected)} from {table}"
        select_items = [
            SelectItem(expression=ColumnExpression(table=table, column=col), alias=col)
            for col in selected
        ]
        query_ir = QueryNode(
            question=question,
            intent="show_records",
            template_id="simple_select",
            dialect="sqlite",
            from_item=FromItem(table=table),
            required_tables=[table],
            select_items=select_items,
            limit=limit,
        )
        sql = f'SELECT {", ".join(f"{table}.{c}" for c in selected)} FROM {table} LIMIT {limit}'
        return question, query_ir, sql


class WhereFilterTemplate(QueryTemplate):
    def __init__(self):
        super().__init__("where_filter", [CapabilityTag.WHERE_FILTER, CapabilityTag.LIMIT])

    def generate(self, schema: SchemaDefinition, rng: random.Random):
        table = rng.choice(list(schema.tables.keys()))
        columns = schema.tables[table]
        if len(columns) < 2:
            return None
        filter_col = rng.choice(columns)
        selected = [c for c in columns if c != filter_col][:3]
        if not selected:
            selected = columns[:2]
        limit = rng.choice([10, 25, 50])
        operators = ["=", ">", "<", ">=", "<="]
        op = rng.choice(operators)
        value = rng.randint(1, 1000)

        question = f"Show {', '.join(selected)} from {table} where {filter_col} {op} {value}"
        select_items = [
            SelectItem(expression=ColumnExpression(table=table, column=c), alias=c)
            for c in selected
        ]
        query_ir = QueryNode(
            question=question,
            intent="simple_filter",
            template_id="where_filter",
            dialect="sqlite",
            from_item=FromItem(table=table),
            required_tables=[table],
            select_items=select_items,
            where=ComparisonPredicate(
                left=ColumnExpression(table=table, column=filter_col),
                operator=op,
                right=LiteralExpression(value=value, value_type=LiteralValueType.INTEGER),
            ),
            limit=limit,
        )
        sql = f'SELECT {", ".join(f"{table}.{c}" for c in selected)} FROM {table} WHERE {table}.{filter_col} {op} {value} LIMIT {limit}'
        return question, query_ir, sql


class AggregationTemplate(QueryTemplate):
    def __init__(self):
        super().__init__("aggregation", [CapabilityTag.AGGREGATION, CapabilityTag.GROUP_BY, CapabilityTag.LIMIT])

    def generate(self, schema: SchemaDefinition, rng: random.Random):
        table = rng.choice(list(schema.tables.keys()))
        columns = schema.tables[table]
        if len(columns) < 2:
            return None
        group_col = rng.choice(columns)
        agg_col = rng.choice([c for c in columns if c != group_col] or columns)
        agg_func = rng.choice(["COUNT", "SUM", "AVG", "MAX", "MIN"])
        limit = rng.choice([10, 25])

        question = f"Show {agg_func.lower()} of {agg_col} for each {group_col} in {table}"
        query_ir = QueryNode(
            question=question,
            intent="aggregate",
            template_id="aggregation",
            dialect="sqlite",
            from_item=FromItem(table=table),
            required_tables=[table],
            select_items=[
                SelectItem(expression=ColumnExpression(table=table, column=group_col), alias=group_col),
                SelectItem(
                    expression=AggregationExpression(
                        function=agg_func,
                        argument=ColumnExpression(table=table, column=agg_col),
                    ),
                    alias=f"{agg_func.lower()}_{agg_col}",
                ),
            ],
            group_by=[ColumnExpression(table=table, column=group_col)],
            select_mode="aggregate",
            limit=limit,
        )
        sql = f'SELECT {table}.{group_col}, {agg_func}({table}.{agg_col}) AS {agg_func.lower()}_{agg_col} FROM {table} GROUP BY {table}.{group_col} LIMIT {limit}'
        return question, query_ir, sql


class JoinTemplate(QueryTemplate):
    def __init__(self):
        super().__init__("join", [CapabilityTag.JOIN, CapabilityTag.MULTI_TABLE, CapabilityTag.LIMIT])

    def generate(self, schema: SchemaDefinition, rng: random.Random):
        if not schema.foreign_keys:
            return None
        fk = rng.choice(schema.foreign_keys)
        src_table, src_col, tgt_table, tgt_col = fk
        src_cols = schema.tables.get(src_table, [])
        tgt_cols = schema.tables.get(tgt_table, [])
        if not src_cols or not tgt_cols:
            return None
        selected_src = rng.sample(src_cols, min(2, len(src_cols)))
        selected_tgt = rng.sample(tgt_cols, min(2, len(tgt_cols)))
        limit = rng.choice([10, 25, 50])

        question = f"Show {', '.join(selected_src)} from {src_table} with {', '.join(selected_tgt)} from {tgt_table}"
        select_items = (
            [SelectItem(expression=ColumnExpression(table=src_table, column=c), alias=f"{src_table}_{c}") for c in selected_src]
            + [SelectItem(expression=ColumnExpression(table=tgt_table, column=c), alias=f"{tgt_table}_{c}") for c in selected_tgt]
        )
        query_ir = QueryNode(
            question=question,
            intent="show_records",
            template_id="join",
            dialect="sqlite",
            from_item=FromItem(table=src_table),
            required_tables=[src_table, tgt_table],
            select_items=select_items,
            joins=[JoinNode(
                join_type="INNER",
                right=FromItem(table=tgt_table),
                on=ComparisonPredicate(
                    left=ColumnExpression(table=src_table, column=src_col),
                    operator="=",
                    right=ColumnExpression(table=tgt_table, column=tgt_col),
                ),
                path_order=1,
            )],
            limit=limit,
        )
        on_clause = f"{src_table}.{src_col} = {tgt_table}.{tgt_col}"
        sql_select = ", ".join(
            [f"{src_table}.{c}" for c in selected_src] + [f"{tgt_table}.{c}" for c in selected_tgt]
        )
        sql = f'SELECT {sql_select} FROM {src_table} JOIN {tgt_table} ON {on_clause} LIMIT {limit}'
        return question, query_ir, sql


class OrderByTemplate(QueryTemplate):
    def __init__(self):
        super().__init__("order_by", [CapabilityTag.ORDER_BY, CapabilityTag.LIMIT])

    def generate(self, schema: SchemaDefinition, rng: random.Random):
        table = rng.choice(list(schema.tables.keys()))
        columns = schema.tables[table]
        if len(columns) < 2:
            return None
        order_col = rng.choice(columns)
        direction = rng.choice(["ASC", "DESC"])
        selected = rng.sample(columns, min(3, len(columns)))
        limit = rng.choice([10, 25, 50])

        question = f"Show {', '.join(selected)} from {table} ordered by {order_col} {'ascending' if direction == 'ASC' else 'descending'}"
        query_ir = QueryNode(
            question=question,
            intent="show_records",
            template_id="order_by",
            dialect="sqlite",
            from_item=FromItem(table=table),
            required_tables=[table],
            select_items=[SelectItem(expression=ColumnExpression(table=table, column=c), alias=c) for c in selected],
            order_by=[OrderByItem(expression=ColumnExpression(table=table, column=order_col), direction=direction)],
            limit=limit,
        )
        sql = f'SELECT {", ".join(f"{table}.{c}" for c in selected)} FROM {table} ORDER BY {table}.{order_col} {direction} LIMIT {limit}'
        return question, query_ir, sql


# ── Template Registry ────────────────────────────────────────────────

DEFAULT_TEMPLATES: list[QueryTemplate] = [
    SimpleSelectTemplate(),
    WhereFilterTemplate(),
    AggregationTemplate(),
    JoinTemplate(),
    OrderByTemplate(),
]


# ── Schema Renaming Augmenter ────────────────────────────────────────

class SchemaRenamingAugmenter:
    """Creates surface-form variants of schemas to increase diversity.

    Renames tables and columns using realistic abbreviations and synonyms
    to ensure the model doesn't overfit to specific naming conventions.
    """

    ABBREVIATIONS = {
        "customer": ["cust", "client", "buyer"],
        "order": ["ord", "purchase", "transaction"],
        "product": ["prod", "item", "sku"],
        "employee": ["emp", "staff", "worker"],
        "department": ["dept", "division", "unit"],
        "category": ["cat", "group", "type"],
        "payment": ["pmt", "pay", "txn"],
        "invoice": ["inv", "bill", "receipt"],
        "inventory": ["inv", "stock", "warehouse"],
        "price": ["cost", "rate", "amount"],
        "name": ["label", "title", "description"],
        "date": ["dt", "timestamp", "created_at"],
        "quantity": ["qty", "count", "num"],
        "total": ["sum", "amount", "value"],
        "status": ["state", "condition", "flag"],
        "address": ["addr", "location", "place"],
    }

    def augment(
        self,
        schema: SchemaDefinition,
        rng: random.Random,
        rename_probability: float = 0.3,
    ) -> tuple[SchemaDefinition, dict[str, str]]:
        """Returns (renamed_schema, rename_map)."""
        rename_map: dict[str, str] = {}
        new_tables: dict[str, list[str]] = {}

        for table, columns in schema.tables.items():
            new_table = self._maybe_rename(table, rng, rename_probability)
            rename_map[table] = new_table
            new_columns = []
            for col in columns:
                new_col = self._maybe_rename(col, rng, rename_probability)
                rename_map[f"{table}.{col}"] = f"{new_table}.{new_col}"
                new_columns.append(new_col)
            new_tables[new_table] = new_columns

        new_pks = {rename_map.get(t, t): schema.primary_keys.get(t, "") for t in schema.primary_keys}
        new_fks = [
            (rename_map.get(st, st), sc, rename_map.get(tt, tt), tc)
            for st, sc, tt, tc in schema.foreign_keys
        ]

        return SchemaDefinition(
            name=f"{schema.name}_augmented_{rng.randint(0, 9999)}",
            tables=new_tables,
            primary_keys=new_pks,
            foreign_keys=new_fks,
        ), rename_map

    def _maybe_rename(self, name: str, rng: random.Random, probability: float) -> str:
        if rng.random() > probability:
            return name
        lower = name.lower()
        for key, variants in self.ABBREVIATIONS.items():
            if key in lower:
                replacement = rng.choice(variants)
                return re.sub(re.escape(key), replacement, lower, count=1)
        return name


# ── Synthetic Generator ──────────────────────────────────────────────

class SyntheticQueryGenerator:
    """Main orchestrator: generates diverse synthetic examples with coverage tracking."""

    def __init__(
        self,
        schemas: list[SchemaDefinition],
        templates: list[QueryTemplate] | None = None,
        seed: int = 42,
        coverage_targets: dict[CapabilityTag, int] | None = None,
    ) -> None:
        self.schemas = schemas
        self.templates = templates or DEFAULT_TEMPLATES
        self.rng = random.Random(seed)
        self.seed = seed
        self.validator = QueryIRV2Validator()
        self.augmenter = SchemaRenamingAugmenter()

        # Default coverage targets
        default_targets = {cap: 50 for cap in CapabilityTag}
        if coverage_targets:
            default_targets.update(coverage_targets)
        self.coverage = {
            cap: CoverageTarget(capability=cap, min_examples=target)
            for cap, target in default_targets.items()
        }

    def generate(self, max_examples: int = 1000) -> list[SyntheticExample]:
        """Generate up to max_examples with provenance and coverage tracking."""
        examples: list[SyntheticExample] = []
        attempts = 0
        max_attempts = max_examples * 5

        while len(examples) < max_examples and attempts < max_attempts:
            attempts += 1

            # Pick a template — prefer those with unsatisfied coverage targets
            template = self._select_template()
            schema = self.rng.choice(self.schemas)

            # Augment schema with probability 0.3
            if self.rng.random() < 0.3:
                schema, rename_map = self.augmenter.augment(schema, self.rng)
                aug_chain = ["schema_rename"]
            else:
                aug_chain = []

            result = template.generate(schema, self.rng)
            if result is None:
                continue

            question, query_ir, sql = result
            provenance = SyntheticProvenance(
                schema_source=schema.name,
                template_id=template.template_id,
                seed=self.seed + attempts,
                capability_tags=[t.value for t in template.capability_tags],
                augmentation_chain=aug_chain,
            )

            # Validate
            validation = self.validator.validate(query_ir)
            example = SyntheticExample(
                question=question,
                query_ir=query_ir,
                sql=sql,
                provenance=provenance,
                is_valid=validation.is_valid,
                validation_issues=list(validation.errors),
            )

            if example.is_valid:
                examples.append(example)
                for tag in template.capability_tags:
                    if tag in self.coverage:
                        self.coverage[tag].current_count += 1

        return examples

    def _select_template(self) -> QueryTemplate:
        """Prefer templates whose capabilities have the largest deficit."""
        deficits: list[tuple[QueryTemplate, int]] = []
        for template in self.templates:
            max_deficit = max(
                (self.coverage.get(tag, CoverageTarget(tag, 0)).deficit for tag in template.capability_tags),
                default=0,
            )
            deficits.append((template, max_deficit))

        # Sort by deficit descending, pick from top 3
        deficits.sort(key=lambda x: x[1], reverse=True)
        candidates = deficits[:3]
        return self.rng.choice([t for t, _ in candidates])

    def coverage_report(self) -> dict[str, Any]:
        """Generate a coverage report with per-capability stats."""
        items = []
        for cap, target in sorted(self.coverage.items(), key=lambda x: x[0].value):
            items.append({
                "capability": cap.value,
                "target": target.min_examples,
                "current": target.current_count,
                "satisfied": target.satisfied,
                "deficit": target.deficit,
            })
        satisfied = sum(1 for t in self.coverage.values() if t.satisfied)
        total = len(self.coverage)
        return {
            "total_capabilities": total,
            "satisfied": satisfied,
            "unsatisfied": total - satisfied,
            "coverage_rate": satisfied / total if total > 0 else 0.0,
            "details": items,
        }


__all__ = [
    "CapabilityTag",
    "CoverageTarget",
    "SchemaDefinition",
    "SchemaRenamingAugmenter",
    "SyntheticExample",
    "SyntheticProvenance",
    "SyntheticQueryGenerator",
]
