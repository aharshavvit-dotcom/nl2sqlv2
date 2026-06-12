from __future__ import annotations

import re
from typing import Any

import sqlglot
from sqlglot import exp

from .candidate_generator import CandidateGenerator
from .candidate_reranker import CandidateReranker
from .prediction_confidence import PredictionConfidenceCalculator
from .prediction_models import PredictionResult, SchemaMapping
from .runtime_join_planner import RuntimeJoinPlanner
from .runtime_schema_context import RuntimeSchemaContext
from .schema_aware_mapper import SchemaAwareMapper
from .slot_resolver import SlotResolver
from .template_selector import TemplateSelector


class PredictionOrchestrator:
    def __init__(self, top_k: int = 10, max_limit: int = 1000):
        self.top_k = top_k
        self.max_limit = max_limit
        self.generator = CandidateGenerator()
        self.reranker = CandidateReranker()
        self.selector = TemplateSelector()
        self.slot_resolver = SlotResolver()
        self.mapper = SchemaAwareMapper()
        self.join_planner = RuntimeJoinPlanner()
        self.confidence = PredictionConfidenceCalculator()

    def predict(
        self,
        question: str,
        schema: Any,
        retriever: Any,
        templates: Any | None = None,
        metric_synonyms: dict[str, Any] | None = None,
        dimension_synonyms: dict[str, Any] | None = None,
        validator: Any | None = None,
    ) -> PredictionResult:
        normalized_question = self._normalize_question(question)
        schema_context = RuntimeSchemaContext(schema)
        candidates = self.generator.generate_candidates(question, retriever, top_k=self.top_k)
        candidates = self.reranker.rerank_candidates(question, candidates, schema_context)
        selected_template = self.selector.select_template(candidates, question)
        slot_payload = self.slot_resolver.resolve_slots(
            question,
            selected_template,
            candidates,
            schema_context,
            {"metrics": metric_synonyms or {}, "dimensions": dimension_synonyms or {}},
        )
        slots = slot_payload["slots"]
        schema_mapping = self.mapper.map_slots_to_schema(slots, schema_context, metric_synonyms, dimension_synonyms)
        base_table = self._select_base_table(schema_mapping, slots)
        required_tables = self._required_tables(selected_template.get("template_id"), schema_mapping)
        join_plan = self.join_planner.plan_joins(schema_context, base_table, required_tables)
        sql = self._render_sql(selected_template.get("template_id"), slots, schema_mapping, join_plan)
        validation = self._validate_sql(sql, schema_context)
        confidence = self.confidence.calculate(
            {
                "candidates": candidates,
                "selected_template": selected_template,
                "slots": slots,
                "schema_mapping": schema_mapping,
                "join_plan": join_plan.model_dump(),
                "validation": validation,
            }
        )
        warnings = [
            *schema_mapping.warnings,
            *join_plan.warnings,
            *([] if validation.get("ok") else [validation.get("message", "validation failed")]),
        ]
        clarification = list(slot_payload["clarification_questions"])
        if confidence["confidence_tier"] == "low" and not clarification:
            clarification.append("Can you clarify the metric or grouping you want?")

        return PredictionResult(
            question=question,
            normalized_question=normalized_question,
            intent=selected_template.get("intent"),
            template_id=selected_template.get("template_id"),
            slots=slots,
            schema_mapping=schema_mapping.model_dump(),
            join_plan=join_plan.model_dump(),
            sql=sql,
            validation=validation,
            confidence=confidence["confidence"],
            confidence_tier=confidence["confidence_tier"],
            retrieved_candidates=[candidate.model_dump() for candidate in candidates],
            selected_candidate=candidates[0].model_dump() if candidates else None,
            warnings=warnings,
            clarification_questions=clarification,
            debug={
                "schema_context": schema_context.serialize_for_debug(),
                "template_selection": selected_template,
                "confidence_components": confidence["components"],
            },
        )

    @staticmethod
    def _normalize_question(question: str) -> str:
        return re.sub(r"\s+", " ", question.strip().lower())

    @staticmethod
    def _select_base_table(mapping: SchemaMapping, slots: dict[str, Any]) -> str:
        required = [
            table
            for table in [
                mapping.metric_table,
                mapping.dimension_table,
                mapping.entity_table,
                mapping.date_table,
            ]
            if table
        ]
        return RuntimeJoinPlanner.choose_base_table(mapping.metric_table, mapping.entity_table, required)

    @staticmethod
    def _required_tables(template_id: str | None, mapping: SchemaMapping) -> list[str]:
        tables = [mapping.metric_table or mapping.entity_table]
        if template_id in {"metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension", "count_by_dimension", "trend_by_date"}:
            tables.append(mapping.dimension_table)
        if template_id == "trend_by_date":
            tables.append(mapping.date_table)
        return [table for table in dict.fromkeys(tables) if table]

    def _render_sql(
        self,
        template_id: str | None,
        slots: dict[str, Any],
        mapping: SchemaMapping,
        join_plan: Any,
    ) -> str | None:
        if not mapping.metric_table and template_id not in {"show_records"}:
            return None
        limit = min(int(slots.get("limit", {}).get("value") or 100), self.max_limit)
        order = str(slots.get("sort_direction", {}).get("value") or "DESC")
        metric_expr = self._metric_expression(mapping)
        metric_alias = self._metric_alias(mapping)
        dimension_expr, dimension_alias = self._dimension_expression(template_id, slots, mapping)
        from_table = join_plan.base_table
        joins = join_plan.join_clause
        lines: list[str]

        if template_id == "count_records":
            lines = ["SELECT", "  COUNT(*) AS row_count", f"FROM {from_table}", joins, f"LIMIT {limit}"]
        elif template_id == "count_by_dimension":
            lines = [
                "SELECT",
                f"  {dimension_expr} AS {dimension_alias},",
                "  COUNT(*) AS row_count",
                f"FROM {from_table}",
                joins,
                f"GROUP BY {dimension_expr}",
                f"ORDER BY row_count {order}",
                f"LIMIT {limit}",
            ]
        elif template_id in {"metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension", "trend_by_date"}:
            if template_id == "bottom_n_metric_by_dimension":
                order = "ASC"
            elif template_id == "top_n_metric_by_dimension":
                order = "DESC"
            lines = [
                "SELECT",
                f"  {dimension_expr} AS {dimension_alias},",
                f"  {metric_expr} AS {metric_alias}",
                f"FROM {from_table}",
                joins,
                f"GROUP BY {dimension_expr}",
                f"ORDER BY {metric_alias if template_id != 'trend_by_date' else dimension_alias} {order if template_id != 'trend_by_date' else 'ASC'}",
                f"LIMIT {limit}",
            ]
        elif template_id == "simple_filter":
            select_columns = self._safe_select_columns(mapping, fallback_table=from_table)
            lines = ["SELECT", f"  {select_columns}", f"FROM {from_table}", joins, f"LIMIT {limit}"]
        elif template_id == "show_records":
            select_columns = self._safe_select_columns(mapping, fallback_table=from_table)
            lines = ["SELECT", f"  {select_columns}", f"FROM {from_table}", joins, f"LIMIT {limit}"]
        else:
            lines = ["SELECT", f"  {metric_expr} AS {metric_alias}", f"FROM {from_table}", joins, f"LIMIT {limit}"]
        return self._clean_sql("\n".join(line for line in lines if line))

    @staticmethod
    def _metric_expression(mapping: SchemaMapping) -> str:
        if mapping.metric_aggregation == "COUNT":
            return "COUNT(*)"
        return f"{mapping.metric_aggregation or 'SUM'}({mapping.metric_table}.{mapping.metric_column})"

    @staticmethod
    def _metric_alias(mapping: SchemaMapping) -> str:
        if mapping.metric_aggregation == "COUNT":
            return "row_count"
        if mapping.metric_name in {"sales", "revenue"}:
            return "revenue"
        return str(mapping.metric_name or "metric")

    @staticmethod
    def _dimension_expression(template_id: str | None, slots: dict[str, Any], mapping: SchemaMapping) -> tuple[str, str]:
        date_grain = slots.get("date_grain", {}).get("value")
        dimension = slots.get("dimension", {}).get("value")
        if template_id == "trend_by_date" or dimension in {"month", "year"}:
            fmt = "%Y" if date_grain == "year" or dimension == "year" else "%Y-%m"
            return f"strftime('{fmt}', {mapping.date_table}.{mapping.date_column})", str(dimension or date_grain or "date")
        return f"{mapping.dimension_table}.{mapping.dimension_column}", str(dimension or mapping.dimension_column or "dimension")

    @staticmethod
    def _safe_select_columns(mapping: SchemaMapping, fallback_table: str) -> str:
        if mapping.dimension_table and mapping.dimension_column:
            return f"{mapping.dimension_table}.{mapping.dimension_column}"
        if mapping.metric_table and mapping.metric_column:
            return f"{mapping.metric_table}.{mapping.metric_column}"
        return f"{fallback_table}.rowid"

    def _validate_sql(self, sql: str | None, schema_context: RuntimeSchemaContext) -> dict[str, Any]:
        if not sql:
            return {"ok": False, "message": "SQL was not generated", "checks": {"generated": False}}
        checks: dict[str, bool] = {}
        try:
            parsed = sqlglot.parse(sql, read="sqlite")
        except Exception as exc:
            return {"ok": False, "message": f"SQL parse failed: {exc}", "checks": {"parse": False}}
        checks["single_statement"] = len(parsed) == 1
        statement = parsed[0] if parsed else None
        checks["select_only"] = isinstance(statement, exp.Select)
        checks["no_mutation"] = not any(word in sql.upper() for word in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"])
        checks["no_select_star"] = "*" not in [item.sql() for item in statement.expressions] if isinstance(statement, exp.Select) else False
        limit = statement.args.get("limit") if isinstance(statement, exp.Select) else None
        checks["has_limit"] = limit is not None
        checks["limit_ok"] = True
        if limit is not None and limit.expression is not None:
            try:
                checks["limit_ok"] = int(limit.expression.name) <= self.max_limit
            except ValueError:
                checks["limit_ok"] = False
        table_ok = True
        column_ok = True
        sensitive_ok = True
        if statement is not None:
            for table in statement.find_all(exp.Table):
                table_ok = table_ok and schema_context.has_table(table.name)
            for column in statement.find_all(exp.Column):
                table = column.table
                name = column.name
                if table and schema_context.has_table(table):
                    column_ok = column_ok and schema_context.has_column(table, name)
                    if schema_context.has_column(table, name):
                        sensitive_ok = sensitive_ok and not schema_context.column_info(table, name)["is_sensitive"]
        checks["known_tables"] = table_ok
        checks["known_columns"] = column_ok
        checks["no_sensitive_columns"] = sensitive_ok
        ok = all(checks.values())
        return {"ok": ok, "message": "ok" if ok else "SQL validation failed", "checks": checks}

    @staticmethod
    def _clean_sql(sql: str) -> str:
        lines = [re.sub(r"\s+", " ", line).rstrip() for line in sql.splitlines()]
        return "\n".join(line for line in lines if line.strip())
