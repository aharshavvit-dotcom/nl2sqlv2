from __future__ import annotations

import re
from typing import Any

from ir.ir_to_sql_renderer import IRToSQLRenderer
from ir.ir_validator import IRValidator
from ir.option_c_to_ir import OptionCToIRConverter
from ir.semantic_metric_resolver import SemanticMetricResolver
from validation.sql_validator import SQLValidator

from .candidate_generator import CandidateGenerator
from .candidate_reranker import CandidateReranker
from .prediction_confidence import PredictionConfidenceCalculator
from .prediction_models import PredictionResult, SchemaMapping
from .runtime_join_planner import RuntimeJoinPlanner
from .runtime_schema_context import RuntimeSchemaContext
from .schema_aware_mapper import SchemaAwareMapper
from .slot_resolver import SlotResolver
from .synonym_loader import load_metric_dimension_maps, normalize_section
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
        self.semantic_metric_resolver = SemanticMetricResolver()
        self.ir_converter = OptionCToIRConverter()
        self.ir_validator = IRValidator(max_limit=max_limit)
        self.sql_renderer = IRToSQLRenderer(max_limit=max_limit)
        self.sql_validator = SQLValidator()
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
        metric_synonyms, dimension_synonyms = self._synonym_maps(metric_synonyms, dimension_synonyms)

        candidates = self.generator.generate_candidates(question, retriever, top_k=self.top_k)
        candidates = self.reranker.rerank_candidates(question, candidates, schema_context)
        selected_template = self.selector.select_template(candidates, question)
        slot_payload = self.slot_resolver.resolve_slots(
            question,
            selected_template,
            candidates,
            schema_context,
            {"metrics": metric_synonyms, "dimensions": dimension_synonyms},
        )
        slots = slot_payload["slots"]
        schema_mapping = self.mapper.map_slots_to_schema(slots, schema_context, metric_synonyms, dimension_synonyms)
        self._apply_semantic_metric_resolution(schema_mapping, schema_context)
        base_table = self._select_base_table(schema_mapping, slots)
        required_tables = self._required_tables(selected_template.get("template_id"), schema_mapping)
        join_plan = self.join_planner.plan_joins(schema_context, base_table, required_tables)

        query_ir = self.ir_converter.convert(
            question=question,
            normalized_question=normalized_question,
            intent=selected_template.get("intent") or selected_template.get("template_id") or "unknown",
            template_id=selected_template.get("template_id"),
            slots=slots,
            schema_mapping=schema_mapping,
            join_plan=join_plan,
            validation_context={"schema_context": schema_context.serialize_for_debug()},
            dialect=schema_context.dialect,
        )
        ir_validation = self.ir_validator.validate(query_ir, schema=schema)
        sql = self.sql_renderer.render(query_ir) if ir_validation.is_valid else None
        sql_validation = self.sql_validator.validate(sql, schema=schema, max_limit=self.max_limit, dialect=schema_context.dialect)

        confidence = self.confidence.calculate(
            {
                "candidates": candidates,
                "selected_template": selected_template,
                "slots": slots,
                "schema_mapping": schema_mapping,
                "join_plan": join_plan.model_dump(),
                "ir_validation": ir_validation.model_dump(),
                "validation": sql_validation,
                "warnings": [
                    *schema_mapping.warnings,
                    *join_plan.warnings,
                    *query_ir.warnings,
                    *ir_validation.warnings,
                    *ir_validation.errors,
                    *([] if sql_validation.get("is_valid") else sql_validation.get("issues", [])),
                ],
            }
        )
        warnings = [
            *schema_mapping.warnings,
            *join_plan.warnings,
            *query_ir.warnings,
            *ir_validation.warnings,
            *ir_validation.errors,
            *([] if sql_validation.get("is_valid") else sql_validation.get("issues", [])),
        ]
        clarification = self._clarification_questions(confidence["confidence"], selected_template.get("template_id"), slots)

        return PredictionResult(
            question=question,
            normalized_question=normalized_question,
            intent=selected_template.get("intent"),
            template_id=selected_template.get("template_id"),
            slots=slots,
            schema_mapping=schema_mapping.model_dump(),
            join_plan=join_plan.model_dump(),
            query_ir=query_ir.model_dump(),
            ir_validation=ir_validation.model_dump(),
            sql=sql,
            validation=sql_validation,
            confidence=confidence["confidence"],
            confidence_tier=confidence["confidence_tier"],
            retrieved_candidates=[candidate.model_dump() for candidate in candidates],
            selected_candidate=candidates[0].model_dump() if candidates else None,
            warnings=list(dict.fromkeys(str(warning) for warning in warnings if warning)),
            clarification_questions=clarification,
            debug={
                "schema_context": schema_context.serialize_for_debug(),
                "template_selection": selected_template,
                "confidence_breakdown": confidence["confidence_breakdown"],
                "confidence_components": confidence["confidence_breakdown"],
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
                mapping.filter_table,
            ]
            if table
        ]
        return RuntimeJoinPlanner.choose_base_table(mapping.metric_table, mapping.entity_table, required)

    @staticmethod
    def _required_tables(template_id: str | None, mapping: SchemaMapping) -> list[str]:
        tables = [mapping.metric_table or mapping.entity_table]
        if template_id in {"metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension", "count_by_dimension", "trend_by_date"}:
            tables.append(mapping.dimension_table)
        if mapping.date_table:
            tables.append(mapping.date_table)
        if mapping.filter_table:
            tables.append(mapping.filter_table)
        return [table for table in dict.fromkeys(tables) if table]

    def _apply_semantic_metric_resolution(
        self,
        mapping: SchemaMapping,
        schema_context: RuntimeSchemaContext,
    ) -> None:
        resolution = self.semantic_metric_resolver.resolve_metric_expression(
            metric_name=mapping.metric_name or "",
            dimension_name=mapping.dimension_name,
            schema_context=schema_context,
            current_metric_table=mapping.metric_table,
            current_metric_column=mapping.metric_column,
        )
        if resolution.get("metric_expression"):
            mapping.metric_table = resolution.get("metric_table")
            mapping.metric_column = resolution.get("metric_column")
            mapping.metric_expression = resolution.get("metric_expression")
            mapping.metric_aggregation = resolution.get("metric_aggregation") or mapping.metric_aggregation
            mapping.metric_alias = resolution.get("metric_alias") or mapping.metric_alias
            mapping.match_scores["semantic_metric"] = 1.0
        if resolution.get("semantic_grain_risk"):
            mapping.semantic_grain_risk = True
            mapping.match_scores["semantic_metric"] = min(mapping.match_scores.get("semantic_metric", 0.4), 0.4)
        for warning in resolution.get("warnings", []):
            if warning not in mapping.warnings:
                mapping.warnings.append(warning)

    @staticmethod
    def _synonym_maps(
        metric_synonyms: dict[str, Any] | None,
        dimension_synonyms: dict[str, Any] | None,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        if metric_synonyms or dimension_synonyms:
            return normalize_section(metric_synonyms or {}), normalize_section(dimension_synonyms or {})
        return load_metric_dimension_maps()

    @staticmethod
    def _clarification_questions(confidence: float, template_id: str | None, slots: dict[str, Any]) -> list[str]:
        if confidence >= 0.60:
            return []
        metric = slots.get("metric") or {}
        dimension = slots.get("dimension") or {}
        metric_value = metric.get("value") if isinstance(metric, dict) else None
        dimension_value = dimension.get("value") if isinstance(dimension, dict) else None
        metric_conf = float(metric.get("confidence", 0.0)) if isinstance(metric, dict) else 0.0
        dimension_conf = float(dimension.get("confidence", 0.0)) if isinstance(dimension, dict) else 0.0
        needs_metric = template_id in {
            "metric_summary",
            "metric_by_dimension",
            "top_n_metric_by_dimension",
            "bottom_n_metric_by_dimension",
            "trend_by_date",
        }
        needs_dimension = template_id in {
            "metric_by_dimension",
            "top_n_metric_by_dimension",
            "bottom_n_metric_by_dimension",
            "count_by_dimension",
        }
        questions = []
        if needs_metric and (metric_value is None or metric_conf < 0.55):
            questions.append("Which metric or value should I aggregate? (e.g. revenue, count, quantity)")
        if needs_dimension and (dimension_value is None or dimension_conf < 0.55):
            questions.append("Which column should I group by? (e.g. customer, product, region)")
        if not questions and confidence < 0.60:
            questions.append(f"Can you rephrase your question? I found {metric_value} and {dimension_value} but the pattern is unclear.")
        return questions
