from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from clarification import AmbiguityDetector, ClarificationQuestionBuilder
from generic_planner import SchemaProfile, TableIntentResolver, infer_join_policy
from neural_ir.calibration import choose_route, load_hybrid_calibration
from ir.ir_to_sql_renderer import IRToSQLRenderer
from ir.ir_validator import IRValidator
from ir.option_c_to_ir import OptionCToIRConverter
from ir.semantic_metric_resolver import SemanticMetricResolver
from semantic_layer import build_semantic_profile
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


# --- Internal name mapping helpers ---
_SOURCE_MODEL_MAP = {
    "option_c": "retrieval_ir",
    "option_a": "neural_ir",
    "hybrid": "adaptive_router",
}

def _normalize_source(name: str) -> str:
    return _SOURCE_MODEL_MAP.get(name, name)


class PredictionOrchestrator:
    def __init__(
        self,
        top_k: int = 10,
        max_limit: int = 1000,
        neural_ir_model_dir: str | Path | None = None,
        use_neural_ir_fallback: bool = True,
        neural_ir_threshold: float = 0.80,
        # Backward-compatible aliases
        option_a_model_dir: str | Path | None = None,
        use_option_a_fallback: bool | None = None,
        option_a_threshold: float | None = None,
    ):
        self.top_k = top_k
        self.max_limit = max_limit
        root = Path(__file__).resolve().parents[1]

        # Accept old param names for backward compat
        _model_dir = neural_ir_model_dir or option_a_model_dir
        _use_fallback = use_neural_ir_fallback if use_option_a_fallback is None else use_option_a_fallback
        _threshold = neural_ir_threshold if option_a_threshold is None else option_a_threshold

        self._explicit_neural_ir_model_dir = _model_dir is not None
        self.neural_ir_v2_model_dir = root / "artifacts" / "neural_ir_model"
        self.neural_ir_v1_model_dir = root / "artifacts" / "neural_ir_model"
        # Fallback to old folder names if new ones don't exist
        if not self.neural_ir_v2_model_dir.exists():
            self.neural_ir_v2_model_dir = root / "artifacts" / "option_a_ir_model_v2"
        if not self.neural_ir_v1_model_dir.exists():
            self.neural_ir_v1_model_dir = root / "artifacts" / "option_a_ir_model"
        self.neural_ir_model_dir = Path(_model_dir) if _model_dir else self._default_neural_ir_model_dir()
        self.hybrid_calibration = load_hybrid_calibration(self.neural_ir_model_dir / "hybrid_calibration.json")
        self.use_neural_ir_fallback = _use_fallback
        self.neural_ir_threshold = float(self.hybrid_calibration.get("retrieval_ir_high_confidence_threshold",
                                        self.hybrid_calibration.get("option_c_high_confidence_threshold", _threshold)))
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

    # Backward-compatible properties
    @property
    def option_a_model_dir(self) -> Path:
        """Deprecated alias. Use ``neural_ir_model_dir``."""
        return self.neural_ir_model_dir

    @property
    def use_option_a_fallback(self) -> bool:
        """Deprecated alias. Use ``use_neural_ir_fallback``."""
        return self.use_neural_ir_fallback

    @property
    def option_a_threshold(self) -> float:
        """Deprecated alias. Use ``neural_ir_threshold``."""
        return self.neural_ir_threshold

    def predict(
        self,
        question: str,
        schema: Any,
        retriever: Any,
        templates: Any | None = None,
        metric_synonyms: dict[str, Any] | None = None,
        dimension_synonyms: dict[str, Any] | None = None,
        validator: Any | None = None,
        use_neural_ir_fallback: bool | None = None,
        use_option_a_fallback: bool | None = None,
    ) -> PredictionResult:
        normalized_question = self._normalize_question(question)
        schema_context = RuntimeSchemaContext(schema)
        semantic_profile = self._build_semantic_profile(schema)

        direct_result = TableIntentResolver(SchemaProfile(schema)).resolve(question)
        clarification = None if direct_result.handled else self._semantic_clarification(
            question=question,
            normalized_question=normalized_question,
            schema=schema,
            semantic_profile=semantic_profile,
            direct_result=direct_result,
        )
        if clarification is not None:
            return clarification

        if direct_result.handled and direct_result.query_ir is not None:
            return self._direct_planner_prediction(
                question=question,
                normalized_question=normalized_question,
                schema=schema,
                schema_context=schema_context,
                direct_result=direct_result,
                semantic_profile=semantic_profile,
            )

        metric_synonyms, dimension_synonyms = self._synonym_maps(metric_synonyms, dimension_synonyms)

        candidates = self.generator.generate_candidates(
            question,
            retriever,
            top_k=self.top_k,
            schema=schema_context.serialize_for_debug(),
        )
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
        template_id = selected_template.get("template_id")
        schema_mapping = self.mapper.map_slots_to_schema(
            slots,
            schema_context,
            metric_synonyms,
            dimension_synonyms,
            template_id=template_id,
            semantic_profile=semantic_profile,
        )
        if template_id not in {"show_records", "simple_filter", "count_records"}:
            self._apply_semantic_metric_resolution(schema_mapping, schema_context)
        base_table = self._select_base_table(selected_template.get("template_id"), schema_mapping)
        required_tables = self._required_tables(selected_template.get("template_id"), schema_mapping)
        join_policy = infer_join_policy(question, selected_template.get("template_id") or selected_template.get("intent") or "")
        join_plan = self.join_planner.plan_joins(schema_context, base_table, required_tables, join_policy=join_policy)

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

        retrieval_ir_result = PredictionResult(
            question=question,
            normalized_question=normalized_question,
            source_model="retrieval_ir",
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
            router_decision={},
            selected_query_ir=query_ir.model_dump(),
            validation_summary={"ir_validation": ir_validation.model_dump(), "sql_validation": sql_validation},
            confidence_breakdown=confidence["confidence_breakdown"],
            debug={
                "schema_context": schema_context.serialize_for_debug(),
                "semantic_profile_summary": self._semantic_summary(semantic_profile),
                "template_selection": selected_template,
                "confidence_breakdown": confidence["confidence_breakdown"],
                "confidence_components": confidence["confidence_breakdown"],
            },
        )

        # Resolve fallback flag: accept both old and new param name
        _fallback = use_neural_ir_fallback if use_neural_ir_fallback is not None else use_option_a_fallback
        return self._maybe_neural_ir_fallback(
            retrieval_ir_result=retrieval_ir_result,
            question=question,
            schema=schema,
            enabled=self.use_neural_ir_fallback if _fallback is None else _fallback,
        )

    def _direct_planner_prediction(
        self,
        question: str,
        normalized_question: str,
        schema: Any,
        schema_context: RuntimeSchemaContext,
        direct_result: Any,
        semantic_profile: dict[str, Any] | None = None,
    ) -> PredictionResult:
        query_ir = direct_result.query_ir
        ir_validation = self.ir_validator.validate(query_ir, schema=schema)
        sql = self.sql_renderer.render(query_ir, dialect=schema_context.dialect) if ir_validation.is_valid else None
        sql_validation = self.sql_validator.validate(sql, schema=schema, max_limit=self.max_limit, dialect=schema_context.dialect)

        warnings = [
            *direct_result.warnings,
            *query_ir.warnings,
            *ir_validation.warnings,
            *ir_validation.errors,
            *([] if sql_validation.get("is_valid") else sql_validation.get("issues", [])),
        ]
        confidence = float(direct_result.confidence or 0.0)
        if not ir_validation.is_valid or not sql_validation.get("is_valid", sql_validation.get("ok", False)):
            confidence = min(confidence, 0.59)
        planner_debug = {
            **(direct_result.debug or {}),
            "intent": query_ir.intent,
            "base_table": query_ir.base_table,
            "required_tables": query_ir.required_tables,
            "join_policy": query_ir.metadata.get("join_policy", "none"),
            "safe_selected_columns": query_ir.metadata.get("safe_selected_columns", []),
            "bypass_reason": direct_result.reason,
        }
        join_plan = {
            "base_table": query_ir.base_table,
            "required_tables": query_ir.required_tables,
            "join_clause": "",
            "join_steps": [],
            "confidence": 1.0,
            "warnings": [],
            "join_policy": "none",
        }
        schema_mapping = {
            "base_table": query_ir.base_table,
            "metric_table": query_ir.metrics[0].table if query_ir.metrics else None,
            "metric_column": query_ir.metrics[0].column if query_ir.metrics else None,
            "filter_table": query_ir.filters[0].table if query_ir.filters else None,
            "filter_column": query_ir.filters[0].column if query_ir.filters else None,
            "match_scores": {"generic_direct_planner": confidence},
            "warnings": list(dict.fromkeys(str(warning) for warning in warnings if warning)),
        }
        return PredictionResult(
            question=question,
            normalized_question=normalized_question,
            source_model="generic_direct_planner",
            intent=query_ir.intent,
            template_id=query_ir.template_id,
            slots={},
            schema_mapping=schema_mapping,
            join_plan=join_plan,
            query_ir=query_ir.model_dump(),
            ir_validation=ir_validation.model_dump(),
            sql=sql,
            validation=sql_validation,
            confidence=confidence,
            confidence_tier=self._confidence_tier(confidence),
            retrieved_candidates=[],
            selected_candidate=None,
            warnings=list(dict.fromkeys(str(warning) for warning in warnings if warning)),
            clarification_questions=[] if sql_validation.get("is_valid", sql_validation.get("ok", False)) else ["Choose specific non-sensitive columns for this table."],
            router_decision={
                "selected": "generic_direct_planner",
                "reason": direct_result.reason or "schema-safe direct query",
                "retrieval_ir_valid": None,
                "neural_ir_valid": None,
            },
            selected_query_ir=query_ir.model_dump(),
            validation_summary={"ir_validation": ir_validation.model_dump(), "sql_validation": sql_validation},
            confidence_breakdown={
                "final": confidence,
                "generic_direct_planner": float(direct_result.confidence or 0.0),
                "ir_validation": 1.0 if ir_validation.is_valid else 0.0,
                "sql_validation": 1.0 if sql_validation.get("is_valid", sql_validation.get("ok", False)) else 0.0,
                "join_planning": 1.0,
            },
            planner_debug=planner_debug,
            debug={
                "schema_context": schema_context.serialize_for_debug(),
                "semantic_profile_summary": self._semantic_summary(semantic_profile),
                "generic_planner": planner_debug,
                "router_decision": {
                    "selected": "generic_direct_planner",
                    "reason": direct_result.reason or "schema-safe direct query",
                },
            },
        )

    def _semantic_clarification(
        self,
        question: str,
        normalized_question: str,
        schema: Any,
        semantic_profile: dict[str, Any] | None,
        direct_result: Any,
    ) -> PredictionResult | None:
        if semantic_profile is None:
            return None
        table_match = (direct_result.debug or {}).get("table_match") if getattr(direct_result, "debug", None) else None
        if isinstance(table_match, dict) and table_match.get("reason") == "ambiguous table match":
            alternatives = [
                {"target": item.get("table"), "score": item.get("score"), "match_type": item.get("match_type")}
                for item in table_match.get("matches", [])
            ]
            ambiguity = AmbiguityDetector().detect(
                question,
                {
                    "requires_clarification": True,
                    "ambiguous": True,
                    "mapping_type": "table_mapping",
                    "alternatives": alternatives,
                },
                schema,
            )
            return self._clarification_prediction(question, normalized_question, ambiguity, semantic_profile)

        phrase = self._simple_schema_phrase(normalized_question)
        if not phrase:
            return None
        from semantic_layer.semantic_mapper import SemanticMapper

        mapper = SemanticMapper(semantic_profile)
        mapping_result = mapper.map_column(phrase)
        if mapping_result.get("requires_clarification") and mapping_result.get("alternatives"):
            ambiguity = AmbiguityDetector().detect(question, {**mapping_result, "phrase": phrase}, schema)
            if ambiguity.get("ambiguous"):
                return self._clarification_prediction(question, normalized_question, ambiguity, semantic_profile)
        return None

    def _clarification_prediction(
        self,
        question: str,
        normalized_question: str,
        ambiguity: dict[str, Any],
        semantic_profile: dict[str, Any] | None,
    ) -> PredictionResult:
        clarification = ClarificationQuestionBuilder().build(ambiguity)
        return PredictionResult(
            question=question,
            normalized_question=normalized_question,
            source_model="retrieval_ir",
            intent=None,
            template_id=None,
            sql=None,
            query_ir=None,
            ir_validation=None,
            validation={"is_valid": False, "message": "clarification_required"},
            confidence=0.0,
            confidence_tier="low",
            warnings=["Ambiguous schema mapping; SQL generation is paused until clarification."],
            clarification_questions=[clarification["question"]],
            needs_clarification=True,
            clarification=clarification,
            debug={
                "ambiguity": ambiguity,
                "semantic_profile_summary": self._semantic_summary(semantic_profile),
            },
        )

    @staticmethod
    def _build_semantic_profile(schema: Any) -> dict[str, Any] | None:
        try:
            return build_semantic_profile(schema)
        except Exception:
            return None

    @staticmethod
    def _simple_schema_phrase(normalized_question: str) -> str | None:
        match = re.match(r"^(show|list|display|view|get|fetch|select)\s+(?:all\s+)?(?P<phrase>[a-z0-9_ -]+)$", normalized_question)
        if not match:
            return None
        phrase = match.group("phrase").strip()
        tokens = set(phrase.split())
        analytical_or_temporal = {
            "revenue",
            "sales",
            "amount",
            "total",
            "average",
            "avg",
            "count",
            "last",
            "this",
            "next",
            "month",
            "year",
            "week",
            "day",
            "today",
            "yesterday",
        }
        if not phrase or len(tokens) > 2 or tokens & analytical_or_temporal:
            return None
        return phrase

    @staticmethod
    def _semantic_summary(profile: dict[str, Any] | None) -> dict[str, Any]:
        if not profile:
            return {}
        table_types: dict[str, int] = {}
        sensitive_columns = []
        for table, info in (profile.get("tables") or {}).items():
            table_type = info.get("table_type", "unknown")
            table_types[table_type] = table_types.get(table_type, 0) + 1
            sensitive_columns.extend(f"{table}.{column}" for column in info.get("sensitive_columns", []))
        return {
            "table_count": len(profile.get("tables") or {}),
            "table_types": table_types,
            "metric_count": len(profile.get("metrics") or {}),
            "dimension_count": len(profile.get("dimensions") or {}),
            "date_count": len(profile.get("dates") or {}),
            "sensitive_columns": sensitive_columns,
            "schema_fingerprint": profile.get("schema_fingerprint"),
        }

    def _default_neural_ir_model_dir(self) -> Path:
        return self.neural_ir_v2_model_dir if (self.neural_ir_v2_model_dir / "model.pt").exists() else self.neural_ir_v1_model_dir

    @staticmethod
    def _normalize_question(question: str) -> str:
        return re.sub(r"\s+", " ", question.strip().lower())

    @staticmethod
    def _select_base_table(template_id: str | None, mapping: SchemaMapping) -> str:
        if mapping.base_table:
            return mapping.base_table
        if template_id in {"simple_filter", "show_records"}:
            required = [table for table in [mapping.filter_table, mapping.entity_table, mapping.date_table] if table]
            return RuntimeJoinPlanner.choose_base_table(mapping.filter_table, mapping.entity_table, required)
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
        if template_id in {"simple_filter", "show_records", "count_records"}:
            tables = [mapping.base_table or mapping.filter_table or mapping.entity_table or mapping.metric_table]
        else:
            tables = [mapping.metric_table or mapping.entity_table]
        tables.append(mapping.base_table)
        if template_id in {"simple_filter", "show_records", "count_records"}:
            return [table for table in dict.fromkeys(tables) if table]
        tables.extend(mapping.semantic_required_tables)
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
            mapping.base_table = resolution.get("base_table") or mapping.base_table
            mapping.metric_table = resolution.get("metric_table")
            mapping.metric_column = resolution.get("metric_column")
            mapping.metric_expression = resolution.get("metric_expression")
            mapping.metric_aggregation = resolution.get("metric_aggregation") or mapping.metric_aggregation
            mapping.metric_alias = resolution.get("metric_alias") or mapping.metric_alias
            mapping.match_scores["semantic_metric"] = 1.0
        elif resolution.get("base_table"):
            mapping.base_table = resolution.get("base_table") or mapping.base_table
        if resolution.get("required_tables"):
            mapping.semantic_required_tables = list(resolution.get("required_tables") or [])
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

    def _maybe_neural_ir_fallback(
        self,
        retrieval_ir_result: PredictionResult,
        question: str,
        schema: Any,
        enabled: bool,
    ) -> PredictionResult:
        retrieval_ir_valid = bool(retrieval_ir_result.validation.get("is_valid", retrieval_ir_result.validation.get("ok", False)))
        if not enabled:
            self._attach_router_decision(retrieval_ir_result, 0.0, False, "retrieval_ir", "neural_ir_disabled")
            return retrieval_ir_result
        if retrieval_ir_result.confidence >= self.neural_ir_threshold and retrieval_ir_valid:
            self._attach_router_decision(retrieval_ir_result, 0.0, False, "retrieval_ir", "retrieval_ir_high_confidence")
            return retrieval_ir_result
        neural_ir_model_dir = self._available_neural_ir_model_dir()
        if neural_ir_model_dir is None:
            self._attach_router_decision(retrieval_ir_result, 0.0, False, "retrieval_ir", "neural_ir_missing")
            return retrieval_ir_result

        try:
            from neural_ir.predictor import NeuralIRPredictor

            neural_ir_raw = NeuralIRPredictor(str(neural_ir_model_dir)).predict(question, schema)
        except Exception as exc:
            self._attach_router_decision(retrieval_ir_result, 0.0, False, "retrieval_ir", "neural_ir_error")
            retrieval_ir_result.debug["neural_ir_error"] = str(exc)
            return retrieval_ir_result

        neural_ir_validation = neural_ir_raw.get("sql_validation") or neural_ir_raw.get("validation") or {}
        neural_ir_valid = bool(neural_ir_validation.get("is_valid", neural_ir_validation.get("ok", False)))
        neural_ir_confidence = float(neural_ir_raw.get("confidence") or 0.0)
        retrieval_ir_result.debug["neural_ir_result"] = neural_ir_raw
        decision = choose_route(
            {
                "confidence": retrieval_ir_result.confidence,
                "validation": retrieval_ir_result.validation,
            },
            {
                "confidence": neural_ir_confidence,
                "sql_validation": neural_ir_validation,
                "repairs_applied": neural_ir_raw.get("repairs_applied", []),
                "debug": neural_ir_raw.get("debug", {}),
            },
            self.hybrid_calibration,
        )

        # Normalize decision field names
        selected = _normalize_source(decision.get("selected", "retrieval_ir"))
        decision["selected"] = selected

        if neural_ir_valid and selected == "neural_ir":
            query_ir = neural_ir_raw.get("query_ir") or {}
            ir_validation = neural_ir_raw.get("ir_validation") or {}
            warnings = []
            warnings.extend(ir_validation.get("warnings") or [])
            warnings.extend(ir_validation.get("errors") or [])
            warnings.extend(neural_ir_raw.get("warnings") or [])
            if not neural_ir_validation.get("is_valid", neural_ir_validation.get("ok", False)):
                warnings.extend(neural_ir_validation.get("issues") or [])
            result = PredictionResult(
                question=question,
                normalized_question=self._normalize_question(question),
                source_model="neural_ir",
                intent=query_ir.get("intent"),
                template_id=query_ir.get("template_id"),
                query_ir=query_ir,
                ir_validation=ir_validation,
                sql=neural_ir_raw.get("sql"),
                validation=neural_ir_validation,
                confidence=neural_ir_confidence,
                confidence_tier=self._confidence_tier(neural_ir_confidence),
                warnings=list(dict.fromkeys(str(warning) for warning in warnings if warning)),
                router_decision=decision,
                neural_ir_version=neural_ir_raw.get("neural_ir_version") or neural_ir_raw.get("option_a_version"),
                retrieval_ir_result=retrieval_ir_result.model_dump(),
                neural_ir_result=neural_ir_raw,
                selected_query_ir=query_ir,
                validation_summary={
                    "ir_validation": ir_validation,
                    "sql_validation": neural_ir_validation,
                },
                confidence_breakdown=(neural_ir_raw.get("debug") or {}).get("calibration", {}),
                debug={
                    "retrieval_ir_result": retrieval_ir_result.model_dump(),
                    "neural_ir_result": neural_ir_raw,
                    "router_decision": decision,
                },
            )
            return result

        retrieval_ir_result.router_decision = decision
        retrieval_ir_result.neural_ir_version = neural_ir_raw.get("neural_ir_version") or neural_ir_raw.get("option_a_version")
        retrieval_ir_result.retrieval_ir_result = retrieval_ir_result.model_dump(exclude={"retrieval_ir_result"})
        retrieval_ir_result.neural_ir_result = neural_ir_raw
        retrieval_ir_result.selected_query_ir = retrieval_ir_result.query_ir
        retrieval_ir_result.validation_summary = {
            "ir_validation": retrieval_ir_result.ir_validation,
            "sql_validation": retrieval_ir_result.validation,
        }
        retrieval_ir_result.debug["router_decision"] = decision
        return retrieval_ir_result

    def _available_neural_ir_model_dir(self) -> Path | None:
        if (self.neural_ir_model_dir / "model.pt").exists():
            return self.neural_ir_model_dir
        if self._explicit_neural_ir_model_dir:
            return None
        if (self.neural_ir_v2_model_dir / "model.pt").exists():
            return self.neural_ir_v2_model_dir
        if (self.neural_ir_v1_model_dir / "model.pt").exists():
            return self.neural_ir_v1_model_dir
        return None

    def _attach_router_decision(
        self,
        result: PredictionResult,
        neural_ir_confidence: float,
        neural_ir_valid: bool,
        selected: str,
        reason: str,
    ) -> None:
        decision = {
            "retrieval_ir_confidence": float(result.confidence),
            "neural_ir_confidence": float(neural_ir_confidence),
            "retrieval_ir_valid": bool(result.validation.get("is_valid", result.validation.get("ok", False))),
            "neural_ir_valid": bool(neural_ir_valid),
            "selected": selected,
            "reason": reason,
        }
        result.router_decision = decision
        result.debug["router_decision"] = decision

    @staticmethod
    def _confidence_tier(confidence: float) -> str:
        if confidence >= 0.80:
            return "high"
        if confidence >= 0.60:
            return "medium"
        return "low"


def create_orchestrator_from_bundle(bundle_dir: str | Path) -> PredictionOrchestrator:
    """Create a PredictionOrchestrator from a validated model bundle.

    Loads the bundle manifest, resolves all artifact paths, and returns
    a configured orchestrator. No legacy artifact folder guessing.

    Args:
        bundle_dir: Path to the model bundle directory
            (e.g. ``artifacts/model_bundle/current``).

    Returns:
        Configured PredictionOrchestrator.

    Raises:
        FileNotFoundError: If the bundle manifest is missing.
        ValueError: If the bundle status is 'failed'.
    """
    from model_bundle.bundle_loader import ModelBundleLoader

    loader = ModelBundleLoader()
    bundle = loader.load(bundle_dir)

    neural_dir = Path(bundle["neural_model_dir"])
    neural_model_dir = neural_dir if (neural_dir / "model.pt").exists() else None

    return PredictionOrchestrator(
        neural_ir_model_dir=neural_model_dir,
        use_neural_ir_fallback=neural_model_dir is not None,
    )
