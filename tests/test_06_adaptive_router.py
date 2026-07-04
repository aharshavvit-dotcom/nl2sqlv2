"""Test 06: Adaptive Router — choose_route, calibration, confidence caps."""

from __future__ import annotations

from neural_ir.calibration import (
    AdaptiveRouterCalibrator,
    HybridRouterCalibrator,
    choose_route,
    DEFAULT_CALIBRATION,
)
from inference.prediction_models import PredictionResult
from inference.prediction_orchestrator import PredictionOrchestrator
from inference.runtime_schema_context import RuntimeSchemaContext
from inference.schema_aware_mapper import SchemaAwareMapper
from inference.slot_resolver import SlotResolver


class TestChooseRoute:
    def test_high_confidence_retrieval_ir_wins(self) -> None:
        decision = choose_route(
            {"confidence": 0.9, "validation": {"is_valid": True}},
            {"confidence": 0.5, "sql_validation": {"is_valid": True}},
        )
        assert decision["selected"] == "retrieval_ir"
        assert decision["reason"] == "retrieval_ir_high_confidence"

    def test_invalid_retrieval_ir_uses_neural_ir(self) -> None:
        decision = choose_route(
            {"confidence": 0.3, "validation": {"is_valid": False}},
            {"confidence": 0.6, "sql_validation": {"is_valid": True}},
        )
        assert decision["selected"] == "neural_ir"
        assert decision["reason"] == "retrieval_ir_invalid_sql"

    def test_higher_neural_ir_selected(self) -> None:
        decision = choose_route(
            {"confidence": 0.3, "validation": {"is_valid": True}},
            {"confidence": 0.7, "sql_validation": {"is_valid": True}},
        )
        assert decision["selected"] == "neural_ir"

    def test_invalid_neural_ir_keeps_retrieval(self) -> None:
        decision = choose_route(
            {"confidence": 0.3, "validation": {"is_valid": True}},
            {"confidence": 0.7, "sql_validation": {"is_valid": False}},
        )
        assert decision["selected"] == "retrieval_ir"

    def test_output_uses_new_field_names(self) -> None:
        decision = choose_route(
            {"confidence": 0.5, "validation": {"is_valid": True}},
            {"confidence": 0.5, "sql_validation": {"is_valid": True}},
        )
        assert "retrieval_ir_confidence" in decision
        assert "neural_ir_confidence" in decision
        assert "retrieval_ir_valid" in decision
        assert "neural_ir_valid" in decision
        # Old names should NOT be in the output
        assert "option_c_confidence" not in decision
        assert "option_a_confidence" not in decision


class TestAdaptiveRouterCalibrator:
    def test_backward_alias(self) -> None:
        assert HybridRouterCalibrator is AdaptiveRouterCalibrator

    def test_calibrate_empty_results(self) -> None:
        calibrator = AdaptiveRouterCalibrator()
        result = calibrator.calibrate([], [])
        assert "router_accuracy" in result

    def test_calibrate_with_results(self) -> None:
        retrieval_results = [
            {"confidence": 0.9, "validation": {"is_valid": True}},
            {"confidence": 0.3, "validation": {"is_valid": True}},
        ]
        neural_results = [
            {"confidence": 0.5, "sql_validation": {"is_valid": True}},
            {"confidence": 0.7, "sql_validation": {"is_valid": True}},
        ]
        calibrator = AdaptiveRouterCalibrator()
        result = calibrator.calibrate(retrieval_results, neural_results)
        assert 0.0 <= result["router_accuracy"] <= 1.0
        assert len(result["cases"]) == 2


class TestRouterInOrchestrator:
    def test_missing_neural_ir_returns_retrieval_ir(self, tmp_path) -> None:
        result = PredictionOrchestrator(neural_ir_model_dir=tmp_path)._maybe_neural_ir_fallback(
            retrieval_ir_result=_make_result(confidence=0.2, valid=True),
            question="How many orders?",
            schema={},
            enabled=True,
        )
        assert result.source_model == "retrieval_ir"
        assert result.router_decision["reason"] == "neural_ir_missing"

    def test_high_confidence_retrieval_ir_skips_neural(self, tmp_path) -> None:
        result = PredictionOrchestrator(neural_ir_model_dir=tmp_path)._maybe_neural_ir_fallback(
            retrieval_ir_result=_make_result(confidence=0.9, valid=True),
            question="Top customers",
            schema={},
            enabled=True,
        )
        assert result.source_model == "retrieval_ir"
        assert result.router_decision["reason"] == "retrieval_ir_high_confidence"

    def test_disabled_neural_ir_returns_retrieval_ir(self, tmp_path) -> None:
        result = PredictionOrchestrator(neural_ir_model_dir=tmp_path)._maybe_neural_ir_fallback(
            retrieval_ir_result=_make_result(confidence=0.2, valid=True),
            question="How many orders?",
            schema={},
            enabled=False,
        )
        assert result.source_model == "retrieval_ir"
        assert result.router_decision["reason"] == "neural_ir_disabled"


class TestConfidenceCaps:
    def test_ir_invalid_caps_confidence(self) -> None:
        from inference.prediction_confidence import PredictionConfidenceCalculator
        calc = PredictionConfidenceCalculator()
        result = calc.calculate({
            "candidates": [], "selected_template": {}, "slots": {},
            "schema_mapping": {}, "join_plan": {},
            "ir_validation": {"is_valid": False},
            "validation": {"is_valid": True},
            "warnings": [],
        })
        assert result["confidence"] <= 0.59

    def test_unsafe_sql_forces_abstention(self) -> None:
        reason = PredictionOrchestrator._forced_abstention_reason(
            True,
            {"is_valid": False, "checks": {"parse": True, "select_only": False, "no_blocked_keywords": False}},
            None,
            {},
            "show_records",
        )
        assert reason == "unsafe_sql"

    def test_confidence_components_vary_and_constant_calibration_is_degenerate(self) -> None:
        from dataset_training.dataset_evaluator import calibration_metrics
        from inference.prediction_confidence import PredictionConfidenceCalculator

        calculator = PredictionConfidenceCalculator()
        high = calculator.calculate({
            "candidates": [],
            "selected_template": {"confidence": 0.95},
            "slots": {"metric": {"value": "amount", "confidence": 0.95}},
            "schema_mapping": {"match_scores": {"metric": 0.95}},
            "join_plan": {"confidence": 1.0},
            "ir_validation": {"is_valid": True},
            "validation": {"is_valid": True},
        })
        low = calculator.calculate({
            "candidates": [],
            "selected_template": {"confidence": 0.2},
            "slots": {},
            "schema_mapping": {"match_scores": {}, "filter_ambiguous": True},
            "join_plan": {"confidence": 0.3},
            "ir_validation": {"is_valid": False},
            "validation": {"is_valid": False},
        })
        report = calibration_metrics([(0.8421, True), (0.8421, False), (0.8421, True)])

        assert high["raw_confidence"] > low["raw_confidence"]
        assert "filter_linking_confidence" in high["confidence_components"]
        assert report["calibration_degenerate"] is True
        assert report["confidence_threshold_usable"] is False
        assert report["conformal_confidence_threshold"] is None

    def test_ambiguous_filter_forces_clarification(self) -> None:
        from inference.prediction_models import SchemaMapping
        mapping = SchemaMapping(
            filter_column="name",
            filter_ambiguous=True,
            filter_alternatives=["players.player_name", "coaches.name"],
            match_scores={"filter": 0.49},
        )
        reason = PredictionOrchestrator._forced_abstention_reason(
            True,
            {"is_valid": True},
            mapping,
            {"filter_value": {"value": "Alex"}},
            "simple_filter",
        )
        assert reason == "ambiguous_filter_column"

    def test_sql_repair_failure_forces_abstention(self) -> None:
        reason = PredictionOrchestrator._forced_abstention_reason(
            True,
            {"is_valid": False, "checks": {"parse": False, "select_only": False, "no_blocked_keywords": True}},
            None,
            {},
            "show_records",
        )
        assert reason == "sql_validation_failed"


class TestFilterDimensionLinking:
    @staticmethod
    def _schema() -> dict:
        return {"tables": {"players": {"columns": {
            "player_name": {"type": "text", "sample_values": ["Bubba Starling", "Mark Sanford"]},
            "hometown": {"type": "text"},
            "school_club_team": {"type": "text"},
            "season": {"type": "integer", "sample_values": [2012]},
            "acquisition_method": {"type": "text", "sample_values": ["trade"]},
            "aircraft_model": {"type": "text", "sample_values": ["Robinson R-22"]},
            "gross_weight": {"type": "numeric"},
        }}}}

    def _resolve(self, question: str, template: str = "metric_by_dimension"):
        context = RuntimeSchemaContext(self._schema())
        payload = SlotResolver().resolve_slots(
            question,
            {"template_id": template, "intent": template},
            [],
            context,
        )
        mapping = SchemaAwareMapper().map_slots_to_schema(
            payload["slots"], context, template_id=template,
        )
        return payload["slots"], mapping

    def test_hometown_and_person_name_link_separately(self) -> None:
        slots, mapping = self._resolve("What is the hometown of Bubba Starling?")
        assert slots["filter_value"]["value"] == "Bubba Starling"
        assert mapping.filter_column == "player_name"
        assert mapping.dimension_column == "hometown"
        assert mapping.filter_linking_method == "value_lookup"

    def test_school_team_and_named_player_link_separately(self) -> None:
        slots, mapping = self._resolve("Which school club team has a player named Mark Sanford?")
        assert slots["filter_value"]["value"] == "Mark Sanford"
        assert mapping.filter_column == "player_name"
        assert mapping.dimension_column == "school_club_team"

    def test_season_value_is_not_confused_with_output_dimension(self) -> None:
        slots, mapping = self._resolve(
            "What was the school club team whose season was in 2012 and were acquired via trade?"
        )
        assert str(slots["filter_value"]["value"]) == "2012"
        assert mapping.filter_column == "season"
        assert mapping.dimension_column == "school_club_team"

    def test_aircraft_model_filters_separately_from_weight_metric(self) -> None:
        slots, mapping = self._resolve(
            "What is the max gross weight of the Robinson R-22?",
            template="metric_summary",
        )
        assert slots["filter_value"]["value"] == "Robinson R-22"
        assert mapping.filter_column == "aircraft_model"
        assert mapping.metric_column == "gross_weight"

    def test_filter_value_diagnostics_include_ranked_value_lookup(self) -> None:
        context = RuntimeSchemaContext(self._schema())
        payload = SlotResolver().resolve_slots(
            "What is the hometown of Bubba Starling?",
            {"template_id": "metric_by_dimension", "intent": "metric_by_dimension"},
            [],
            context,
        )

        candidate = payload["filter_value_candidates"][0]
        assert candidate["value"] == "Bubba Starling"
        assert candidate["span"]
        assert candidate["candidate_columns"][0]["column"] == "players.player_name"
        assert "value_lookup" in candidate["candidate_columns"][0]["signals"]

    def test_ambiguous_value_lookup_caps_filter_confidence(self) -> None:
        schema = {"tables": {"people": {"columns": {
            "player_name": {"type": "text", "sample_values": ["Alex Smith"]},
            "coach_name": {"type": "text", "sample_values": ["Alex Smith"]},
            "hometown": {"type": "text"},
        }}}}
        context = RuntimeSchemaContext(schema)
        payload = SlotResolver().resolve_slots(
            "What is the hometown of Alex Smith?",
            {"template_id": "metric_by_dimension", "intent": "metric_by_dimension"},
            [],
            context,
        )

        assert payload["slots"]["filter_column"]["confidence"] < 0.5
        assert payload["slots"]["filter_column"]["alternatives"]

    def test_sql_invalid_caps_confidence(self) -> None:
        from inference.prediction_confidence import PredictionConfidenceCalculator
        calc = PredictionConfidenceCalculator()
        result = calc.calculate({
            "candidates": [], "selected_template": {}, "slots": {},
            "schema_mapping": {}, "join_plan": {},
            "ir_validation": {"is_valid": True},
            "validation": {"is_valid": False},
            "warnings": [],
        })
        assert result["confidence"] <= 0.59


def _make_result(confidence: float, valid: bool) -> PredictionResult:
    return PredictionResult(
        question="q", normalized_question="q", source_model="retrieval_ir",
        intent="show_records", template_id="show_records",
        sql="SELECT order_id FROM orders LIMIT 100" if valid else None,
        validation={"is_valid": valid, "ok": valid, "issues": [] if valid else ["bad"]},
        confidence=confidence, confidence_tier="high" if confidence >= 0.8 else "low",
        debug={},
    )
