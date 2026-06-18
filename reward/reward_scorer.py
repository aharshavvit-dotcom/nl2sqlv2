from __future__ import annotations

from typing import Any

from validation.sql_validator import SQLValidator

from .reward_features import (
    asks_for_date_filter,
    asks_for_dimension,
    asks_for_filter,
    asks_for_join,
    asks_for_metric,
    has_select_star,
    has_unsafe_sql,
    requested_base_table,
    unnecessary_join,
)


class RewardScorer:
    def __init__(self, max_limit: int = 1000):
        self.max_limit = max_limit
        self.sql_validator = SQLValidator()

    def score(
        self,
        candidate: dict[str, Any],
        question: str,
        schema: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = context or {}
        query_ir = candidate.get("query_ir") or candidate.get("selected_query_ir") or {}
        sql = candidate.get("sql") or candidate.get("generated_sql") or candidate.get("rendered_sql")
        validation = candidate.get("sql_validation") or candidate.get("validation") or {}
        ir_validation = candidate.get("ir_validation") or {}
        features: dict[str, Any] = {}
        penalties: list[str] = []
        bonuses: list[str] = []
        score = 0.50

        features["ir_validation_passed"] = bool(ir_validation.get("is_valid", True))
        if features["ir_validation_passed"]:
            score += 0.08
        else:
            score -= 0.20
            penalties.append("ir_validation_failed")

        if sql:
            calculated_validation = self.sql_validator.validate(sql, schema=schema, max_limit=self.max_limit, dialect=schema.get("dialect", "sqlite") if isinstance(schema, dict) else "sqlite")
            validation = {**calculated_validation, **validation}
        features["sql_validation_passed"] = bool(validation.get("is_valid", validation.get("ok", True)))
        if features["sql_validation_passed"]:
            score += 0.10
        else:
            score -= 0.30
            penalties.append("sql_validation_failed")

        features["no_unsafe_sql"] = not has_unsafe_sql(sql)
        if not features["no_unsafe_sql"]:
            return {
                "reward_score": 0.0,
                "features": features,
                "penalties": [*penalties, "unsafe_sql_rejected"],
                "bonuses": bonuses,
            }

        features["no_select_star"] = not has_select_star(sql, query_ir)
        if not features["no_select_star"]:
            score -= 0.25
            penalties.append("select_star")

        expected_table = requested_base_table(question, schema)
        features["expected_base_table"] = expected_table
        features["correct_base_table_match"] = expected_table is None or query_ir.get("base_table") == expected_table
        if features["correct_base_table_match"]:
            score += 0.08
        else:
            score -= 0.25
            penalties.append("wrong_base_table")

        features["no_unnecessary_joins"] = not unnecessary_join(question, query_ir)
        if features["no_unnecessary_joins"]:
            score += 0.08
        else:
            score -= 0.35
            penalties.append("unnecessary_join_for_direct_query")

        join_policy = str((query_ir.get("metadata") or {}).get("join_policy") or candidate.get("join_policy") or "")
        features["join_policy_respected"] = join_policy != "none" or not query_ir.get("joins")
        if features["join_policy_respected"]:
            score += 0.04
        else:
            score -= 0.30
            penalties.append("join_policy_violated")

        features["filter_requested_and_present"] = (not asks_for_filter(question)) or bool(query_ir.get("filters") or query_ir.get("date_filters"))
        features["date_filter_requested_and_present"] = (not asks_for_date_filter(question)) or bool(query_ir.get("date_filters") or query_ir.get("filters"))
        features["metric_requested_and_present"] = (not asks_for_metric(question)) or bool(query_ir.get("metrics"))
        features["dimension_requested_and_present"] = (not asks_for_dimension(question)) or bool(query_ir.get("dimensions"))
        for feature_name in [
            "filter_requested_and_present",
            "date_filter_requested_and_present",
            "metric_requested_and_present",
            "dimension_requested_and_present",
        ]:
            if features[feature_name]:
                score += 0.03
            else:
                score -= 0.10
                penalties.append(feature_name.replace("_requested_and_present", "_missing"))

        execution_status = candidate.get("execution_status") or context.get("execution_status") or {}
        features["sql_execution_success"] = bool(execution_status.get("success", execution_status.get("ok", False))) if execution_status else None
        if features["sql_execution_success"] is True:
            score += 0.08
            bonuses.append("execution_success")

        feedback_match_boost = float(candidate.get("feedback_match_boost", 0.0) or context.get("feedback_match_boost", 0.0) or 0.0)
        if candidate.get("source") == "feedback_index":
            feedback_match_boost = max(feedback_match_boost, 0.15)
        hard_negative_similarity = float(candidate.get("hard_negative_similarity", 0.0) or context.get("hard_negative_similarity", 0.0) or 0.0)
        features["feedback_match_boost"] = feedback_match_boost
        features["hard_negative_similarity"] = hard_negative_similarity
        if feedback_match_boost:
            score += min(0.20, feedback_match_boost)
            bonuses.append("feedback_corrected_match")
        if hard_negative_similarity:
            score -= min(0.30, hard_negative_similarity)
            penalties.append("hard_negative_similarity")

        if asks_for_join(question):
            features["join_requested"] = True
        return {
            "reward_score": max(0.0, min(1.0, round(score, 6))),
            "features": features,
            "penalties": penalties,
            "bonuses": bonuses,
        }
