from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class DatasetScaleEvaluator:
    def __init__(self, predictor: Any | None = None):
        self.predictor = predictor

    def evaluate_model(
        self,
        model_name: str,
        examples: list[dict[str, Any]],
        schema_mode: str = "gold",
        max_examples: int | None = None,
        evaluation_mode: str = "real_model_predictions",
        model_artifact_source: str = "none",
        predictor_used: bool | None = None,
        calibration_coverage_target: float = 0.95,
        calibration_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        mode = _normalize_evaluation_mode(evaluation_mode)
        rows = examples[:max_examples] if max_examples is not None else examples
        failures: list[dict[str, Any]] = []
        metrics = Counter()
        totals = Counter()
        by_dataset = defaultdict(Counter)
        by_intent = defaultdict(Counter)
        by_complexity = defaultdict(Counter)
        by_database = defaultdict(Counter)
        label_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
        confidence_outcomes: list[tuple[float, bool]] = []
        distributions: dict[str, list[float]] = defaultdict(list)
        per_example: list[dict[str, Any]] = []
        real_predictions_generated = 0
        prediction_failures = 0

        for row in rows:
            gold = row.get("query_ir") or {}
            pred, prediction_failed = self._predict(row, schema_mode=schema_mode, evaluation_mode=mode)
            if prediction_failed:
                prediction_failures += 1
            elif mode == "real_model_predictions":
                real_predictions_generated += 1
            item_metrics = self._metrics(gold, pred, row)
            pairs = self._label_pairs(gold, pred, row)
            for level, pair in pairs.items():
                label_pairs[level].append(pair)
            for key, value in item_metrics.items():
                totals[key] += 1
                if value:
                    metrics[key] += 1
            for bucket, name in [
                (by_dataset, row.get("dataset_name") or "unknown"),
                (by_intent, gold.get("intent") or row.get("intent") or "unknown"),
                (by_complexity, row.get("complexity") or "unknown"),
                (by_database, row.get("db_id") or "unknown"),
            ]:
                bucket[name]["total"] += 1
                for key, value in item_metrics.items():
                    if value:
                        bucket[name][key] += 1
            if not all(item_metrics.values()):
                failures.append({"example_id": row.get("example_id"), "question": row.get("question"), "metrics": item_metrics})
            correct = bool(item_metrics.get("intent_accuracy") and item_metrics.get("base_table_accuracy") and item_metrics.get("join_accuracy"))
            confidence = _number(row.get("calibrated_confidence", row.get("confidence", pred.get("confidence"))))
            if confidence is not None:
                confidence_outcomes.append((confidence, correct))
                distributions["confidence"].append(confidence)
                distributions["wrong_prediction_confidence" if not correct else "correct_prediction_confidence"].append(confidence)
            self._collect_distributions(row, distributions)
            per_example.append({
                "example_id": row.get("example_id"),
                "intent_correct": item_metrics.get("intent_accuracy", False),
                "base_table_correct": item_metrics.get("base_table_accuracy", False),
                "join_correct": item_metrics.get("join_accuracy", False),
                "final_correct": correct,
                "confidence": confidence,
                "sql_valid": item_metrics.get("sql_validation", False),
                "execution_match": bool(row.get("execution_match", False)),
                "unnecessary_join": bool((pred.get("joins") or []) and not (gold.get("joins") or [])),
                "wrong_table": gold.get("base_table") != pred.get("base_table"),
                # Bootstrap promotion fields (required by promotion_policy.py)
                "simple_query_pass": bool(item_metrics.get("simple_query_match", False)),
                "gold_comparison_score": float(
                    row.get("gold_comparison_score")
                    or item_metrics.get("gold_comparison_score")
                    or (1.0 if correct else 0.0)
                ),
                "unseen_db_sql_valid": (
                    bool(item_metrics.get("sql_validation", False))
                    if schema_mode == "unseen_db"
                    else None
                ),
            })

        summary = {f"{key}_rate": metrics[key] / totals[key] if totals[key] else 0.0 for key in totals}
        summary["total_examples"] = len(rows)
        summary["unnecessary_join_rate"] = 1.0 - summary.get("no_unnecessary_join_rate", 1.0)
        summary["wrong_table_rate"] = 1.0 - summary.get("base_table_accuracy_rate", 1.0)
        classification = {
            level: classification_metrics(pairs)
            for level, pairs in sorted(label_pairs.items())
        }
        calibration = calibration_metrics(
            confidence_outcomes,
            coverage_target=calibration_coverage_target,
            config=calibration_config,
        )
        percentiles = percentile_report(distributions)
        summary.update({
            "intent_macro_f1": classification.get("intent", {}).get("macro_f1", 0.0),
            "base_table_macro_f1": classification.get("base_table", {}).get("macro_f1", 0.0),
            "join_decision_macro_f1": classification.get("join_decision", {}).get("macro_f1", 0.0),
            "router_accuracy": classification.get("router", {}).get("accuracy", 0.0),
            "router_macro_f1": classification.get("router", {}).get("macro_f1", 0.0),
            "unsafe_sql_count": sum(1 for row in rows if not _is_select_safe(row)),
            "execution_match_rate": _optional_rate(rows, "execution_match"),
        })
        gold_replay_used = mode in {"explicit_gold_replay_baseline", "explicit_oracle_upper_bound"}
        inferred_predictor_used = self.predictor is not None or (
            mode == "real_model_predictions" and bool(rows) and real_predictions_generated > 0
        )
        predictor_used = inferred_predictor_used if predictor_used is None else bool(predictor_used)
        is_valid_for_quality_gate = (
            mode == "real_model_predictions"
            and not gold_replay_used
            and predictor_used is True
            and real_predictions_generated > 0
            and len(rows) > 0
            and real_predictions_generated + prediction_failures == len(rows)
        )
        return {
            "model_name": model_name,
            "schema_mode": schema_mode,
            "evaluation_mode": mode,
            "test_source": "real_model_predictions" if mode == "real_model_predictions" else mode.replace("explicit_", ""),
            "gold_replay_used": gold_replay_used,
            "gold_replay_baseline": mode == "explicit_gold_replay_baseline",
            "predictor_used": predictor_used,
            "model_artifact_source": model_artifact_source,
            "is_valid_for_quality_gate": is_valid_for_quality_gate,
            "rows_evaluated": len(rows),
            "real_predictions_generated": real_predictions_generated,
            "prediction_failures": prediction_failures,
            "summary": summary,
            "by_dataset": self._bucket_rates(by_dataset),
            "by_intent": self._bucket_rates(by_intent),
            "by_complexity": self._bucket_rates(by_complexity),
            "by_database": self._bucket_rates(by_database),
            "classification_metrics": classification,
            "confusion_matrices": {level: metrics["confusion_matrix"] for level, metrics in classification.items()},
            "percentiles": percentiles,
            "calibration": calibration,
            "per_example": per_example,
            "failure_examples": failures[:50],
        }

    def _predict(self, row: dict[str, Any], schema_mode: str, evaluation_mode: str) -> tuple[dict[str, Any], bool]:
        if evaluation_mode in {"explicit_gold_replay_baseline", "explicit_oracle_upper_bound"}:
            return row.get("query_ir") or {}, False
        if self.predictor is not None:
            try:
                prediction = self.predictor(row, schema_mode=schema_mode)
                return prediction or {}, False
            except Exception as exc:
                row["prediction_error"] = str(exc)
                return {}, True
        if "predicted_query_ir" in row and row.get("predicted_query_ir") is not None:
            return row.get("predicted_query_ir") or {}, bool(row.get("prediction_failed", False))
        raise ValueError(
            "DatasetScaleEvaluator requires real predicted_query_ir rows or a predictor. "
            "Use evaluation_mode='explicit_gold_replay_baseline' only for debug baselines."
        )

    @staticmethod
    def _metrics(gold: dict[str, Any], pred: dict[str, Any], row: dict[str, Any]) -> dict[str, bool]:
        gold_joins = gold.get("joins") or []
        pred_joins = pred.get("joins") or []
        return {
            "intent_accuracy": gold.get("intent") == pred.get("intent"),
            "template_accuracy": gold.get("template_id") == pred.get("template_id"),
            "base_table_accuracy": gold.get("base_table") == pred.get("base_table"),
            "metric_accuracy": _projection(gold, "metrics", ["aggregation", "expression"]) == _projection(pred, "metrics", ["aggregation", "expression"]),
            "dimension_accuracy": _projection(gold, "dimensions", ["expression"]) == _projection(pred, "dimensions", ["expression"]),
            "filter_accuracy": _projection(gold, "filters", ["expression", "operator", "value"]) == _projection(pred, "filters", ["expression", "operator", "value"]),
            "date_filter_accuracy": _projection(gold, "date_filters", ["date_expression", "filter_type", "start_date", "end_date", "date_grain"]) == _projection(pred, "date_filters", ["date_expression", "filter_type", "start_date", "end_date", "date_grain"]),
            "join_accuracy": _projection(gold, "joins", ["condition"]) == _projection(pred, "joins", ["condition"]),
            "no_unnecessary_join": not pred_joins if not gold_joins else True,
            "query_ir_validity": bool(row.get("ir_validation", {}).get("is_valid", True)),
            "sql_validation": bool(row.get("sql_validation", {}).get("is_valid", row.get("sql_validation", {}).get("ok", True))),
            "structural_sql_match": _structural_sql_match(row),
        }

    @staticmethod
    def _bucket_rates(buckets: dict[str, Counter]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for name, counter in buckets.items():
            total = counter.get("total", 0)
            result[name] = {
                f"{key}_rate": value / total
                for key, value in counter.items()
                if key != "total" and total
            }
            result[name]["total_examples"] = total
        return result

    @staticmethod
    def _label_pairs(gold: dict[str, Any], pred: dict[str, Any], row: dict[str, Any]) -> dict[str, tuple[str, str]]:
        pairs = {
            "intent": (_label(gold.get("intent")), _label(pred.get("intent"))),
            "base_table": (_label(gold.get("base_table")), _label(pred.get("base_table"))),
            "join_decision": _join_labels(gold, pred),
            "router": (_gold_route(gold, row), _predicted_route(row, pred)),
            "error_type": (_label(row.get("gold_error_type")), _predicted_error_type(gold, pred, row)),
        }
        sections = {
            "metric_column": ("metrics", ["expression", "aggregation"]),
            "dimension_column": ("dimensions", ["expression"]),
            "filter_column": ("filters", ["expression"]),
            "date_column": ("date_filters", ["date_expression"]),
            "order_by_column": ("order_by", ["expression"]),
            "join_column": ("joins", ["condition"]),
        }
        for name, (section, keys) in sections.items():
            pairs[name] = (_section_label(gold, section, keys), _section_label(pred, section, keys))
        return pairs

    @staticmethod
    def _collect_distributions(row: dict[str, Any], distributions: dict[str, list[float]]) -> None:
        aliases = {
            "prediction_latency_ms": ["prediction_latency_ms", "latency_ms"],
            "retrieval_latency_ms": ["retrieval_latency_ms"],
            "neural_inference_latency_ms": ["neural_inference_latency_ms", "inference_latency_ms"],
            "sql_validation_latency_ms": ["sql_validation_latency_ms"],
            "sql_execution_latency_ms": ["sql_execution_latency_ms", "execution_latency_ms"],
            "train_loss": ["train_loss", "loss"],
            "validation_loss": ["validation_loss", "val_loss"],
            "question_token_count": ["question_token_count"],
            "schema_table_count": ["schema_table_count"],
            "schema_column_count": ["schema_column_count"],
            "candidate_column_count": ["candidate_column_count"],
            "schema_token_length": ["schema_token_length"],
        }
        for output, keys in aliases.items():
            for key in keys:
                value = _number(row.get(key))
                if value is not None:
                    distributions[output].append(value)
                    break
        scores = row.get("retrieval_scores") or []
        if len(scores) >= 2:
            first, second = _number(scores[0]), _number(scores[1])
            if first is not None and second is not None:
                distributions["retrieval_margin"].append(first - second)


def classification_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    if not pairs:
        return _empty_classification_metrics()
    labels = sorted({label for pair in pairs for label in pair})
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for gold, pred in pairs:
        matrix[gold][pred] += 1
    per_label: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[gold][label] for gold in labels if gold != label)
        fn = sum(matrix[label][pred] for pred in labels if pred != label)
        support = sum(matrix[label].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    total = len(pairs)
    correct = sum(matrix[label][label] for label in labels)
    macro_precision = sum(float(item["precision"]) for item in per_label.values()) / len(labels)
    macro_recall = sum(float(item["recall"]) for item in per_label.values()) / len(labels)
    macro_f1 = sum(float(item["f1"]) for item in per_label.values()) / len(labels)
    weighted_f1 = sum(float(item["f1"]) * int(item["support"]) for item in per_label.values()) / total
    accuracy = correct / total
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_f1": accuracy,
        "weighted_f1": weighted_f1,
        "labels": labels,
        "per_label": per_label,
        "confusion_matrix": matrix,
    }


def percentile_report(distributions: dict[str, list[float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    percentile_sets = {
        "confidence": (5, 50, 95, 99),
        "wrong_prediction_confidence": (95, 99),
        "correct_prediction_confidence": (5,),
        "retrieval_margin": (5, 50, 95),
    }
    for name, values in distributions.items():
        requested = percentile_sets.get(name, (50, 95, 99))
        for percentile in requested:
            result[f"{name}_p{percentile:02d}"] = _percentile(values, percentile)
    return result


def calibration_metrics(
    outcomes: list[tuple[float, bool]],
    buckets: int = 10,
    coverage_target: float = 0.95,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reliability = []
    cfg = config or {}
    target = max(0.0, min(1.0, float(cfg.get("abstention_coverage_target", coverage_target))))
    use_conformal = bool(cfg.get("use_conformal_threshold", True))
    if not outcomes:
        return {
            "sample_count": 0,
            "expected_calibration_error": 0.0,
            "brier_score": 0.0,
            "buckets": reliability,
            "coverage_target": target,
            "abstention_coverage_target": target,
            "use_conformal_threshold": use_conformal,
            "conformal_confidence_threshold": None,
            "isotonic_points": [],
        }
    brier = sum((confidence - float(correct)) ** 2 for confidence, correct in outcomes) / len(outcomes)
    ece = 0.0
    for index in range(buckets):
        lower, upper = index / buckets, (index + 1) / buckets
        selected = [(c, ok) for c, ok in outcomes if lower <= c <= upper and (index == buckets - 1 or c < upper)]
        if selected:
            avg_confidence = sum(item[0] for item in selected) / len(selected)
            avg_correctness = sum(float(item[1]) for item in selected) / len(selected)
        else:
            avg_confidence = avg_correctness = 0.0
        ece += len(selected) / len(outcomes) * abs(avg_confidence - avg_correctness)
        reliability.append({
            "range": f"{lower:.1f}-{upper:.1f}",
            "count": len(selected),
            "avg_confidence": avg_confidence,
            "avg_correctness": avg_correctness,
        })
    nonconformity = [1.0 - confidence for confidence, _ in outcomes]
    conformal_score = _percentile(nonconformity, int(round(target * 100)))
    configured_threshold = cfg.get("abstain_when_calibrated_confidence_below")
    if isinstance(configured_threshold, (int, float)):
        conformal_confidence_threshold = float(configured_threshold)
    elif use_conformal:
        conformal_confidence_threshold = max(0.0, min(1.0, 1.0 - conformal_score))
    else:
        conformal_confidence_threshold = None
    points = []
    monotonic = 0.0
    for bucket in reliability:
        if bucket["count"]:
            monotonic = max(monotonic, float(bucket["avg_correctness"]))
            points.append([float(bucket["avg_confidence"]), monotonic])
    return {
        "sample_count": len(outcomes),
        "expected_calibration_error": ece,
        "brier_score": brier,
        "buckets": reliability,
        "coverage_target": target,
        "abstention_coverage_target": target,
        "use_conformal_threshold": use_conformal,
        "conformal_nonconformity_threshold": conformal_score,
        "conformal_confidence_threshold": conformal_confidence_threshold,
        "isotonic_points": points,
    }


def write_confusion_matrix_csv(path: str | Path, matrix: dict[str, dict[str, int]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted(set(matrix) | {label for row in matrix.values() for label in row})
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold\\predicted", *labels])
        for gold in labels:
            writer.writerow([gold, *[matrix.get(gold, {}).get(pred, 0) for pred in labels]])


def _empty_classification_metrics() -> dict[str, Any]:
    return {"accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "weighted_f1": 0.0, "labels": [], "per_label": {}, "confusion_matrix": {}}


def _label(value: Any) -> str:
    return str(value) if value not in {None, ""} else "__none__"


def _section_label(ir: dict[str, Any], section: str, keys: list[str]) -> str:
    values = []
    for item in ir.get(section) or []:
        if isinstance(item, dict):
            values.append("|".join(str(item.get(key) or "") for key in keys))
    return ";".join(sorted(values)) if values else "__none__"


def _join_labels(gold: dict[str, Any], pred: dict[str, Any]) -> tuple[str, str]:
    gold_joins, pred_joins = gold.get("joins") or [], pred.get("joins") or []
    expected = "join_required" if gold_joins else "no_join_required"
    if not gold_joins and pred_joins:
        actual = "unnecessary_join"
    elif gold_joins and not pred_joins:
        actual = "missing_join"
    elif gold_joins and _projection(gold, "joins", ["condition"]) != _projection(pred, "joins", ["condition"]):
        actual = "wrong_join_path"
    else:
        actual = expected
    return expected, actual


def _gold_route(gold: dict[str, Any], row: dict[str, Any]) -> str:
    explicit = row.get("gold_route")
    if explicit:
        return _normalize_route(explicit)
    intent = gold.get("intent")
    if intent in {"show_records", "count_records", "simple_filter"} and not (gold.get("joins") or []):
        return "generic_direct_planner"
    if intent == "needs_clarification":
        return "clarification"
    if intent == "unsupported":
        return "unsupported"
    return "adaptive_router"


def _predicted_route(row: dict[str, Any], pred: dict[str, Any]) -> str:
    value = row.get("predicted_route") or row.get("route") or row.get("prediction_source") or pred.get("source")
    return _normalize_route(value or _gold_route(pred, row))


def _normalize_route(value: Any) -> str:
    normalized = str(value).lower().replace("-", "_").replace(" ", "_")
    aliases = {"retrieval": "retrieval_queryir", "neural": "neural_queryir", "generic_direct": "generic_direct_planner", "gold_replay_baseline": "adaptive_router"}
    return aliases.get(normalized, normalized)


def _predicted_error_type(gold: dict[str, Any], pred: dict[str, Any], row: dict[str, Any]) -> str:
    if row.get("predicted_error_type"):
        return _label(row.get("predicted_error_type"))
    if gold.get("base_table") != pred.get("base_table"):
        return "wrong_table"
    expected, actual = _join_labels(gold, pred)
    if expected != actual:
        return actual
    if gold.get("filters") != pred.get("filters"):
        return "wrong_filter"
    return "__none__"


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def _optional_rate(rows: list[dict[str, Any]], key: str) -> float:
    values = [bool(row[key]) for row in rows if key in row]
    return sum(values) / len(values) if values else 0.0


def _is_select_safe(row: dict[str, Any]) -> bool:
    candidate = row.get("predicted_sql") if "predicted_sql" in row else row.get("rendered_sql") or row.get("source_sql")
    sql = str(candidate or "").strip().lower()
    if not sql:
        return True
    validation = row.get("sql_validation") or {}
    if validation.get("is_safe") is False or validation.get("is_valid") is False or validation.get("ok") is False:
        return False
    return sql.startswith("select") or sql.startswith("with")


def _structural_sql_match(row: dict[str, Any]) -> bool:
    gold = row.get("source_sql") or row.get("gold_sql")
    predicted = row.get("predicted_sql") or row.get("rendered_sql")
    if not gold or not predicted:
        return False
    try:
        from execution_eval.sql_structure_comparator import SQLStructureComparator

        result = SQLStructureComparator().compare(
            str(predicted),
            str(gold),
            schema=row.get("schema"),
            dialect=str((row.get("schema") or {}).get("dialect") or "sqlite"),
        )
        return float(result.get("structure_score", 0.0)) >= 0.999
    except Exception:
        return normalize_sql(str(gold)) == normalize_sql(str(predicted))


def _projection(ir: dict[str, Any], section: str, keys: list[str]) -> list[tuple[Any, ...]]:
    return sorted(tuple(item.get(key) for key in keys) for item in ir.get(section) or [])


def normalize_sql(value: str | None) -> str:
    return " ".join(str(value or "").lower().split())


def _normalize_evaluation_mode(value: str) -> str:
    normalized = str(value or "real_model_predictions").strip().lower()
    aliases = {
        "real": "real_model_predictions",
        "model": "real_model_predictions",
        "gold_replay": "explicit_gold_replay_baseline",
        "gold_replay_baseline": "explicit_gold_replay_baseline",
        "oracle": "explicit_oracle_upper_bound",
        "oracle_upper_bound": "explicit_oracle_upper_bound",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"real_model_predictions", "explicit_gold_replay_baseline", "explicit_oracle_upper_bound"}
    if normalized not in allowed:
        raise ValueError(f"Unsupported evaluation_mode {value!r}; expected one of {sorted(allowed)}")
    return normalized
