from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .option_a_to_ir import OptionAToIRConverter
from .schema_linearizer import schema_from_example
from .trainer import HEAD_TO_LABEL, MODEL_INPUT_KEYS
from validation.sql_validator import SQLValidator
from ir.ir_to_sql_renderer import IRToSQLRenderer
from ir.ir_validator import IRValidator


class OptionAIREvaluator:
    def __init__(self) -> None:
        self.converter = OptionAToIRConverter()
        self.ir_validator = IRValidator()
        self.sql_renderer = IRToSQLRenderer()
        self.sql_validator = SQLValidator()

    def evaluate(self, model, data_loader, label_encoder, db_path: str | None = None) -> dict[str, Any]:
        model.eval()
        counts = defaultdict(int)
        correct = defaultdict(int)
        query_ir_valid = 0
        sql_valid = 0
        failures = []
        by_intent = defaultdict(lambda: {"total": 0, "intent_correct": 0, "sql_valid": 0})
        total = 0
        with torch.no_grad():
            for batch in data_loader:
                outputs = model(**{key: batch[key] for key in MODEL_INPUT_KEYS if key in batch})
                pred_indices = {head: outputs[head].argmax(dim=-1).cpu().tolist() for head in HEAD_TO_LABEL if head in outputs}
                batch_size = len(batch["raw_examples"])
                for idx in range(batch_size):
                    total += 1
                    labels = {key: value[idx].item() for key, value in batch["labels"].items()}
                    predictions = {label: pred_indices[head][idx] for head, label in HEAD_TO_LABEL.items()}
                    gold_intent = label_encoder.decode(labels, batch["schema_items"][idx])["intent"]
                    by_intent[gold_intent]["total"] += 1
                    for label_key, gold_value in labels.items():
                        if gold_value == -1:
                            continue
                        counts[label_key] += 1
                        if predictions.get(label_key) == gold_value:
                            correct[label_key] += 1
                    if predictions.get("intent_label") == labels.get("intent_label"):
                        by_intent[gold_intent]["intent_correct"] += 1

                    row = batch["raw_examples"][idx]
                    schema = schema_from_example(row)
                    decoded = label_encoder.decode(predictions, batch["schema_items"][idx])
                    try:
                        query_ir = self.converter.convert(row.get("question", ""), schema, decoded)
                        ir_validation = self.ir_validator.validate(query_ir, schema=schema)
                        if ir_validation.is_valid:
                            query_ir_valid += 1
                            sql = self.sql_renderer.render(query_ir)
                            sql_validation = self.sql_validator.validate(sql, schema=schema, dialect=query_ir.dialect)
                            if sql_validation.get("is_valid"):
                                sql_valid += 1
                                by_intent[gold_intent]["sql_valid"] += 1
                            else:
                                _failure(failures, row, sql_validation.get("issues", []))
                        else:
                            _failure(failures, row, ir_validation.errors)
                    except Exception as exc:
                        _failure(failures, row, [str(exc)])

        return {
            "total_examples": total,
            "intent_accuracy": _acc(correct, counts, "intent_label"),
            "template_accuracy": _acc(correct, counts, "intent_label"),
            "base_table_accuracy": _acc(correct, counts, "base_table_index"),
            "metric_aggregation_accuracy": _acc(correct, counts, "metric_aggregation_label"),
            "metric_column_accuracy": _acc(correct, counts, "metric_column_index"),
            "dimension_column_accuracy": _acc(correct, counts, "dimension_column_index"),
            "date_column_accuracy": _acc(correct, counts, "date_column_index"),
            "filter_column_accuracy": _acc(correct, counts, "filter_column_index"),
            "metric_expression_type_accuracy": _acc(correct, counts, "metric_expression_type_label"),
            "date_grain_accuracy": _acc(correct, counts, "date_grain_label"),
            "filter_operator_accuracy": _acc(correct, counts, "filter_operator_label"),
            "order_direction_accuracy": _acc(correct, counts, "order_direction_label"),
            "limit_bucket_accuracy": _acc(correct, counts, "limit_bucket_label"),
            "query_ir_validity_rate": query_ir_valid / max(total, 1),
            "sql_validation_rate": sql_valid / max(total, 1),
            "execution_success_rate": None if db_path is None else 0.0,
            "end_to_end_case_pass_rate": None,
            "by_intent": {key: _intent_payload(value) for key, value in sorted(by_intent.items())},
            "sample_failures": failures[:25],
        }


def _acc(correct: dict[str, int], counts: dict[str, int], key: str) -> float:
    return correct.get(key, 0) / max(counts.get(key, 0), 1)


def _intent_payload(value: dict[str, int]) -> dict[str, float | int]:
    total = max(value["total"], 1)
    return {
        "total": value["total"],
        "intent_accuracy": value["intent_correct"] / total,
        "sql_validation_rate": value["sql_valid"] / total,
    }


def _failure(failures: list[dict[str, Any]], row: dict[str, Any], issues: list[Any]) -> None:
    if len(failures) >= 25:
        return
    failures.append({"example_id": row.get("example_id"), "question": row.get("question"), "issues": [str(item) for item in issues]})
