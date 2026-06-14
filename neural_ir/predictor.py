from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ir.ir_to_sql_renderer import IRToSQLRenderer
from ir.ir_validator import IRValidator
from validation.sql_validator import SQLValidator

from .model_registry import load_model_bundle
from .option_a_to_ir import OptionAToIRConverter
from .schema_linearizer import SchemaLinearizer, extract_schema_items
from .tokenizer import tokenize


class OptionAIRPredictor:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        bundle = load_model_bundle(self.model_dir)
        self.model = bundle["model"]
        self.vocab = bundle["vocab"]
        self.label_encoder = bundle["label_encoder"]
        self.config = bundle["config"]
        self.model.eval()
        self.linearizer = SchemaLinearizer()
        self.converter = OptionAToIRConverter()
        self.ir_validator = IRValidator()
        self.sql_renderer = IRToSQLRenderer()
        self.sql_validator = SQLValidator()

    def predict(self, question: str, schema: dict) -> dict[str, Any]:
        schema_items = extract_schema_items(schema)
        schema_text = self.linearizer.linearize(schema)
        question_ids = torch.tensor(
            [self.vocab.encode(tokenize(question), int(self.config.get("max_question_len", 64)))],
            dtype=torch.long,
        )
        schema_ids = torch.tensor(
            [self.vocab.encode(tokenize(schema_text), int(self.config.get("max_schema_len", 256)))],
            dtype=torch.long,
        )
        question_mask = question_ids.ne(self.vocab.pad_id).float()
        schema_mask = schema_ids.ne(self.vocab.pad_id).float()
        with torch.no_grad():
            outputs = self.model(question_ids, schema_ids, question_mask, schema_mask)
        prediction_indices = _prediction_indices(outputs)
        decoded = self.label_encoder.decode(prediction_indices, schema_items)
        raw_summary = _logit_summary(outputs)
        confidence = _confidence(outputs)
        try:
            query_ir = self.converter.convert(question, schema, decoded)
            ir_validation = self.ir_validator.validate(query_ir, schema=schema)
            sql = self.sql_renderer.render(query_ir) if ir_validation.is_valid else None
            sql_validation = self.sql_validator.validate(sql, schema=schema, dialect=query_ir.dialect)
            query_ir_payload = query_ir.model_dump()
            ir_payload = ir_validation.model_dump()
        except Exception as exc:
            sql = None
            sql_validation = {"is_valid": False, "ok": False, "issues": [str(exc)], "message": "Option A conversion failed"}
            query_ir_payload = None
            ir_payload = {"is_valid": False, "errors": [str(exc)], "warnings": [], "issues": []}
        return {
            "query_ir": query_ir_payload,
            "ir_validation": ir_payload,
            "sql": sql,
            "sql_validation": sql_validation,
            "validation": sql_validation,
            "confidence": confidence,
            "debug": {
                "decoded_prediction": decoded,
                "raw_logits_summary": raw_summary,
                "prediction_indices": prediction_indices,
            },
        }


def _prediction_indices(outputs: dict[str, torch.Tensor]) -> dict[str, int]:
    mapping = {
        "intent_logits": "intent_label",
        "base_table_logits": "base_table_index",
        "metric_aggregation_logits": "metric_aggregation_label",
        "metric_column_logits": "metric_column_index",
        "metric_expression_type_logits": "metric_expression_type_label",
        "dimension_column_logits": "dimension_column_index",
        "date_column_logits": "date_column_index",
        "date_grain_logits": "date_grain_label",
        "date_filter_type_logits": "date_filter_type_label",
        "filter_column_logits": "filter_column_index",
        "filter_operator_logits": "filter_operator_label",
        "order_direction_logits": "order_direction_label",
        "limit_bucket_logits": "limit_bucket_label",
    }
    return {label: int(outputs[head].argmax(dim=-1).item()) for head, label in mapping.items()}


def _confidence(outputs: dict[str, torch.Tensor]) -> float:
    probs = []
    for logits in outputs.values():
        probs.append(float(torch.softmax(logits, dim=-1).max(dim=-1).values.item()))
    return sum(probs) / max(len(probs), 1)


def _logit_summary(outputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        head: {
            "argmax": int(logits.argmax(dim=-1).item()),
            "max_probability": float(torch.softmax(logits, dim=-1).max(dim=-1).values.item()),
        }
        for head, logits in outputs.items()
    }
