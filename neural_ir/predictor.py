from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ir.query_ir_models import QueryIR
from ir.ir_to_sql_renderer import IRToSQLRenderer
from ir.ir_validator import IRValidator
from validation.sql_validator import SQLValidator

from .candidate_builder import SchemaCandidateBuilder, build_candidate_masks, schema_link_score_vector
from .confidence_calibrator import NeuralIRConfidenceCalibrator
from .ir_repair import NeuralIRRepairer
from .ir_dataset import (
    build_candidate_metadata,
    build_candidate_pairwise_relation_matrix,
    build_question_schema_relation_type_ids,
    build_schema_relation_type_ids,
)
from .model_registry import load_model_bundle
from .option_a_to_ir import NeuralIRToIRConverter
from .schema_linearizer import SchemaLinearizer, extract_schema_items
from .schema_linker import SchemaLinker
from .tokenizer import tokenize


class NeuralIRPredictor:
    """Predicts QueryIR using the neural model.

    Formerly named ``OptionAIRPredictor``.
    """
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        bundle = load_model_bundle(self.model_dir)
        self.model = bundle["model"]
        self.vocab = bundle["vocab"]
        self.label_encoder = bundle["label_encoder"]
        self.config = bundle["config"]
        self.model.eval()
        self.linearizer = SchemaLinearizer()
        self.candidate_builder = SchemaCandidateBuilder()
        self.schema_linker = SchemaLinker()
        self.converter = NeuralIRToIRConverter()
        self.repairer = NeuralIRRepairer()
        self.calibrator = NeuralIRConfidenceCalibrator.load(str(self.model_dir / "option_a_calibration.json"))
        self.ir_validator = IRValidator()
        self.sql_renderer = IRToSQLRenderer()
        self.sql_validator = SQLValidator()

    def predict(self, question: str, schema: dict) -> dict[str, Any]:
        schema_items = extract_schema_items(schema)
        schema_text = self.linearizer.linearize(schema)
        candidates = self.candidate_builder.build_candidates(schema, question)
        link_result = self.schema_linker.link(question, candidates)
        candidate_masks = build_candidate_masks(
            candidates,
            int(self.config.get("max_tables", 64)),
            int(self.config.get("max_columns", 256)),
        )
        link_scores = schema_link_score_vector(link_result, int(self.config.get("max_columns", 256)))
        question_ids = torch.tensor(
            [self.vocab.encode(tokenize(question), int(self.config.get("max_question_len", 64)))],
            dtype=torch.long,
        )
        schema_ids = torch.tensor(
            [self.vocab.encode(tokenize(schema_text), int(self.config.get("max_schema_len", 256)))],
            dtype=torch.long,
        )
        schema_tokens = tokenize(schema_text)
        question_mask = question_ids.ne(self.vocab.pad_id).float()
        schema_mask = schema_ids.ne(self.vocab.pad_id).float()
        tensor_masks = {
            key: torch.tensor([value], dtype=torch.float32)
            for key, value in candidate_masks.items()
            if key.endswith("_mask")
        }
        candidate_tokens = _candidate_token_tensors(
            candidates,
            self.vocab,
            int(self.config.get("max_tables", 64)),
            int(self.config.get("max_columns", 256)),
            int(self.config.get("max_candidate_tokens", 16)),
        )
        schema_link_scores = torch.tensor([link_scores], dtype=torch.float32)
        relation_tensors: dict[str, torch.Tensor] = {}
        if (self.config.get("relation_aware_attention") or {}).get("enabled", False):
            max_question_len = int(self.config.get("max_question_len", 64))
            max_schema_len = int(self.config.get("max_schema_len", 256))
            max_tables = int(self.config.get("max_tables", 64))
            max_columns = int(self.config.get("max_columns", 256))
            relation_tensors = {
                "relation_type_ids": torch.tensor([
                    build_question_schema_relation_type_ids(schema_items, schema_tokens, max_question_len, max_schema_len)
                ], dtype=torch.long),
                "schema_relation_type_ids": torch.tensor([
                    build_schema_relation_type_ids(schema_items, schema_tokens, max_schema_len)
                ], dtype=torch.long),
                "candidate_relation_type_ids": torch.tensor([
                    build_candidate_pairwise_relation_matrix(
                        build_candidate_metadata(candidates, schema_items, max_tables),
                        max_tables + max_columns,
                    )
                ], dtype=torch.long),
            }
        with torch.no_grad():
            outputs = self.model(
                question_ids=question_ids,
                schema_ids=schema_ids,
                question_mask=question_mask,
                schema_mask=schema_mask,
                schema_link_scores=schema_link_scores,
                **tensor_masks,
                **candidate_tokens,
                **relation_tensors,
            )
        prediction_indices = _prediction_indices(outputs)
        decoded = self.label_encoder.decode(prediction_indices, schema_items)
        raw_summary = _logit_summary(outputs)
        raw_confidence = _confidence(outputs)
        confidence = raw_confidence
        warnings: list[str] = []
        repair_payload = {"query_ir": None, "repairs_applied": [], "repair_warnings": [], "repair_success": False}
        try:
            query_ir = self.converter.convert(question, schema, decoded)
            before_ir_validation = self.ir_validator.validate(query_ir, schema=schema)
            repair_payload = self.repairer.repair(query_ir, schema=schema, question=question, validation_result=before_ir_validation)
            repaired_query_ir = _query_ir_from_payload(repair_payload.get("query_ir")) if repair_payload.get("query_ir") else query_ir
            ir_validation = self.ir_validator.validate(repaired_query_ir, schema=schema)
            sql = self.sql_renderer.render(repaired_query_ir) if ir_validation.is_valid else None
            sql_validation = self.sql_validator.validate(sql, schema=schema, dialect=repaired_query_ir.dialect)
            query_ir_payload = query_ir.model_dump()
            repaired_query_ir_payload = repaired_query_ir.model_dump()
            ir_payload = ir_validation.model_dump()
            warnings.extend(str(item) for item in query_ir.warnings)
            warnings.extend(str(item) for item in repaired_query_ir.warnings)
            warnings.extend(str(item) for item in ir_validation.warnings)
            warnings.extend(str(item) for item in ir_validation.errors)
            warnings.extend(str(item) for item in repair_payload.get("repair_warnings", []))
            if not sql_validation.get("is_valid", sql_validation.get("ok", False)):
                warnings.extend(str(item) for item in sql_validation.get("issues", []))
                confidence = min(confidence, 0.40)
            if not ir_validation.is_valid:
                confidence = min(confidence, 0.20)
            confidence = self.calibrator.calibrate(
                raw_confidence=raw_confidence,
                validation_summary={
                    "ir_validation": ir_payload,
                    "sql_validation": sql_validation,
                    "repairs": repair_payload,
                },
                prediction_debug={
                    "decoded_prediction": decoded,
                    "schema_linking": link_result,
                    "candidate_scores": _candidate_score_debug(outputs),
                    "repairs": repair_payload,
                    "raw_logits_summary": raw_summary,
                },
            )
        except Exception as exc:
            sql = None
            sql_validation = {"is_valid": False, "ok": False, "issues": [str(exc)], "message": "Option A conversion failed"}
            query_ir_payload = None
            repaired_query_ir_payload = None
            ir_payload = {"is_valid": False, "errors": [str(exc)], "warnings": [], "issues": []}
            warnings.append(str(exc))
            confidence = min(confidence, 0.10)
        neural_spans = []
        if "span_logits" in outputs:
            neural_spans = _decode_spans(question, outputs["span_logits"])
            
        return {
            "source_model": "neural_ir",
            "neural_ir_version": str(self.config.get("model_version") or "schema_aware_queryir_v1"),
            "query_ir": query_ir_payload,
            "repaired_query_ir": repaired_query_ir_payload,
            "repairs_applied": repair_payload.get("repairs_applied", []),
            "ir_validation": ir_payload,
            "sql": sql,
            "sql_validation": sql_validation,
            "validation": sql_validation,
            "raw_confidence": raw_confidence,
            "calibrated_confidence": confidence,
            "confidence": confidence,
            "neural_spans": neural_spans,
            "warnings": list(dict.fromkeys(warnings)),
            "debug": {
                "decoded_prediction": decoded,
                "raw_logits_summary": raw_summary,
                "prediction_indices": prediction_indices,
                "schema_candidates": candidates,
                "schema_linking": link_result,
                "candidate_scores": _candidate_score_debug(outputs),
                "attention": _attention_debug(outputs),
                "repairs": repair_payload,
                "calibration": {
                    "raw_confidence": raw_confidence,
                    "calibrated_confidence": confidence,
                },
                "candidate_warnings": candidate_masks.get("candidate_warnings", []),
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
    for head, logits in outputs.items():
        if not head.endswith("_logits") or not torch.is_tensor(logits):
            continue
        probs.append(float(torch.softmax(logits, dim=-1).max(dim=-1).values.item()))
    return sum(probs) / max(len(probs), 1)


def _logit_summary(outputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        head: {
            "argmax": int(logits.argmax(dim=-1).item()),
            "max_probability": float(torch.softmax(logits, dim=-1).max(dim=-1).values.item()),
        }
        for head, logits in outputs.items()
        if head.endswith("_logits") and torch.is_tensor(logits)
    }


def _candidate_token_tensors(candidates: dict[str, Any], vocab, max_tables: int, max_columns: int, max_candidate_tokens: int) -> dict[str, torch.Tensor]:
    table_ids = _candidate_token_ids(candidates.get("tables", []), vocab, max_tables, max_candidate_tokens)
    column_ids = _candidate_token_ids(candidates.get("columns", []), vocab, max_columns, max_candidate_tokens)
    return {
        "table_candidate_token_ids": torch.tensor([table_ids], dtype=torch.long),
        "column_candidate_token_ids": torch.tensor([column_ids], dtype=torch.long),
        "candidate_token_ids": torch.tensor([column_ids], dtype=torch.long),
    }


def _candidate_token_ids(candidates: list[dict[str, Any]], vocab, max_candidates: int, max_candidate_tokens: int) -> list[list[int]]:
    rows = [[vocab.pad_id] * max_candidate_tokens for _ in range(max_candidates)]
    for candidate in candidates:
        index = int(candidate.get("index", -1))
        if 0 <= index < max_candidates:
            rows[index] = vocab.encode(list(candidate.get("tokens") or tokenize(str(candidate.get("display") or ""))), max_candidate_tokens)
    return rows


def _query_ir_from_payload(payload: Any) -> QueryIR:
    if isinstance(payload, QueryIR):
        return payload
    if hasattr(QueryIR, "model_validate"):
        return QueryIR.model_validate(payload)
    return QueryIR.parse_obj(payload)


def _candidate_score_debug(outputs: dict[str, Any]) -> dict[str, Any]:
    raw = outputs.get("candidate_scores") or {}
    return {key: value.detach().cpu().squeeze(0).tolist() if torch.is_tensor(value) else value for key, value in raw.items()}


def _attention_debug(outputs: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if torch.is_tensor(outputs.get("top_schema_candidates")):
        payload["top_schema_candidates"] = outputs["top_schema_candidates"].detach().cpu().squeeze(0).tolist()
    if torch.is_tensor(outputs.get("attention_weights")):
        weights = outputs["attention_weights"].detach().cpu()
        payload["shape"] = list(weights.shape)
    return payload


def _decode_spans(question: str, span_logits: torch.Tensor) -> list[str]:
    import re
    token_re = re.compile(r"[a-z0-9_]+")
    token_spans = []
    for m in token_re.finditer(question.lower()):
        token_spans.append((m.start(), m.end(), m.group()))
        
    preds = span_logits.squeeze(0).argmax(dim=-1).cpu().tolist()
    spans = []
    current_span = []
    for idx, pred in enumerate(preds):
        if idx >= len(token_spans):
            break
        if pred == 1:
            current_span.append(idx)
        else:
            if current_span:
                spans.append(current_span)
                current_span = []
    if current_span:
        spans.append(current_span)
        
    extracted_values = []
    for span in spans:
        start_char = token_spans[span[0]][0]
        end_char = token_spans[span[-1]][1]
        extracted_values.append(question[start_char:end_char])
    return extracted_values


# Backward-compatible alias
OptionAIRPredictor = NeuralIRPredictor
"""Deprecated alias. Use ``NeuralIRPredictor``."""
