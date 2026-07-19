"""
Purpose: Protects ir unit behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import hashlib
import json

import torch

from capabilities import ALL_CAPABILITIES, ALL_SAFETY_LABELS
from inference.prediction_models import PredictionResult
from neural_ir.ir_dataset import COMPLEXITY_LABELS
from neural_optimization.loss_registry import masked_cross_entropy_fn
from neural_optimization.training_config import NeuralTrainingConfig
from nl2sql_v1.schema import ColumnInfo, SchemaGraph, TableInfo
from orchestration.pipeline_config import build_pipeline_steps
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel
from training.train_neural_ir_optimized import (
    _metrics_from_counts,
    _semantic_checkpoint_metrics,
    _supervised_head_losses,
)


def test_auxiliary_losses_use_task_masks_without_training_full_ir_head() -> None:
    outputs = {
        "intent_logits": torch.tensor([[3.0, -1.0]], requires_grad=True),
        "complexity_logits": torch.zeros((1, len(COMPLEXITY_LABELS)), requires_grad=True),
        "capability_logits": torch.zeros((1, len(ALL_CAPABILITIES)), requires_grad=True),
        "safety_logits": torch.zeros((1, len(ALL_SAFETY_LABELS)), requires_grad=True),
    }
    capability_labels = torch.zeros((1, len(ALL_CAPABILITIES)))
    safety_labels = torch.zeros((1, len(ALL_SAFETY_LABELS)))
    capability_labels[0, 0] = 1.0
    safety_labels[0, 0] = 1.0
    batch = {
        "labels": {"intent_label": torch.tensor([0]), "complexity_label": torch.tensor([1])},
        "capability_labels": capability_labels,
        "safety_labels": safety_labels,
        "task_masks": {
            "full_query_ir": torch.tensor([0.0]),
            "capability": torch.tensor([1.0]),
            "safety": torch.tensor([1.0]),
            "complexity": torch.tensor([1.0]),
        },
    }

    losses = _supervised_head_losses(
        outputs,
        batch,
        {"intent_logits": "intent_label", "complexity_logits": "complexity_label"},
        {},
        {"intent_logits": "intent", "complexity_logits": "complexity"},
        masked_cross_entropy_fn,
    )
    loss = sum(losses.values())
    loss.backward()

    assert "intent" not in losses
    assert "complexity" in losses
    assert outputs["intent_logits"].grad is None
    assert outputs["complexity_logits"].grad.abs().sum().item() > 0
    assert outputs["capability_logits"].grad.abs().sum().item() > 0
    assert outputs["safety_logits"].grad.abs().sum().item() > 0


def test_semantic_checkpoint_metric_names_are_diagnostic_not_misleading() -> None:
    metrics = _semantic_checkpoint_metrics(
        {
            "intent_label": 9,
            "metric_column_index": 8,
            "filter_column_index": 7,
            "span": 6,
            "dimension_column_index": 5,
        },
        {
            "intent_label": 10,
            "metric_column_index": 10,
            "filter_column_index": 10,
            "span": 10,
            "dimension_column_index": 10,
        },
        NeuralTrainingConfig(),
    )
    values = metrics["semantic_checkpoint_metric_values"]

    assert "intent_accuracy" in values
    assert "metric_column_pointer_accuracy" in values
    assert "component_accuracy_floor" in values
    assert "intent_macro_f1" not in values
    assert "projection_exact_match" not in values
    assert metrics["semantic_checkpoint_score_valid_for_checkpoint_selection"] is False


def test_true_intent_macro_f1_is_not_alias_for_accuracy() -> None:
    metrics = _metrics_from_counts(
        0.1,
        {"intent_label": 3},
        {"intent_label": 4},
        NeuralTrainingConfig(),
        example_metrics={
            "intent_targets": [0, 0, 0, 1],
            "intent_predictions": [0, 0, 0, 0],
            "projection_exact_match_correct": 1,
            "projection_exact_match_total": 2,
            "semantic_pass_correct": 1,
            "semantic_pass_total": 4,
        },
    )

    assert metrics["intent_accuracy"] == 0.75
    assert round(metrics["intent_macro_f1"], 4) == 0.4286
    assert metrics["projection_exact_match_rate"] == 0.5
    assert metrics["semantic_pass_rate"] == 0.25


def test_runtime_normalization_does_not_rewrite_customer_sales_sql() -> None:
    schema = SchemaGraph(
        tables={
            "orders": TableInfo(
                name="orders",
                columns={
                    "customer_id": ColumnInfo("customer_id", "INTEGER", False, False),
                    "amount": ColumnInfo("amount", "REAL", False, False),
                },
            ),
            "customers": TableInfo(
                name="customers",
                columns={"customer_id": ColumnInfo("customer_id", "INTEGER", False, True)},
            ),
        }
    )
    result = PredictionResult(
        question="Top customers by sales",
        normalized_question="top customers by sales",
        source_model="retrieval",
        sql='SELECT "orders"."customer_id" FROM "orders"',
        validation={"is_valid": True},
        slots={"limit": {"value": 5}},
    )

    normalized = RetrievalNL2SQLModel._normalize_runtime_result(result.question, schema, result)

    assert normalized.source_model == "retrieval_ir"
    assert normalized.sql == result.sql
    assert "JOIN" not in normalized.sql


def test_runtime_cache_identity_uses_bundle_content_hashes(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    retrieval_dir = bundle_dir / "retrieval_ir"
    neural_dir = bundle_dir / "neural_ir"
    retrieval_dir.mkdir(parents=True)
    neural_dir.mkdir()
    manifest = {
        "bundle_id": "bundle-a",
        "paths": {"retrieval_ir": "retrieval_ir/", "neural_ir": "neural_ir/"},
        "routing_policy": {"neural_fallback_enabled": True},
    }
    (bundle_dir / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    retrieval_payload = b'{"artifact":"retrieval-v1"}'
    checkpoint_payload = b"weights-v1"
    (retrieval_dir / "manifest.json").write_bytes(retrieval_payload)
    (neural_dir / "model.pt").write_bytes(checkpoint_payload)
    model = RetrievalNL2SQLModel(
        retriever=object(),
        artifact_dir=retrieval_dir,
        metadata={"model_bundle": manifest, "model_bundle_dir": str(bundle_dir)},
        neural_ir_model_dir=neural_dir,
    )

    identity = model._cache_bundle_identity(manifest["routing_policy"])

    assert identity.bundle_id == "bundle-a"
    assert identity.retrieval_artifact_hash == hashlib.sha256(retrieval_payload).hexdigest()
    assert identity.checkpoint_state_dict_hash == hashlib.sha256(checkpoint_payload).hexdigest()


def test_route_diagnostics_step_is_enabled_by_config() -> None:
    steps = build_pipeline_steps({"evaluation": {"enabled": True, "run_route_diagnostics": True}})

    assert "evaluate_generic_models" in steps
    assert "run_route_diagnostics" in steps
    assert steps.index("evaluate_generic_models") < steps.index("run_route_diagnostics")


def test_multi_seed_variance_runs_before_quality_gate_and_bundle() -> None:
    steps = build_pipeline_steps({
        "evaluation": {"enabled": True},
        "seeds": {"enabled": True},
        "bundle": {"build": True},
    })

    assert "multi_seed_variance" in steps
    assert steps.index("evaluate_generic_models") < steps.index("multi_seed_variance")
    assert steps.index("multi_seed_variance") < steps.index("run_quality_gate")
    assert steps.index("multi_seed_variance") < steps.index("build_model_bundle")
