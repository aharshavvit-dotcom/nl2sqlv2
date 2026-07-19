"""Training diagnostics tracker.

Records per-epoch metrics and produces JSON plus Markdown reports.
"""

from __future__ import annotations

import json
import hashlib
import math
import time
from pathlib import Path
from typing import Any


class TrainingDiagnostics:
    """Accumulates per-epoch training diagnostics and writes reports."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        self.epochs: list[dict[str, Any]] = []
        self._start_time: float | None = None
        self.config_summary: dict[str, Any] = {}
        self._candidate_counts: list[int] = []
        self._candidate_padding_ratios: list[float] = []
        self.leakage_summary: dict[str, Any] = {}
        self.baseline_score: float | None = None

    def set_leakage_summary(self, summary: dict[str, Any]) -> None:
        self.leakage_summary = summary

    def set_baseline_score(self, score: float | None) -> None:
        self.baseline_score = score

    def set_config(self, config: dict[str, Any]) -> None:
        self._candidate_counts = []
        self._candidate_padding_ratios = []
        rat_config = config.get("model", {}).get("relation_aware_attention", {})
        curriculum_config = config.get("training", {}).get("curriculum", {})
        relation_enabled = bool(rat_config.get("enabled", False))
        pairwise_relation_matrix = bool(rat_config.get("pairwise_relation_matrix", True))
        relation_bias_mode = str(
            rat_config.get("relation_bias_mode")
            or ("schema_pairwise_relation_bias" if pairwise_relation_matrix else "schema_token_role_bias")
        )
        relation_types = rat_config.get("relation_types", [])

        # Phase 6: Track both bias paths independently
        question_schema_role_bias_active = bool(
            relation_enabled and relation_bias_mode in ("schema_token_role_bias", "combined")
        )
        schema_pairwise_relation_bias_active = bool(
            relation_enabled and relation_bias_mode in ("schema_pairwise_relation_bias", "combined")
        )
        candidate_pairwise_relation_bias_configured = bool(
            relation_enabled
            and relation_bias_mode in (
                "candidate_pairwise_relation_bias",
                "schema_candidate_pairwise_relation_bias",
                "combined",
            )
        )
        effective_mode = "disabled"
        if relation_enabled:
            effective_mode = relation_bias_mode

        max_tables = config.get("model", {}).get("max_tables", 0)
        max_columns = config.get("model", {}).get("max_columns", 0)
        padded_candidate_count = max_tables + max_columns
        candidate_matrix_size = padded_candidate_count ** 2
        candidate_graph = {
            "actual_candidate_count_min": 0,
            "actual_candidate_count_mean": 0.0,
            "actual_candidate_count_max": 0,
            "padded_candidate_count": padded_candidate_count,
            "candidate_matrix_size": candidate_matrix_size,
            "padding_ratio_mean": 0.0,
        }

        self.config_summary = {
            "optimizer_name": config.get("optimizer", {}).get("name", "unknown"),
            "activation_name": config.get("model", {}).get("activation", "unknown"),
            "learning_rate": config.get("optimizer", {}).get("learning_rate"),
            "batch_size": config.get("training", {}).get("batch_size"),
            "gradient_accumulation_steps": config.get("training", {}).get("gradient_accumulation_steps"),
            "epochs": config.get("training", {}).get("epochs"),
            "loss_weights": config.get("loss", {}),
            "hard_negative_weight": config.get("loss", {}).get("hard_negative"),
            "save_best_metric": config.get("training", {}).get("save_best_metric"),
            "checkpoint_monitor": config.get("training", {}).get("save_best_metric"),
            "checkpoint_mode": config.get("training", {}).get("save_best_mode"),
            "pointer_head_weight_decay": config.get("optimizer", {}).get("pointer_head_weight_decay"),
            "weight_decay": config.get("optimizer", {}).get("weight_decay"),
            "pointer_dropout": config.get("model", {}).get("pointer_dropout"),
            "early_stopping_patience": config.get("training", {}).get("early_stopping_patience"),
            "gradient_clipping_value": config.get("training", {}).get("gradient_clipping"),
            "train_path": config.get("data", {}).get("train_path"),
            "validation_path": config.get("data", {}).get("validation_path"),
            "hard_negatives_path": config.get("data", {}).get("hard_negatives_path"),
            "curriculum": {
                "enabled": bool(curriculum_config.get("enabled", True)),
                "active": True,
                "mode": curriculum_config.get("mode", "ordered_dataset"),
                "phased_epochs": False,
            },
            "relation_aware_attention": {
                "enabled": relation_enabled,
                "relation_type_ids_configured": question_schema_role_bias_active,
                "relation_type_ids_observed_in_dataset": False,
                "relation_type_ids_observed_in_batch": False,
                "relation_type_ids_used_in_forward": False,
                "schema_relation_type_ids_configured": schema_pairwise_relation_bias_active,
                "schema_relation_type_ids_observed_in_dataset": False,
                "schema_relation_type_ids_observed_in_batch": False,
                "schema_relation_type_ids_used_in_forward": False,
                "candidate_relation_type_ids_configured": candidate_pairwise_relation_bias_configured,
                "candidate_relation_type_ids_observed_in_dataset": False,
                "candidate_relation_type_ids_observed_in_batch": False,
                "candidate_relation_type_ids_used_in_forward": False,
                # Compatibility fields now reflect observed evidence, not config.
                "relation_type_ids_available": False,
                "schema_relation_type_ids_available": False,
                "candidate_relation_type_ids_available": False,
                "relation_bias_mode": effective_mode,
                "question_schema_role_bias_active": False,
                "schema_pairwise_relation_bias_active": False,
                "candidate_pairwise_relation_bias_active": False,
                "candidate_level_relation_graph_available": False,
                "candidate_relation_attention_uses_mask": False,
                "pairwise_relation_matrix": bool(pairwise_relation_matrix and relation_enabled),
                "relation_types": relation_types,
                "relation_bias_parameters": len(relation_types) if relation_enabled else 0,
                "candidate_relation_graph": candidate_graph,
            },
            "candidate_relation_graph": candidate_graph,
        }

    def observe_dataset_item(self, item: dict[str, Any]) -> None:
        """Record relation tensors that an actual dataset item emitted."""
        rat = self._relation_summary()
        for key in ("relation_type_ids", "schema_relation_type_ids", "candidate_relation_type_ids"):
            if key in item and item.get(key) is not None:
                rat[f"{key}_observed_in_dataset"] = True

    def observe_batch(self, batch: dict[str, Any]) -> None:
        """Record relation tensors and candidate utilization from a collated batch."""
        rat = self._relation_summary()
        for key in ("relation_type_ids", "schema_relation_type_ids", "candidate_relation_type_ids"):
            if key in batch and batch.get(key) is not None:
                rat[f"{key}_observed_in_batch"] = True
                rat[f"{key}_available"] = True

        table_mask = batch.get("table_candidate_mask")
        column_mask = batch.get("column_candidate_mask")
        if table_mask is None or column_mask is None:
            return
        try:
            import torch

            if not torch.is_tensor(table_mask) or not torch.is_tensor(column_mask):
                return
            if table_mask.dim() != 2 or column_mask.dim() != 2 or table_mask.size(0) != column_mask.size(0):
                return
            candidate_mask = torch.cat([table_mask, column_mask], dim=1).bool()
            padded_count = int(candidate_mask.size(1))
            counts = candidate_mask.sum(dim=1).detach().cpu().tolist()
        except (RuntimeError, TypeError, ValueError):
            return
        for count_value in counts:
            count = int(count_value)
            self._candidate_counts.append(count)
            ratio = 0.0 if padded_count <= 0 else 1.0 - (count / padded_count)
            self._candidate_padding_ratios.append(ratio)
        self._refresh_candidate_stats(padded_count)

    def observe_forward(self, outputs: dict[str, Any]) -> None:
        """Record relation paths the model reports it actually used."""
        rat = self._relation_summary()
        for key in ("relation_type_ids", "schema_relation_type_ids", "candidate_relation_type_ids"):
            used_key = f"{key}_used_in_forward"
            if bool(outputs.get(used_key, False)):
                rat[used_key] = True
        for key in (
            "question_schema_role_bias_active",
            "schema_pairwise_relation_bias_active",
            "candidate_pairwise_relation_bias_active",
            "candidate_level_relation_graph_available",
            "candidate_relation_attention_uses_mask",
        ):
            if bool(outputs.get(key, False)):
                rat[key] = True
        if outputs.get("relation_bias_mode"):
            rat["relation_bias_mode"] = str(outputs["relation_bias_mode"])

    def observe_step(self, batch: dict[str, Any], outputs: dict[str, Any]) -> None:
        self.observe_batch(batch)
        self.observe_forward(outputs)

    def _relation_summary(self) -> dict[str, Any]:
        return self.config_summary.setdefault("relation_aware_attention", {})

    def _refresh_candidate_stats(self, padded_count: int) -> None:
        graph = self._relation_summary().setdefault("candidate_relation_graph", {})
        graph.update({
            "actual_candidate_count_min": min(self._candidate_counts, default=0),
            "actual_candidate_count_mean": (
                sum(self._candidate_counts) / len(self._candidate_counts)
                if self._candidate_counts else 0.0
            ),
            "actual_candidate_count_max": max(self._candidate_counts, default=0),
            "padded_candidate_count": padded_count,
            "candidate_matrix_size": padded_count ** 2,
            "padding_ratio_mean": (
                sum(self._candidate_padding_ratios) / len(self._candidate_padding_ratios)
                if self._candidate_padding_ratios else 0.0
            ),
        })
        self.config_summary["candidate_relation_graph"] = graph

    def start_training(self) -> None:
        self._start_time = time.time()

    def record_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        lr: float | None = None,
        epoch_time: float | None = None,
        loss_by_head: dict[str, float] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_total_loss": train_metrics.get("loss", 0.0),
            "validation_total_loss": val_metrics.get("loss", 0.0),
            "intent_accuracy": val_metrics.get("intent_accuracy", 0.0),
            "base_table_accuracy": val_metrics.get("base_table_accuracy", val_metrics.get("metric_pointer_accuracy", 0.0)),
            "metric_column_accuracy": val_metrics.get("metric_column_accuracy", 0.0),
            "dimension_column_accuracy": val_metrics.get("dimension_column_accuracy", 0.0),
            "filter_column_accuracy": val_metrics.get("filter_column_accuracy", 0.0),
            "date_column_accuracy": val_metrics.get("date_column_accuracy", 0.0),
            "overall_slot_accuracy": val_metrics.get("overall_slot_accuracy", 0.0),
            "validation_composite_score": val_metrics.get("validation_composite_score", 0.0),
            "learning_rate": lr,
            "epoch_time_seconds": epoch_time,
        }
        if loss_by_head:
            row["loss_by_head"] = loss_by_head
        for source, target in [(train_metrics, "train_loss_samples"), (val_metrics, "validation_loss_samples")]:
            values = source.get("example_losses") or source.get("batch_losses") or []
            if isinstance(values, list):
                row[target] = [float(value) for value in values]
        high_loss = val_metrics.get("high_loss_examples") or train_metrics.get("high_loss_examples")
        if isinstance(high_loss, list):
            row["high_loss_examples"] = high_loss[:100]
        self.epochs.append(row)

    def best_epoch(self) -> dict[str, Any]:
        if not self.epochs:
            return {}
        metric = self.config_summary.get("checkpoint_monitor") or "overall_slot_accuracy"
        mode = self.config_summary.get("checkpoint_mode") or "max"
        epoch_key = "validation_total_loss" if metric == "loss" else metric
        return (min if mode == "min" else max)(
            self.epochs,
            key=lambda e: e.get(epoch_key, float("inf") if mode == "min" else float("-inf")),
        )

    def to_dict(self) -> dict[str, Any]:
        total_time = (time.time() - self._start_time) if self._start_time else None
        best = self.best_epoch()
        train_losses = [value for epoch in self.epochs for value in (epoch.get("train_loss_samples") or [epoch.get("train_total_loss", 0.0)])]
        validation_losses = [value for epoch in self.epochs for value in (epoch.get("validation_loss_samples") or [epoch.get("validation_total_loss", 0.0)])]
        high_loss_examples = [item for epoch in self.epochs for item in (epoch.get("high_loss_examples") or [])]
        current = self.epochs[-1] if self.epochs else {}
        best_val_loss = min(
            (epoch.get("validation_total_loss") for epoch in self.epochs),
            default=None,
        )
        current_val_loss = current.get("validation_total_loss")
        current_train_loss = current.get("train_total_loss")
        ratio = (
            current_val_loss / current_train_loss
            if isinstance(current_val_loss, (int, float))
            and isinstance(current_train_loss, (int, float))
            and current_train_loss > 0
            else None
        )
        return {
            "effective_epochs": self.config_summary.get("epochs"),
            "effective_batch_size": self.config_summary.get("batch_size"),
            "save_best_metric": self.config_summary.get("save_best_metric"),
            "save_best_mode": self.config_summary.get("checkpoint_mode"),
            "early_stopping_patience": self.config_summary.get("early_stopping_patience"),
            "checkpoint_selected_epoch": best.get("epoch"),
            "checkpoint_selected_metric": self.config_summary.get("checkpoint_monitor"),
            "best_validation_loss": best_val_loss,
            "best_validation_slot_accuracy": best.get("overall_slot_accuracy"),
            "final_train_loss": current_train_loss,
            "final_validation_loss": current_val_loss,
            "weight_decay": self.config_summary.get("weight_decay"),
            "effective_config_hash": _effective_config_hash(self.config_summary),
            "config": self.config_summary,
            "total_training_time_seconds": total_time,
            "total_epochs": len(self.epochs),
            "best_epoch": best.get("epoch"),
            "best_overall_slot_accuracy": best.get("overall_slot_accuracy"),
            "checkpoint_monitor": self.config_summary.get("checkpoint_monitor"),
            "checkpoint_mode": self.config_summary.get("checkpoint_mode"),
            "pointer_head_weight_decay": self.config_summary.get("pointer_head_weight_decay"),
            "pointer_dropout": self.config_summary.get("pointer_dropout"),
            "best_val_loss": best_val_loss,
            "current_val_loss": current_val_loss,
            "val_train_loss_ratio": ratio,
            "overfitting_warning": bool(ratio is not None and ratio > 1.5),
            "loss_percentiles": {
                **{f"train_loss_p{p}": _percentile(train_losses, p) for p in [50, 95, 99]},
                **{f"validation_loss_p{p}": _percentile(validation_losses, p) for p in [50, 95, 99]},
            },
            "top_p95_high_loss_examples": high_loss_examples[:50],
            "top_p99_high_loss_examples": high_loss_examples[:10],
            "leakage_summary": self.leakage_summary,
            "baseline_score": self.baseline_score,
            "epochs": self.epochs,
        }

    def save(self, output_dir: str | Path | None = None) -> None:
        target = Path(output_dir or self.output_dir or ".")
        target.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        (target / "training_diagnostics.json").write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
        (target / "training_diagnostics.md").write_text(
            _render_markdown(data), encoding="utf-8"
        )


def _render_markdown(data: dict[str, Any]) -> str:
    lines = ["# Neural QueryIR Training Diagnostics", ""]
    cfg = data.get("config", {})
    lines.append("## Configuration")
    lines.append(f"- **Optimizer**: {cfg.get('optimizer_name', '-')}")
    lines.append(f"- **Activation**: {cfg.get('activation_name', '-')}")
    lines.append(f"- **Learning rate**: {cfg.get('learning_rate', '-')}")
    lines.append(f"- **Batch size**: {cfg.get('batch_size', '-')}")
    lines.append(f"- **Gradient accumulation steps**: {cfg.get('gradient_accumulation_steps', '-')}")
    lines.append(f"- **Gradient clipping**: {cfg.get('gradient_clipping_value', '-')}")
    lines.append(f"- **Train path**: {cfg.get('train_path', '-')}")
    lines.append(f"- **Validation path**: {cfg.get('validation_path', '-')}")
    
    loss_weights = cfg.get("loss_weights")
    if loss_weights:
        lines.append("")
        lines.append("### Loss Weights Configuration Footprint")
        for k, v in sorted(loss_weights.items()):
            lines.append(f"- **{k}**: {v}")
            
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- **Total epochs**: {data.get('total_epochs', 0)}")
    lines.append(f"- **Best epoch**: {data.get('best_epoch', '-')}")
    lines.append(f"- **Best slot accuracy**: {data.get('best_overall_slot_accuracy', 0):.4f}")
    
    # Baseline vs Current Comparison
    baseline = data.get("baseline_score")
    best_epoch_data = data.get("epochs", [])[data.get("best_epoch", 0) - 1] if data.get("epochs") and data.get("best_epoch") else {}
    current_score = best_epoch_data.get("semantic_checkpoint_score") or data.get("best_overall_slot_accuracy") or 0.0
    
    lines.append(f"- **Current Best Diagnostic Component Score**: {current_score:.4f}")
    lines.append("- **Diagnostic score note**: not valid for production checkpoint selection")
    if baseline is not None:
        lines.append(f"- **Baseline Score**: {float(baseline):.4f}")
        diff = current_score - float(baseline)
        lines.append(f"- **Improvement over Baseline**: {diff:+.4f}")
    else:
        lines.append("- **Baseline Score**: Not available")

    total_time = data.get("total_training_time_seconds")
    if total_time is not None:
        lines.append(f"- **Total training time**: {total_time:.1f}s")
    lines.append("")
    
    # Data Leakage Audit Summary
    leakage = data.get("leakage_summary") or {}
    lines.append("## Data Leakage Audit")
    if leakage:
        lines.append(f"- **Status**: {'PASSED (No Leakage)' if leakage.get('ok', True) else 'FAILED (Leakage Detected)'}")
        lines.append(f"- **Total issues found**: {leakage.get('total_issues', 0)}")
        if leakage.get("issues"):
            lines.append("#### Issues List:")
            for issue in leakage.get("issues", [])[:10]:
                lines.append(f"- {issue}")
    else:
        lines.append("- **Status**: Audit not available or skipped")
    lines.append("")

    epochs = data.get("epochs", [])
    if epochs:
        lines.append("## Per-Epoch Metrics")
        lines.append("")
        lines.append("| Epoch | Train Loss | Val Loss | Intent Acc | Slot Acc | Diagnostic Component Score | LR | Time (s) |")
        lines.append("|------:|-----------:|---------:|-----------:|---------:|---------------------------:|---:|---------:|")
        for epoch in epochs:
            lines.append(
                f"| {epoch.get('epoch', '-')} "
                f"| {epoch.get('train_total_loss', 0):.4f} "
                f"| {epoch.get('validation_total_loss', 0):.4f} "
                f"| {epoch.get('intent_accuracy', 0):.4f} "
                f"| {epoch.get('overall_slot_accuracy', 0):.4f} "
                f"| {epoch.get('semantic_checkpoint_score', epoch.get('support_weighted_semantic_score', 0)):.4f} "
                f"| {epoch.get('learning_rate', '-')} "
                f"| {epoch.get('epoch_time_seconds', '-')} |"
            )
    return "\n".join(lines) + "\n"


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _effective_config_hash(summary: dict[str, Any]) -> str:
    payload = {
        "epochs": summary.get("epochs"),
        "batch_size": summary.get("batch_size"),
        "gradient_accumulation_steps": summary.get("gradient_accumulation_steps"),
        "save_best_metric": summary.get("save_best_metric"),
        "save_best_mode": summary.get("checkpoint_mode"),
        "early_stopping_patience": summary.get("early_stopping_patience"),
        "hard_negative_weight": summary.get("hard_negative_weight"),
        "weight_decay": summary.get("weight_decay"),
        "pointer_head_weight_decay": summary.get("pointer_head_weight_decay"),
        "pointer_dropout": summary.get("pointer_dropout"),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
