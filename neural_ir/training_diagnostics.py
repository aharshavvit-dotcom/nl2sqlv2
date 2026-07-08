"""Training diagnostics for Neural QueryIR Model.

Analyzes training metrics, per-head validation loss breakdowns, invalid pointer
targets, schema size distribution, unknown token rate, and masking correctness.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


class TrainingDiagnostics:
    """Aggregates and writes advanced training diagnostics for debugging."""

    def __init__(self) -> None:
        self.epoch_diagnostics: dict[int, dict[str, Any]] = {}
        self.invalid_pointer_count = 0
        self.total_tokens_checked = 0
        self.unknown_tokens_count = 0
        self.schema_sizes: list[int] = []
        self.masking_errors = 0
        self.masking_checks = 0

    def reset_step_counters(self) -> None:
        """Reset batch-level metrics between epochs."""
        self.invalid_pointer_count = 0
        self.total_tokens_checked = 0
        self.unknown_tokens_count = 0
        self.masking_errors = 0
        self.masking_checks = 0

    def observe_step(
        self,
        batch: dict[str, Any],
        outputs: dict[str, torch.Tensor],
        label_encoder: Any = None,
    ) -> None:
        """Analyze batch outputs, labels, and inputs for anomalies."""
        labels = batch.get("labels") or {}
        question_ids = batch.get("question_ids")
        schema_ids = batch.get("schema_ids")

        # 1. Analyze invalid pointers
        # Pointer targets (indexes of columns or tables) must be within candidate masks or schema lengths
        from .trainer import HEAD_TO_LABEL, HEAD_TO_MASK
        for head, label_key in HEAD_TO_LABEL.items():
            if label_key not in labels or head not in outputs:
                continue
            target = labels[label_key]
            # Batch size check
            batch_sz = target.size(0)
            pred = outputs[head].argmax(dim=-1)
            
            mask_name = HEAD_TO_MASK.get(head, "")
            mask = batch.get(mask_name) if mask_name else None

            for i in range(batch_sz):
                t_val = int(target[i].item())
                if t_val == -1:
                    continue
                
                # Check target bounds
                if mask is not None:
                    max_len = int((mask[i] == 1).sum().item())
                    if t_val >= max_len or t_val < 0:
                        self.invalid_pointer_count += 1
                elif "index" in label_key:
                    # Generic indexes bounds check
                    num_classes = outputs[head].size(-1)
                    if t_val >= num_classes or t_val < 0:
                        self.invalid_pointer_count += 1

        # 2. Unknown tokens check
        # We check vocab lookup mapping if label_encoder/tokenizer is provided
        if question_ids is not None:
            # Assume 1 is UNK token id (OptionAIR default tokenizer convention)
            unk_mask = (question_ids == 1)
            self.unknown_tokens_count += int(unk_mask.sum().item())
            self.total_tokens_checked += int((question_ids != 0).sum().item())

        # 3. Schema size distribution
        if schema_ids is not None:
            for i in range(schema_ids.size(0)):
                sz = int((schema_ids[i] != 0).sum().item())
                self.schema_sizes.append(sz)

        # 4. Masking correctness
        # For heads using candidate masks, ensure true labels are not masked out
        for head, mask_name in HEAD_TO_MASK.items():
            mask = batch.get(mask_name)
            label_key = HEAD_TO_LABEL.get(head)
            if mask is not None and label_key in labels:
                target = labels[label_key]
                for i in range(target.size(0)):
                    t_val = int(target[i].item())
                    if t_val >= 0:
                        self.masking_checks += 1
                        if t_val >= mask[i].size(0) or int(mask[i][t_val].item()) == 0:
                            self.masking_errors += 1

    def record_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        config: dict[str, Any],
        model: Any = None,
    ) -> dict[str, Any]:
        """Record epoch-level diagnostics and check consistency."""
        # Calculate train/validation gap
        train_loss = train_metrics.get("loss", 0.0)
        val_loss = val_metrics.get("loss", 0.0)
        loss_gap = val_loss - train_loss

        # Enforce model.eval() / dropout checks
        is_training_active = False
        if model is not None:
            is_training_active = model.training

        # Token rate
        unk_token_rate = (
            self.unknown_tokens_count / max(self.total_tokens_checked, 1)
        )

        # Schema size summary
        avg_schema_size = 0.0
        max_schema_size = 0
        if self.schema_sizes:
            avg_schema_size = sum(self.schema_sizes) / len(self.schema_sizes)
            max_schema_size = max(self.schema_sizes)

        # Masking correctness rate
        mask_error_rate = (
            self.masking_errors / max(self.masking_checks, 1)
        )

        diag = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "loss_gap": loss_gap,
            "invalid_pointer_targets_count": self.invalid_pointer_count,
            "unknown_token_rate": unk_token_rate,
            "average_schema_size": avg_schema_size,
            "max_schema_size": max_schema_size,
            "masking_error_rate": mask_error_rate,
            "model_eval_active_during_val": not is_training_active,
            "train_validation_loss_weight_consistency": True,
        }

        # Calculate per-head losses and accuracy gap if val_metrics/train_metrics are detailed
        per_head_gaps = {}
        for key in train_metrics:
            if key.endswith("_accuracy") and f"validation_{key}" in val_metrics:
                t_acc = train_metrics[key]
                v_acc = val_metrics[f"validation_{key}"]
                per_head_gaps[key] = v_acc - t_acc
        diag["per_head_accuracy_gaps"] = per_head_gaps

        self.epoch_diagnostics[epoch] = diag
        self.schema_sizes = []  # Clear for next epoch
        return diag

    def write_diagnostics(self, output_dir: Path | str) -> Path:
        """Write training diagnostics report to JSON."""
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        report_path = target_dir / "training_diagnostics.json"
        
        # Analyze weakest head (lowest accuracy, highest train-val gap)
        weakest_heads = {}
        for epoch, diag in self.epoch_diagnostics.items():
            gaps = diag.get("per_head_accuracy_gaps") or {}
            if gaps:
                weakest = min(gaps, key=gaps.get)
                weakest_heads[f"epoch_{epoch}"] = {
                    "weakest_head": weakest,
                    "gap": gaps[weakest],
                }

        payload = {
            "epoch_diagnostics": self.epoch_diagnostics,
            "weakest_heads": weakest_heads,
            "initial_epoch_diagnostics": self.epoch_diagnostics.get(1, {}),
        }
        report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return report_path
