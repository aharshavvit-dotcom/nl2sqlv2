"""Optimized Neural QueryIR Model training CLI.

Full-featured training loop with configurable optimizer, scheduler,
early stopping, gradient clipping, checkpoint management, and diagnostics.

Usage:
    python training/train_neural_ir_optimized.py \\
      --config configs/neural_training_default.yaml \\
      --train data/processed/generic_ir_train.jsonl \\
      --validation data/processed/generic_ir_validation.jsonl \\
      --output-dir artifacts/neural_ir_model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_optimization.training_config import (
    NeuralTrainingConfig,
    load_training_config,
    merge_cli_overrides,
    save_effective_config,
)
from neural_optimization.optimizer_factory import build_optimizer
from neural_optimization.scheduler_factory import build_scheduler
from neural_optimization.checkpoint_manager import CheckpointManager, _state_dict_sha256
from neural_optimization.early_stopping import EarlyStopping
from neural_optimization.loss_weighter import MultiTaskLossWeighter
from neural_optimization.training_diagnostics import TrainingDiagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized Neural QueryIR Model training")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--train", type=str, default=None, help="Training JSONL path")
    parser.add_argument("--validation", type=str, default=None, help="Validation JSONL path")
    parser.add_argument("--hard-negatives", type=str, default=None, help="Hard-negatives JSONL path")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--activation", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_training_config(args.config)
    else:
        config = NeuralTrainingConfig()

    # Apply CLI overrides
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "activation": args.activation,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "max_examples": args.max_examples,
        "train": args.train,
        "validation": args.validation,
        "hard_negatives": args.hard_negatives,
    }
    config = merge_cli_overrides(config, overrides)

    output_dir = Path(config.output.get("output_dir", "artifacts/neural_ir_model"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save effective config
    if config.output.get("save_effective_config", True):
        save_effective_config(config, output_dir / "effective_config.yaml")

    report = run_optimized_training(config, output_dir)
    print(json.dumps(report, indent=2, default=str))


def run_optimized_training(
    config: NeuralTrainingConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute the full optimized training loop.

    Returns a metrics dict suitable for experiment comparison.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.output.get("save_effective_config", True):
        save_effective_config(config, output_dir / "effective_config.yaml")
    # Seed
    seed = int(config.training.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Resolve data paths
    train_path = _resolve_path(config.data.get("train_path", "data/processed/generic_ir_train.jsonl"))
    val_path = _resolve_path(config.data.get("validation_path", "data/processed/generic_ir_validation.jsonl"))
    hard_neg_path_str = config.data.get("hard_negatives_path", "")
    hard_neg_path = _resolve_path(hard_neg_path_str) if hard_neg_path_str else None
    max_examples = int(config.data.get("max_examples", 0)) or None
    legacy_mode = bool(config.data.get("legacy_mode", False))
    sample_mode = bool(config.data.get("sample_mode", False))
    smoke_mode = bool(config.training.get("smoke", False) or (max_examples is not None and max_examples <= 200))

    if not train_path.exists():
        print(f"Error: Training file not found: {train_path}")
        return {"error": f"Training file not found: {train_path}"}
    if _looks_like_legacy_or_sample_path(train_path) and not (legacy_mode or sample_mode):
        return {
            "error": (
                "Refusing to train from legacy/sample data path without explicit legacy_mode/sample_mode: "
                f"{train_path}"
            )
        }

    # Build datasets and model using existing neural_ir infrastructure
    from neural_ir.ir_dataset import IRTrainingDataset, collate_ir_batch, load_jsonl
    from neural_ir.ir_label_encoder import IRLabelEncoder
    from neural_ir.attention_model import SchemaAwareOptionAIRModel
    from neural_ir.trainer import HEAD_TO_LABEL, HEAD_TO_MASK, MODEL_INPUT_KEYS
    from neural_ir.vocab import Vocabulary
    from neural_optimization.loss_registry import masked_cross_entropy_fn, margin_ranking_loss

    train_rows_for_stats = load_jsonl(train_path)
    val_rows_for_stats = load_jsonl(val_path) if val_path.exists() else []
    hard_negative_weight = float(config.loss.get("hard_negative", 0) or 0)
    hard_negative_rows, hard_negative_format_issues = _load_hard_negative_rows(hard_neg_path)
    train_example_ids = {str(row.get("example_id") or "") for row in train_rows_for_stats if row.get("example_id")}
    hard_negative_examples_matched = sum(
        1 for row in hard_negative_rows if str(row.get("example_id") or "") in train_example_ids
    )
    hard_negative_loss_active = hard_negative_weight > 0 and bool(hard_negative_rows)
    hard_negative_warning: str | None = None
    if hard_negative_weight > 0 and not hard_negative_rows:
        message = (
            f"Hard-negative weight is {hard_negative_weight}, but no valid hard negatives were loaded "
            f"from {hard_neg_path or '<none>'}."
        )
        if smoke_mode:
            hard_negative_warning = message + " Disabling hard-negative loss for smoke training."
            hard_negative_loss_active = False
        else:
            return {
                "error": message,
                "hard_negative_file": str(hard_neg_path) if hard_neg_path else "",
                "hard_negative_examples_loaded": 0,
                "hard_negative_format_issues": hard_negative_format_issues,
            }
    if hard_negative_loss_active and hard_negative_examples_matched <= 0:
        message = (
            f"Loaded {len(hard_negative_rows)} hard negatives, but none match training example IDs "
            f"from {train_path}."
        )
        if smoke_mode:
            hard_negative_warning = (hard_negative_warning + " " if hard_negative_warning else "") + (
                message + " Disabling hard-negative loss for smoke training."
            )
            hard_negative_loss_active = False
        else:
            return {
                "error": message,
                "hard_negative_file": str(hard_neg_path) if hard_neg_path else "",
                "hard_negative_examples_loaded": len(hard_negative_rows),
                "hard_negative_examples_matched": hard_negative_examples_matched,
                "hard_negative_format_issues": hard_negative_format_issues,
            }

    label_encoder = IRLabelEncoder()
    vocab = Vocabulary()
    vocab.build(_token_sequences(train_rows_for_stats, max_examples=max_examples))

    # Build model config from our training config
    model_config = {
        **config.model,
        "learning_rate": config.optimizer.get("learning_rate", 0.0007),
        "weight_decay": config.optimizer.get("weight_decay", 0.00001),
        "use_hard_negative_loss": hard_negative_loss_active,
        "hard_negative_loss_weight": hard_negative_weight,
        "batch_size": config.training.get("batch_size", 8),
        "epochs": config.training.get("epochs", 10),
    }

    train_dataset = IRTrainingDataset(
        str(train_path),
        vocab=vocab,
        label_encoder=label_encoder,
        max_question_len=int(model_config.get("max_question_len", 64)),
        max_schema_len=int(model_config.get("max_schema_len", 320)),
        max_candidate_tokens=int(model_config.get("max_candidate_tokens", 16)),
        max_tables=int(model_config.get("max_tables", 64)),
        max_columns=int(model_config.get("max_columns", 256)),
        max_examples=max_examples,
        hard_negative_rows=hard_negative_rows if hard_negative_loss_active else None,
    )
    if len(train_dataset) == 0:
        return {"error": "No training examples loaded"}

    curriculum_cfg = config.training.get("curriculum") or {}
    curriculum_enabled = bool(curriculum_cfg.get("enabled", False))
    curriculum_distribution: dict[str, int] = {}
    if curriculum_enabled:
        from dataset_training.curriculum_builder import CurriculumBuilder

        train_dataset.examples, curriculum_distribution = CurriculumBuilder().order_examples(
            train_dataset.examples,
            curriculum_cfg.get("phases") or [],
            mode=str(curriculum_cfg.get("mode", "ordered_dataset")),
            allow_ordered_dataset_fallback=bool(curriculum_cfg.get("allow_ordered_dataset_fallback", False)),
        )

    val_dataset = IRTrainingDataset(
        str(val_path),
        vocab=vocab,
        label_encoder=train_dataset.label_encoder,
        max_question_len=int(model_config.get("max_question_len", 64)),
        max_schema_len=int(model_config.get("max_schema_len", 320)),
        max_candidate_tokens=int(model_config.get("max_candidate_tokens", 16)),
        max_tables=int(model_config.get("max_tables", 64)),
        max_columns=int(model_config.get("max_columns", 256)),
        max_examples=max_examples,
    ) if val_path.exists() else None

    batch_size = int(config.training.get("batch_size", 8))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=not curriculum_enabled,
        collate_fn=collate_ir_batch,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_ir_batch,
    ) if val_dataset and len(val_dataset) > 0 else None

    # Build model
    vocab_size = len(vocab)
    label_sizes = train_dataset.label_encoder.label_sizes
    model = SchemaAwareOptionAIRModel(model_config, vocab_size, label_sizes)
    device = _resolve_torch_device(str(config.training.get("device", "auto")))
    model.to(device)

    # Build optimizer & scheduler
    optimizer = build_optimizer(model, config.optimizer)
    epochs = int(config.training.get("epochs", 10))
    total_steps = epochs * len(train_loader)
    scheduler = build_scheduler(optimizer, config.scheduler, total_steps=total_steps)

    # Build loss weighter
    loss_weighter = MultiTaskLossWeighter(config.loss)

    # Loss head name mapping (head logit key → loss weight key)
    _HEAD_LOSS_NAME = {
        "intent_logits": "intent",
        "base_table_logits": "base_table",
        "metric_aggregation_logits": "metric_aggregation",
        "metric_column_logits": "metric_column",
        "metric_expression_type_logits": "metric_expression_type",
        "dimension_column_logits": "dimension_column",
        "date_column_logits": "date_column",
        "date_grain_logits": "date_grain",
        "date_filter_type_logits": "date_filter_type",
        "filter_column_logits": "filter_column",
        "filter_operator_logits": "filter_operator",
        "order_direction_logits": "order_direction",
        "limit_bucket_logits": "limit_bucket",
    }

    # Checkpoint manager & early stopping
    best_metric = str(config.training.get("save_best_metric", "loss"))
    best_mode = str(config.training.get("save_best_mode", "min"))
    ckpt_manager = CheckpointManager(output_dir, metric_name=best_metric, mode=best_mode)
    early_stopper = EarlyStopping(
        patience=int(config.training.get("early_stopping_patience", 2)),
        metric_name=best_metric,
        mode=best_mode,
    )

    # Load baseline performance comparison
    baseline_score = None
    prev_meta_path = Path(output_dir) / "checkpoint_metadata.json"
    if prev_meta_path.exists():
        try:
            prev_meta = json.loads(prev_meta_path.read_text(encoding="utf-8"))
            baseline_score = prev_meta.get("best_metric_value")
        except Exception:
            pass

    # Run data leakage audit
    from dataset_training.leakage_checker import DatasetLeakageChecker
    try:
        leakage_checker = DatasetLeakageChecker()
        leakage_res = leakage_checker.check_leakage(train_path, val_path)
        leakage_summary = {
            "ok": leakage_res.ok,
            "total_issues": len(leakage_res.issues),
            "issues": [str(issue) for issue in leakage_res.issues],
        }
    except Exception as exc:
        leakage_summary = {
            "ok": False,
            "total_issues": 1,
            "issues": [f"Leakage audit execution failed: {exc}"],
        }

    # Diagnostics
    diagnostics = TrainingDiagnostics(output_dir)
    diagnostics.set_config(config.to_dict())
    diagnostics.set_baseline_score(baseline_score)
    diagnostics.set_leakage_summary(leakage_summary)
    diagnostics.observe_dataset_item(train_dataset[0])
    diagnostics.start_training()

    grad_clip = float(config.training.get("gradient_clipping", 1.0))

    print(f"Starting optimized Neural QueryIR training for {epochs} epochs")
    print(f"  Optimizer: {config.optimizer.get('name')} | Activation: {config.model.get('activation')}")
    print(f"  Training: {len(train_dataset)} examples | Validation: {len(val_dataset) if val_dataset else 0}")
    print(f"  Batch size: {batch_size} | Gradient clipping: {grad_clip}")
    print(f"  Checkpoint monitor: {best_metric}")
    print(f"  Checkpoint mode: {best_mode}")
    print(f"  Early stopping patience: {early_stopper.patience}")
    if best_metric == "loss" and best_mode == "min":
        print("  Best checkpoint selected by lowest validation loss")
    if hard_negative_warning:
        print(f"  Warning: {hard_negative_warning}")
    print(
        "  Hard negatives: "
        f"{len(hard_negative_rows)} loaded | active={hard_negative_loss_active} | weight={hard_negative_weight}"
    )

    history: list[dict[str, Any]] = []
    last_loss_by_head: dict[str, float] = {}
    hard_negative_batches_used = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch:02d}/{epochs:02d} ---")

        if curriculum_enabled:
            from dataset_training.curriculum_builder import CurriculumBuilder
            # Use deterministic epoch-based seed
            epoch_seed = int(config.training.get("seed", 42)) + epoch
            train_dataset.examples = CurriculumBuilder().shuffle_within_buckets(
                train_dataset.examples,
                seed=epoch_seed,
            )

        # ── Train ────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        total_items = 0
        epoch_head_losses: dict[str, float] = {}
        epoch_correct: dict[str, int] = {}
        epoch_total: dict[str, int] = {}

        for batch_idx, batch in enumerate(train_loader, 1):
            batch = _to_device(batch, device)
            optimizer.zero_grad()
            outputs = _model_outputs(model, batch, MODEL_INPUT_KEYS)
            diagnostics.observe_step(batch, outputs)

            # Per-head losses
            head_losses: dict[str, torch.Tensor] = {}
            for head, label_key in HEAD_TO_LABEL.items():
                target = batch["labels"][label_key]
                if not target.ne(-1).any():
                    continue
                mask = batch.get(HEAD_TO_MASK.get(head, ""))
                loss_name = _HEAD_LOSS_NAME.get(head, head.replace("_logits", ""))
                head_losses[loss_name] = masked_cross_entropy_fn(outputs[head], target, mask=mask, ignore_index=-1)

            # Hard-negative loss
            if hard_negative_loss_active:
                hn_loss = _hard_negative_loss(outputs, batch["labels"], HEAD_TO_LABEL, margin_ranking_loss)
                if hn_loss is not None:
                    head_losses["hard_negative"] = hn_loss
                    hard_negative_batches_used += 1

            # Span loss
            if "span" in batch["labels"] and "span_logits" in outputs:
                span_logits = outputs["span_logits"]
                span_target = batch["labels"]["span"]
                span_loss = torch.nn.functional.cross_entropy(
                    span_logits.view(-1, 2),
                    span_target.view(-1),
                    ignore_index=-1
                )
                head_losses["span"] = span_loss

            combined = loss_weighter.combine(head_losses)
            loss = combined["total_loss"]
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            bs = int(batch["question_ids"].size(0))
            total_loss += float(loss.item()) * bs
            total_items += bs

            # Track per-head losses
            for k, v in combined["raw_losses"].items():
                epoch_head_losses[k] = epoch_head_losses.get(k, 0.0) + v

            # Track accuracies
            for head, label_key in HEAD_TO_LABEL.items():
                target = batch["labels"][label_key]
                valid = target.ne(-1)
                if not valid.any():
                    continue
                pred = outputs[head].argmax(dim=-1)
                c = int(pred.eq(target).logical_and(valid).sum().item())
                t = int(valid.sum().item())
                epoch_correct[label_key] = epoch_correct.get(label_key, 0) + c
                epoch_total[label_key] = epoch_total.get(label_key, 0) + t

            if batch_idx % max(1, len(train_loader) // 5) == 0 or batch_idx == len(train_loader):
                print(f"  [Train] Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        train_loss = total_loss / max(total_items, 1)
        train_metrics = {"loss": train_loss}
        for lk in epoch_total:
            train_metrics[lk.replace("_label", "_accuracy").replace("_index", "_accuracy")] = (
                epoch_correct.get(lk, 0) / max(epoch_total.get(lk, 0), 1)
            )
        keys = list(epoch_total)
        train_metrics["overall_slot_accuracy"] = (
            sum(epoch_correct.get(k, 0) / max(epoch_total.get(k, 0), 1) for k in keys) / max(len(keys), 1)
        )
        train_metrics.update(_semantic_checkpoint_metrics(epoch_correct, epoch_total, config))

        # ── Validate ─────────────────────────────────────────────
        val_metrics: dict[str, float] = {}
        if val_loader:
            model.eval()
            val_loss_total = 0.0
            val_items = 0
            val_correct: dict[str, int] = {}
            val_total: dict[str, int] = {}
            with torch.no_grad():
                for batch in val_loader:
                    batch = _to_device(batch, device)
                    outputs = _model_outputs(model, batch, MODEL_INPUT_KEYS)
                    diagnostics.observe_step(batch, outputs)
                    head_losses_v: dict[str, torch.Tensor] = {}
                    for head, label_key in HEAD_TO_LABEL.items():
                        target = batch["labels"][label_key]
                        if not target.ne(-1).any():
                            continue
                        mask = batch.get(HEAD_TO_MASK.get(head, ""))
                        loss_name = _HEAD_LOSS_NAME.get(head, head.replace("_logits", ""))
                        head_losses_v[loss_name] = masked_cross_entropy_fn(outputs[head], target, mask=mask, ignore_index=-1)
                    
                    if "span" in batch["labels"] and "span_logits" in outputs:
                        span_logits_v = outputs["span_logits"]
                        span_target_v = batch["labels"]["span"]
                        span_loss_v = torch.nn.functional.cross_entropy(
                            span_logits_v.view(-1, 2),
                            span_target_v.view(-1),
                            ignore_index=-1
                        )
                        head_losses_v["span"] = span_loss_v

                    combined_v = loss_weighter.combine(head_losses_v)
                    bs = int(batch["question_ids"].size(0))
                    val_loss_total += float(combined_v["total_loss"].item()) * bs
                    val_items += bs
                    for head, label_key in HEAD_TO_LABEL.items():
                        target = batch["labels"][label_key]
                        valid = target.ne(-1)
                        if not valid.any():
                            continue
                        pred = outputs[head].argmax(dim=-1)
                        val_correct[label_key] = val_correct.get(label_key, 0) + int(pred.eq(target).logical_and(valid).sum().item())
                        val_total[label_key] = val_total.get(label_key, 0) + int(valid.sum().item())

                    if "span" in batch["labels"] and "span_logits" in outputs:
                        span_pred_v = outputs["span_logits"].argmax(dim=-1)
                        span_target_v = batch["labels"]["span"]
                        valid_span_v = span_target_v.ne(-1)
                        if valid_span_v.any():
                            c_span = int(span_pred_v.eq(span_target_v).logical_and(valid_span_v).sum().item())
                            t_span = int(valid_span_v.sum().item())
                            val_correct["span"] = val_correct.get("span", 0) + c_span
                            val_total["span"] = val_total.get("span", 0) + t_span

            val_metrics["loss"] = val_loss_total / max(val_items, 1)
            for lk in val_total:
                val_metrics[lk.replace("_label", "_accuracy").replace("_index", "_accuracy")] = (
                    val_correct.get(lk, 0) / max(val_total.get(lk, 0), 1)
                )
            vkeys = list(val_total)
            val_metrics["overall_slot_accuracy"] = (
                sum(val_correct.get(k, 0) / max(val_total.get(k, 0), 1) for k in vkeys) / max(len(vkeys), 1)
            )
            composite_parts = [
                val_metrics.get("intent_accuracy", 0.0),
                val_metrics.get("base_table_accuracy", 0.0),
                val_metrics.get("overall_slot_accuracy", 0.0),
            ]
            val_metrics["validation_composite_score"] = sum(composite_parts) / len(composite_parts)
            val_metrics.update(_semantic_checkpoint_metrics(val_correct, val_total, config))

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:02d} in {epoch_time:.1f}s - "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics.get('loss', 0):.4f} | "
              f"Slot Acc: {val_metrics.get('overall_slot_accuracy', train_metrics.get('overall_slot_accuracy', 0)):.4f}")

        # Diagnostics
        diagnostics.record_epoch(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=current_lr,
            epoch_time=epoch_time,
            loss_by_head=epoch_head_losses,
        )
        last_loss_by_head = dict(epoch_head_losses)

        # Checkpoint
        check_metrics = val_metrics if val_metrics else train_metrics
        saved = ckpt_manager.maybe_save_best(model, optimizer, epoch, check_metrics, config.to_dict())
        if saved:
            print("  New best checkpoint saved")

            # Also save as model.pt for predictor compatibility
            torch.save(model.state_dict(), output_dir / "model.pt")
            _record_runtime_export_identity(output_dir, model.state_dict())

        ckpt_manager.save_last(model, optimizer, epoch, check_metrics, config.to_dict())

        # Scheduler step
        if scheduler is not None:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(check_metrics.get("loss", train_loss))
            else:
                scheduler.step()

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        # Early stopping
        if early_stopper.step(check_metrics):
            print(f"  Early stopping at epoch {epoch}")
            break

    # Save final model.pt (best model)
    best_ckpt = ckpt_manager.load_best()
    if best_ckpt:
        model.load_state_dict(best_ckpt["model_state_dict"])
        torch.save(model.state_dict(), output_dir / "model.pt")
        _record_runtime_export_identity(output_dir, model.state_dict())

    # Save label encoder artifacts
    train_dataset.label_encoder.save(str(output_dir / "label_maps.json"))
    vocab.save(str(output_dir / "vocab.json"))

    # Save diagnostics
    if config.output.get("save_diagnostics", True):
        diagnostics.save(output_dir)

    # Save training metrics
    final_metrics = val_metrics if val_metrics else train_metrics
    best_epoch = diagnostics.best_epoch()
    checkpoint_path = output_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = output_dir / "model.pt"
    from datetime import datetime, timezone
    pipeline_run_id = str(config.output.get("pipeline_run_id", ""))
    report = {
        "pipeline_run_id": pipeline_run_id,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "best_epoch": best_epoch.get("epoch"),
        "best_metric": best_epoch.get("validation_total_loss") if best_metric == "loss" else best_epoch.get(best_metric),
        "best_overall_slot_accuracy": best_epoch.get("overall_slot_accuracy"),
        "final_train_loss": train_loss,
        "final_val_loss": val_metrics.get("loss"),
        "optimizer": config.optimizer.get("name"),
        "optimizer_name": config.optimizer.get("name"),
        "activation": config.model.get("activation"),
        "activation_name": config.model.get("activation"),
        "gradient_clipping_value": grad_clip,
        "loss_by_head": last_loss_by_head,
        "train_path": str(train_path),
        "validation_path": str(val_path),
        "train_examples_count": len(train_dataset),
        "validation_examples_count": len(val_dataset) if val_dataset else 0,
        "train_by_dataset": _dataset_distribution(train_dataset.examples),
        "validation_by_dataset": _dataset_distribution(val_dataset.examples if val_dataset else []),
        "curriculum_enabled": curriculum_enabled,
        "curriculum_distribution": curriculum_distribution,
        "curriculum_mode": curriculum_distribution.get("_curriculum_mode", "ordered_dataset") if curriculum_enabled else "disabled",
        "legacy_mode": legacy_mode,
        "sample_mode": sample_mode,
        "hard_negative_file": str(hard_neg_path) if hard_neg_path else "",
        "hard_negative_examples_loaded": len(hard_negative_rows),
        "hard_negative_examples_matched": hard_negative_examples_matched,
        "hard_negative_format_issues": hard_negative_format_issues,
        "hard_negative_batches_used": hard_negative_batches_used,
        "hard_negative_loss_active": hard_negative_loss_active,
        "hard_negative_weight": hard_negative_weight,
        "model_architecture": config.model.get("architecture", "schema_aware_queryir"),
        "ffn_heads_enabled": bool(config.model.get("feed_forward_heads", False)),
        "scheduler": config.scheduler.get("name"),
        "best_checkpoint_metric": best_metric,
        "checkpoint_monitor": best_metric,
        "checkpoint_mode": best_mode,
        "early_stopping_patience": early_stopper.patience,
        "effective_epochs": epochs,
        "effective_batch_size": batch_size,
        "weight_decay": float(config.optimizer.get("weight_decay", 0.0001)),
        "pointer_head_weight_decay": float(config.optimizer.get("pointer_head_weight_decay", 0.001)),
        "pointer_dropout": float(config.model.get("pointer_dropout", 0.30)),
        "device": str(device),
        "precision": str(config.training.get("precision", "float32")),
        "determinism_mode": str(config.training.get("determinism_mode", "seeded")),
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "effective_config_hash": _effective_config_hash(config),
        "best_val_loss": best_epoch.get("validation_total_loss"),
        "current_val_loss": val_metrics.get("loss"),
        "val_train_loss_ratio": (
            val_metrics.get("loss") / train_loss
            if val_metrics.get("loss") is not None and train_loss > 0
            else None
        ),
        "overfitting_warning": bool(
            val_metrics.get("loss") is not None
            and train_loss > 0
            and val_metrics.get("loss") / train_loss > 1.5
        ),
        "loss_weights": dict(config.loss),
        "validation_gold_score_available": "validation_gold_score" in final_metrics,
        "validation_gold_score_unavailable_reason": (
            None if "validation_gold_score" in final_metrics else "gold comparator not available inside neural training loop"
        ),
        "validation_composite_score": final_metrics.get("validation_composite_score"),
        "checkpoint_path": str(checkpoint_path),
        "epochs_ran": len(history),
        "early_stopped": early_stopper.counter >= early_stopper.patience,
        **{k: v for k, v in final_metrics.items()},
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8",
    )

    # Save model config for predictor compatibility
    _save_model_config(output_dir, model_config, train_dataset.label_encoder)
    _save_training_manifest(output_dir, report, config)

    return report


def _effective_config_hash(config: NeuralTrainingConfig) -> str:
    payload = json.dumps({
        "epochs": config.training.get("epochs"),
        "batch_size": config.training.get("batch_size"),
        "save_best_metric": config.training.get("save_best_metric"),
        "save_best_mode": config.training.get("save_best_mode"),
        "early_stopping_patience": config.training.get("early_stopping_patience"),
        "weight_decay": config.optimizer.get("weight_decay"),
        "pointer_head_weight_decay": config.optimizer.get("pointer_head_weight_decay"),
        "pointer_dropout": config.model.get("pointer_dropout"),
    }, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _save_training_manifest(output_dir: Path, report: dict[str, Any], config: NeuralTrainingConfig) -> None:
    manifest = {
        "artifact_type": "neural_queryir_model",
        "pipeline_run_id": report.get("pipeline_run_id", ""),
        "generated_at": report.get("generated_at", ""),
        "source_train_file": report.get("train_path"),
        "source_validation_file": report.get("validation_path"),
        "train_examples_count": report.get("train_examples_count", 0),
        "validation_examples_count": report.get("validation_examples_count", 0),
        "train_by_dataset": report.get("train_by_dataset", {}),
        "validation_by_dataset": report.get("validation_by_dataset", {}),
        "hard_negative_file": report.get("hard_negative_file", ""),
        "hard_negative_examples_loaded": report.get("hard_negative_examples_loaded", 0),
        "hard_negative_examples_matched": report.get("hard_negative_examples_matched", 0),
        "hard_negative_batches_used": report.get("hard_negative_batches_used", 0),
        "hard_negative_loss_active": report.get("hard_negative_loss_active", False),
        "optimizer_name": report.get("optimizer_name"),
        "activation_name": report.get("activation_name"),
        "gradient_clipping_value": report.get("gradient_clipping_value"),
        "checkpoint_path": report.get("checkpoint_path"),
        "config": config.to_dict(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _looks_like_legacy_or_sample_path(path: Path) -> bool:
    normalized = str(path).replace("\\", "/").lower()
    return (
        normalized.endswith("ir_training_examples.jsonl")
        or "/training_data/examples.jsonl" in normalized
        or "/sample" in normalized
    )


def _load_hard_negative_rows(path: Path | None) -> tuple[list[dict[str, Any]], list[str]]:
    if path is None or not path.exists():
        return [], []
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    issues.append(f"line {line_number}: invalid JSON ({exc})")
                    continue
                example_id = row.get("example_id")
                negative_ir = row.get("negative_query_ir") or row.get("query_ir")
                if not example_id:
                    issues.append(f"line {line_number}: missing example_id")
                    continue
                if not isinstance(negative_ir, dict):
                    issues.append(f"line {line_number}: missing negative_query_ir/query_ir object")
                    continue
                rows.append(row)
    except Exception as exc:
        issues.append(f"failed to read hard-negative file: {exc}")
        return [], issues
    return rows, issues


def _dataset_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("dataset_name") or row.get("dataset") or "unknown") for row in rows))


def _token_sequences(rows: list[dict[str, Any]], max_examples: int | None = None) -> list[list[str]]:
    from neural_ir.tokenizer import tokenize

    limited = rows[:max_examples] if max_examples is not None and max_examples > 0 else rows
    sequences: list[list[str]] = []
    for row in limited:
        question = str(row.get("question") or "")
        schema_text = str(row.get("serialized_schema") or "")
        sequences.append(tokenize(question))
        if schema_text:
            sequences.append(tokenize(schema_text))
    return sequences


def _save_model_config(output_dir: Path, model_config: dict, label_encoder: Any) -> None:
    """Save config.yaml for predictor compatibility."""
    import yaml
    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml.dump(model_config, default_flow_style=False), encoding="utf-8")


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            {ik: iv.to(device) for ik, iv in value.items()}
            if key == "labels"
            else value.to(device) if torch.is_tensor(value) else value
        )
        for key, value in batch.items()
    }


def _model_outputs(model, batch: dict[str, Any], keys: list[str]) -> dict[str, torch.Tensor]:
    kwargs = {key: batch[key] for key in keys if key in batch}
    return model(**kwargs)


def _resolve_torch_device(requested: str) -> torch.device:
    mode = requested.strip().lower()
    if mode == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if mode == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured training.device=cuda but CUDA is not available.")
    if mode == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("Configured training.device=mps but MPS is not available.")
    if mode not in {"cpu", "cuda", "mps"}:
        raise ValueError("training.device must be one of: auto, cpu, cuda, mps")
    return torch.device(mode)


def _hard_negative_loss(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    head_to_label: dict[str, str],
    margin_fn,
) -> torch.Tensor | None:
    """Compute margin ranking loss from hard-negative labels."""
    losses = []
    for label_key, head in [
        ("negative_base_table_index", "base_table_logits"),
        ("negative_metric_column_index", "metric_column_logits"),
        ("negative_dimension_column_index", "dimension_column_logits"),
        ("negative_date_column_index", "date_column_logits"),
        ("negative_filter_column_index", "filter_column_logits"),
    ]:
        gold_label = head_to_label[head]
        if label_key not in labels or gold_label not in labels:
            continue
        gold_index = labels[gold_label]
        negative_index = labels[label_key]
        valid = gold_index.ge(0) & negative_index.ge(0) & (gold_index != negative_index)
        if not valid.any():
            continue
        logits = outputs[head]
        gold_scores = logits.gather(1, gold_index.clamp_min(0).unsqueeze(1)).squeeze(1)[valid]
        negative_scores = logits.gather(1, negative_index.clamp_min(0).unsqueeze(1)).squeeze(1)[valid]
        losses.append(margin_fn(gold_scores, negative_scores, margin=0.2))
    if not losses:
        return None
    return torch.stack(losses).sum()


def _semantic_checkpoint_metrics(
    correct: dict[str, int],
    total: dict[str, int],
    config: NeuralTrainingConfig,
) -> dict[str, Any]:
    weights = {
        "intent_macro_f1": 0.20,
        "projection_exact_match": 0.15,
        "filter_column_accuracy": 0.20,
        "filter_value_accuracy": 0.20,
        "dimension_column_accuracy": 0.10,
        "semantic_pass_rate": 0.15,
    }
    weights.update(config.training.get("semantic_score_weights") or {})
    minimum_support = int(config.training.get("semantic_score_min_support", 1) or 1)

    label_map = {
        "intent_macro_f1": "intent_label",
        "projection_exact_match": "metric_column_index",
        "filter_column_accuracy": "filter_column_index",
        "filter_value_accuracy": "span",
        "dimension_column_accuracy": "dimension_column_index",
    }

    metric_values: dict[str, float | None] = {}
    metric_supports: dict[str, int] = {}
    missing_metrics: list[str] = []
    for metric_name, label_key in label_map.items():
        support = int(total.get(label_key, 0))
        metric_supports[metric_name] = support
        if support < minimum_support:
            metric_values[metric_name] = None
            missing_metrics.append(metric_name)
            continue
        metric_values[metric_name] = float(correct.get(label_key, 0)) / max(support, 1)

    available = [value for value in metric_values.values() if value is not None]
    if available:
        metric_values["semantic_pass_rate"] = min(float(value) for value in available)
        metric_supports["semantic_pass_rate"] = min(
            support for name, support in metric_supports.items()
            if metric_values.get(name) is not None
        )
    else:
        metric_values["semantic_pass_rate"] = None
        metric_supports["semantic_pass_rate"] = 0
        missing_metrics.append("semantic_pass_rate")

    weighted_value = 0.0
    available_weight = 0.0
    for metric_name, weight in weights.items():
        value = metric_values.get(metric_name)
        if value is None:
            continue
        weighted_value += float(weight) * float(value)
        available_weight += float(weight)
    semantic_score = weighted_value / available_weight if available_weight else 0.0
    return {
        "support_weighted_semantic_score": semantic_score,
        "semantic_checkpoint_score": semantic_score,
        "semantic_checkpoint_score_definition_version": "2.0",
        "semantic_checkpoint_score_weights": weights,
        "semantic_checkpoint_metric_values": metric_values,
        "semantic_checkpoint_metric_supports": metric_supports,
        "semantic_checkpoint_missing_metrics": sorted(set(missing_metrics)),
        "semantic_checkpoint_minimum_support": minimum_support,
        "semantic_checkpoint_score_valid_for_production": not missing_metrics,
    }


def _record_runtime_export_identity(output_dir: Path, runtime_state_dict: dict[str, Any]) -> None:
    metadata_path = output_dir / "checkpoint_metadata.json"
    if not metadata_path.exists():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except ValueError:
        return
    runtime_hash = _state_dict_sha256(runtime_state_dict)
    selected_hash = metadata.get("best_checkpoint_state_dict_sha256")
    metadata.update({
        "selected_checkpoint_file": "best_model.pt",
        "selected_checkpoint_epoch": metadata.get("best_epoch"),
        "selected_checkpoint_state_dict_sha256": selected_hash,
        "runtime_export_file": "model.pt",
        "runtime_export_state_dict_sha256": runtime_hash,
        "runtime_export_equivalent_to_selected_checkpoint": (
            bool(selected_hash) and selected_hash == runtime_hash
        ),
    })
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
