from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .loss_utils import accuracy_from_logits, margin_ranking_slot_loss, masked_cross_entropy
from neural_optimization.optimizer_factory import build_optimizer


HEAD_TO_LABEL = {
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
    "complexity_logits": "complexity_label",
}

HEAD_TO_MASK = {
    "base_table_logits": "table_candidate_mask",
    "metric_column_logits": "metric_column_mask",
    "dimension_column_logits": "dimension_column_mask",
    "date_column_logits": "date_column_mask",
    "filter_column_logits": "filter_column_mask",
}

MODEL_INPUT_KEYS = [
    "question_ids",
    "schema_ids",
    "question_mask",
    "schema_mask",
    "candidate_token_ids",
    "table_candidate_token_ids",
    "column_candidate_token_ids",
    "table_candidate_mask",
    "column_candidate_mask",
    "metric_column_mask",
    "dimension_column_mask",
    "date_column_mask",
    "filter_column_mask",
    "schema_link_scores",
    "relation_type_ids",
    "schema_relation_type_ids",
    "candidate_relation_type_ids",
]


class OptionAIRTrainer:
    def __init__(self, model, config, diagnostics=None):
        self.model = model
        self.config = config
        
        # Support device auto-configuration
        device_name = config.get("runtime", {}).get("device", "auto")
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)
            
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = build_optimizer(self.model, {
            "name": config.get("optimizer", "adam"),
            "learning_rate": float(config.get("learning_rate", 0.001)),
            "weight_decay": float(config.get("weight_decay", 0.0001)),
            "pointer_head_weight_decay": float(config.get("pointer_head_weight_decay", 0.001)),
        })
        self.diagnostics = diagnostics

    def train(self, train_loader, val_loader, output_dir=None, label_encoder=None) -> dict[str, Any]:
        import time
        if output_dir is not None and not isinstance(output_dir, (str, Path)):
            output_dir, label_encoder = label_encoder, output_dir
        output_path = Path(output_dir or "artifacts/work/neural_ir")
        output_path.mkdir(parents=True, exist_ok=True)
        best_loss = float("inf")
        best_metric_value: float | None = None
        best_state = None
        best_epoch = 1
        patience = 0
        history = []
        epochs = int(self.config.get("epochs", 10))
        batch_size = int(self.config.get("batch_size", 8))
        patience_limit = int(self.config.get("early_stopping_patience", 2))
        save_best_metric = str(self.config.get("save_best_metric", "loss"))
        save_best_mode = str(self.config.get("save_best_mode", "min"))
        total_start_time = time.time()
        early_stopping_epoch = None
        
        print(f"Starting Neural QueryIR Model training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Checkpoint monitor: {save_best_metric}")
        print(f"Checkpoint mode: {save_best_mode}")
        print(f"Early stopping patience: {patience_limit}")
        print(f"Training set: {len(train_loader)} batches | Validation set: {len(val_loader) if val_loader else 0} batches")
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            print(f"\n--- Epoch {epoch:02d}/{epochs:02d} ---")
            print(f"Started training epoch at {time.strftime('%H:%M:%S')}")
            
            train_metrics = self.train_epoch(train_loader)
            
            if val_loader is not None:
                print("Running evaluation on validation set...")
                val_metrics = self.evaluate_epoch(val_loader)
            else:
                val_metrics = {}
                
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"validation_{k}": v for k, v in val_metrics.items()}}
            history.append(row)
            
            val_loss = float(val_metrics.get("loss", train_metrics.get("loss", 0.0)))
            train_loss = float(train_metrics.get("loss", 0.0))
            train_acc = float(train_metrics.get("overall_slot_accuracy", 0.0))
            val_acc = float(val_metrics.get("overall_slot_accuracy", 0.0)) if val_metrics else 0.0
            
            epoch_duration = time.time() - epoch_start
            print(f"Finished Epoch {epoch:02d}/{epochs:02d} in {epoch_duration:.2f} seconds.")
            print(f"Summary -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
            
            monitored = float((val_metrics or train_metrics).get(save_best_metric, val_loss))
            improved = (
                best_metric_value is None
                or (save_best_mode == "min" and monitored < best_metric_value)
                or (save_best_mode == "max" and monitored > best_metric_value)
            )
            if improved:
                best_loss = val_loss
                best_metric_value = monitored
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                patience = 0
                print(f"Checkpoint saved: {save_best_metric}={monitored:.4f}")
            else:
                patience += 1
                print(f"No {save_best_metric} improvement. Patience: {patience}/{patience_limit}")
                if patience >= patience_limit:
                    early_stopping_epoch = epoch
                    print(f"Early stopping triggered at Epoch {epoch}.")
                    break
                    
        total_duration = time.time() - total_start_time
        print(f"\nAll training epochs completed in {total_duration:.2f} seconds.")
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        torch.save(self.model.state_dict(), output_path / "model.pt")
        metrics = {
            "best_validation_loss": best_loss,
            "best_checkpoint_metric": save_best_metric,
            "checkpoint_mode": save_best_mode,
            "early_stopping_patience": patience_limit,
            "epochs_requested": epochs,
            "epochs_ran": len(history),
            "early_stopping_epoch": early_stopping_epoch,
            "selected_checkpoint_epoch": best_epoch,
            "history": history,
        }
        (output_path / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    def train_epoch(self, loader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_items = 0
        metric_state = _MetricState()
        num_batches = len(loader)
        for i, batch in enumerate(loader, 1):
            batch = _to_device(batch, self.device)
            self.optimizer.zero_grad()
            outputs = _model_outputs(self.model, batch)
            _observe_diagnostics(self.diagnostics, batch, outputs)
            loss = self._loss(outputs, batch["labels"], batch)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item()) * int(batch["question_ids"].size(0))
            total_items += int(batch["question_ids"].size(0))
            metric_state.update(outputs, batch["labels"], batch)
            if i % max(1, num_batches // 5) == 0 or i == num_batches:
                print(f"  [Train Step] Batch {i}/{num_batches} - Batch Loss: {loss.item():.4f}")
        return {"loss": total_loss / max(total_items, 1), **metric_state.compute()}

    def evaluate_epoch(self, loader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_items = 0
        metric_state = _MetricState()
        num_batches = len(loader)
        with torch.no_grad():
            for i, batch in enumerate(loader, 1):
                batch = _to_device(batch, self.device)
                outputs = _model_outputs(self.model, batch)
                _observe_diagnostics(self.diagnostics, batch, outputs)
                loss = self._loss(outputs, batch["labels"], batch)
                total_loss += float(loss.item()) * int(batch["question_ids"].size(0))
                total_items += int(batch["question_ids"].size(0))
                metric_state.update(outputs, batch["labels"], batch)
                if i % max(1, num_batches // 5) == 0 or i == num_batches:
                    print(f"  [Val Step] Batch {i}/{num_batches} - Batch Loss: {loss.item():.4f}")
        return {"loss": total_loss / max(total_items, 1), **metric_state.compute()}

    def _loss(self, outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor], batch: dict[str, Any] | None = None) -> torch.Tensor:
        losses = []
        loss_config = self.config.get("loss") or {}
        head_weight_keys = {
            "intent_logits": "intent",
            "base_table_logits": "base_table",
            "metric_column_logits": "metric_column",
            "metric_aggregation_logits": "metric_aggregation",
            "metric_expression_type_logits": "metric_expression_type",
            "dimension_column_logits": "dimension_column",
            "filter_column_logits": "filter_column",
            "date_column_logits": "date_column",
            "date_grain_logits": "date_grain",
            "date_filter_type_logits": "date_filter_type",
            "filter_operator_logits": "filter_operator",
            "order_direction_logits": "order_direction",
            "limit_bucket_logits": "limit_bucket",
            "complexity_logits": "complexity",
        }
        for head, label in HEAD_TO_LABEL.items():
            if head not in outputs or label not in labels:
                continue
            target = labels[label]
            if not target.ne(-1).any():
                continue
            mask = (batch or {}).get(HEAD_TO_MASK.get(head, "")) if batch else None
            raw_loss = masked_cross_entropy(outputs[head], target, mask=mask, ignore_index=-1)
            weight_key = head_weight_keys.get(head)
            weight = float(loss_config.get(weight_key, 1.0)) if weight_key else 1.0
            losses.append(raw_loss * weight)
        if not losses:
            base_loss = outputs["intent_logits"].sum() * 0.0
        else:
            base_loss = torch.stack(losses).sum()
        if not self.config.get("use_hard_negative_loss"):
            return base_loss
        hard_negative_loss = self._hard_negative_loss(outputs, labels)
        return base_loss + float(self.config.get("hard_negative_loss_weight", 0.3)) * hard_negative_loss

    def _hard_negative_loss(self, outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = []
        for label_key, head in [
            ("negative_base_table_index", "base_table_logits"),
            ("negative_metric_column_index", "metric_column_logits"),
            ("negative_dimension_column_index", "dimension_column_logits"),
            ("negative_date_column_index", "date_column_logits"),
            ("negative_filter_column_index", "filter_column_logits"),
        ]:
            gold_label = HEAD_TO_LABEL[head]
            if label_key not in labels or gold_label not in labels:
                continue
            gold_index = labels[gold_label]
            negative_index = labels[label_key]
            valid = gold_index.ge(0) & negative_index.ge(0)
            if not valid.any():
                continue
            logits = outputs[head]
            gold_scores = logits.gather(1, gold_index.clamp_min(0).unsqueeze(1)).squeeze(1)[valid]
            negative_scores = logits.gather(1, negative_index.clamp_min(0).unsqueeze(1)).squeeze(1)[valid]
            losses.append(margin_ranking_slot_loss(gold_scores, negative_scores, margin=float(self.config.get("hard_negative_margin", 0.2))))
        if not losses:
            return outputs["intent_logits"].sum() * 0.0
        return torch.stack(losses).sum()


class _MetricState:
    def __init__(self) -> None:
        self.correct: dict[str, int] = {}
        self.total: dict[str, int] = {}

    def update(self, outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor], batch: dict[str, Any] | None = None) -> None:
        for head, label_key in HEAD_TO_LABEL.items():
            if head not in outputs or label_key not in labels:
                continue
            target = labels[label_key]
            candidate_mask = (batch or {}).get(HEAD_TO_MASK.get(head, "")) if batch else None
            correct, total = accuracy_from_logits(outputs[head], target, mask=candidate_mask, ignore_index=-1)
            if total == 0:
                continue
            self.correct[label_key] = self.correct.get(label_key, 0) + correct
            self.total[label_key] = self.total.get(label_key, 0) + total

    def compute(self) -> dict[str, float]:
        def acc(key: str) -> float:
            return self.correct.get(key, 0) / max(self.total.get(key, 0), 1)

        keys = list(self.total)
        return {
            "intent_accuracy": acc("intent_label"),
            "metric_aggregation_accuracy": acc("metric_aggregation_label"),
            "metric_column_accuracy": acc("metric_column_index"),
            "dimension_column_accuracy": acc("dimension_column_index"),
            "date_column_accuracy": acc("date_column_index"),
            "filter_column_accuracy": acc("filter_column_index"),
            "dimension_pointer_accuracy": acc("dimension_column_index"),
            "metric_pointer_accuracy": acc("metric_column_index"),
            "date_pointer_accuracy": acc("date_column_index"),
            "filter_pointer_accuracy": acc("filter_column_index"),
            "complexity_accuracy": acc("complexity_label"),
            "overall_slot_accuracy": sum(acc(key) for key in keys) / max(len(keys), 1),
        }


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: ({inner_key: inner_value.to(device) for inner_key, inner_value in value.items()} if key == "labels" else value.to(device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


def _model_outputs(model, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    kwargs = {key: batch[key] for key in MODEL_INPUT_KEYS if key in batch}
    return model(**kwargs)


def _observe_diagnostics(diagnostics, batch: dict[str, Any], outputs: dict[str, Any]) -> None:
    if diagnostics is not None and hasattr(diagnostics, "observe_step"):
        diagnostics.observe_step(batch, outputs)
