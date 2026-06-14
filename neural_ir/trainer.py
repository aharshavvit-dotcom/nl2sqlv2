from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


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
}


class OptionAIRTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config.get("learning_rate", 0.001)))

    def train(self, train_loader, val_loader, label_encoder, output_dir) -> dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        best_loss = float("inf")
        best_state = None
        patience = 0
        history = []
        epochs = int(self.config.get("epochs", 5))
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate_epoch(val_loader) if val_loader is not None else {}
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"validation_{k}": v for k, v in val_metrics.items()}}
            history.append(row)
            val_loss = float(val_metrics.get("loss", train_metrics.get("loss", 0.0)))
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 3:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        torch.save(self.model.state_dict(), output_path / "model.pt")
        metrics = {"best_validation_loss": best_loss, "epochs_ran": len(history), "history": history}
        (output_path / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    def train_epoch(self, loader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_items = 0
        metric_state = _MetricState()
        for batch in loader:
            batch = _to_device(batch, self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch["question_ids"], batch["schema_ids"], batch["question_mask"], batch["schema_mask"])
            loss = self._loss(outputs, batch["labels"])
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item()) * int(batch["question_ids"].size(0))
            total_items += int(batch["question_ids"].size(0))
            metric_state.update(outputs, batch["labels"])
        return {"loss": total_loss / max(total_items, 1), **metric_state.compute()}

    def evaluate_epoch(self, loader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_items = 0
        metric_state = _MetricState()
        with torch.no_grad():
            for batch in loader:
                batch = _to_device(batch, self.device)
                outputs = self.model(batch["question_ids"], batch["schema_ids"], batch["question_mask"], batch["schema_mask"])
                loss = self._loss(outputs, batch["labels"])
                total_loss += float(loss.item()) * int(batch["question_ids"].size(0))
                total_items += int(batch["question_ids"].size(0))
                metric_state.update(outputs, batch["labels"])
        return {"loss": total_loss / max(total_items, 1), **metric_state.compute()}

    def _loss(self, outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = []
        for head, label in HEAD_TO_LABEL.items():
            target = labels[label]
            if not target.ne(-1).any():
                continue
            losses.append(self.loss_fn(outputs[head], target))
        if not losses:
            return outputs["intent_logits"].sum() * 0.0
        return torch.stack(losses).sum()


class _MetricState:
    def __init__(self) -> None:
        self.correct: dict[str, int] = {}
        self.total: dict[str, int] = {}

    def update(self, outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> None:
        for head, label_key in HEAD_TO_LABEL.items():
            target = labels[label_key]
            mask = target.ne(-1)
            total = int(mask.sum().item())
            if total == 0:
                continue
            pred = outputs[head].argmax(dim=-1)
            self.correct[label_key] = self.correct.get(label_key, 0) + int(pred.eq(target).logical_and(mask).sum().item())
            self.total[label_key] = self.total.get(label_key, 0) + total

    def compute(self) -> dict[str, float]:
        def acc(key: str) -> float:
            return self.correct.get(key, 0) / max(self.total.get(key, 0), 1)

        keys = list(self.total)
        return {
            "intent_accuracy": acc("intent_label"),
            "metric_aggregation_accuracy": acc("metric_aggregation_label"),
            "dimension_pointer_accuracy": acc("dimension_column_index"),
            "metric_pointer_accuracy": acc("metric_column_index"),
            "date_pointer_accuracy": acc("date_column_index"),
            "filter_pointer_accuracy": acc("filter_column_index"),
            "overall_slot_accuracy": sum(acc(key) for key in keys) / max(len(keys), 1),
        }


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: ({inner_key: inner_value.to(device) for inner_key, inner_value in value.items()} if key == "labels" else value.to(device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }
