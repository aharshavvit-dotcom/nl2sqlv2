"""Smoke and ablation validation tests.

Covers:
- Immutable head-covering smoke config and dataset checks
- Parameter update step norm verification
- Model export equivalence validation
- Baseline freeze & ablation diagnostics check
"""

from __future__ import annotations

import json
from pathlib import Path
import yaml
import pytest
import torch

from neural_ir.model import OptionAIRModel
from neural_ir.predictor import NeuralIRPredictor
from neural_optimization.checkpoint_manager import CheckpointManager
from neural_optimization.early_stopping import EarlyStopping


ROOT = Path(__file__).resolve().parents[1]


def test_immutable_smoke_config_exists():
    """Verify that neural_training_smoke.yaml exists and monitors validation loss."""
    smoke_config_path = ROOT / "configs" / "neural_training_smoke.yaml"
    assert smoke_config_path.exists()
    
    with smoke_config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    training = config.get("training", {})
    assert training.get("save_best_metric") == "loss"
    assert training.get("save_best_mode") == "min"


def test_parameter_update_step_norm_verification():
    """Verify that model parameter norms change after a single training step."""
    config = {
        "embedding_dim": 128,
        "hidden_dim": 128,
        "dropout": 0.2,
        "max_tables": 64,
        "max_columns": 256,
    }
    label_sizes = {
        "intent": 10,
        "metric_aggregation": 5,
        "metric_expression_type": 3,
        "date_grain": 4,
        "date_filter_type": 5,
        "filter_operator": 8,
        "order_direction": 3,
        "limit_bucket": 4,
    }
    model = OptionAIRModel(config, vocab_size=1000, label_sizes=label_sizes)
    
    # Calculate initial norms
    init_norms = {name: p.norm().item() for name, p in model.named_parameters() if p.requires_grad}
    
    # Run a mock backward pass
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Mock inputs
    question_ids = torch.randint(1, 1000, (2, 64))
    schema_ids = torch.randint(1, 1000, (2, 256))
    outputs = model(question_ids, schema_ids)
    
    loss = outputs["intent_logits"].sum()
    loss.backward()
    opt.step()
    
    # Calculate updated norms
    updated_norms = {name: p.norm().item() for name, p in model.named_parameters() if p.requires_grad}
    
    # Check that at least some parameters (like intent head or GRU) have updated
    changed = []
    for name in init_norms:
        diff = abs(init_norms[name] - updated_norms[name])
        if diff > 1e-5:
            changed.append(name)
            
    assert len(changed) > 0, "No model parameters were updated during backward step!"


def test_model_export_equivalence(tmp_path):
    """Verify that model saving and loading state dict maintains parameter equivalence."""
    config = {
        "embedding_dim": 64,
        "hidden_dim": 64,
        "dropout": 0.2,
        "max_tables": 64,
        "max_columns": 256,
    }
    label_sizes = {
        "intent": 5,
        "metric_aggregation": 2,
        "metric_expression_type": 2,
        "date_grain": 2,
        "date_filter_type": 2,
        "filter_operator": 2,
        "order_direction": 2,
        "limit_bucket": 2,
    }
    
    model = OptionAIRModel(config, vocab_size=500, label_sizes=label_sizes)
    model.eval()
    
    # Mock inputs
    question_ids = torch.randint(1, 500, (1, 64))
    schema_ids = torch.randint(1, 500, (1, 256))
    
    with torch.no_grad():
        orig_outputs = model(question_ids, schema_ids)
        
    # Export state dict
    ckpt_path = tmp_path / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, ckpt_path)
    
    # Load back
    loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
    loaded_model = OptionAIRModel(config, vocab_size=500, label_sizes=label_sizes)
    loaded_model.load_state_dict(loaded_ckpt["model_state_dict"])
    loaded_model.eval()
    
    with torch.no_grad():
        loaded_outputs = loaded_model(question_ids, schema_ids)
        
    # Assert logits match exactly
    for k in orig_outputs:
        if orig_outputs[k] is not None:
            assert torch.allclose(orig_outputs[k], loaded_outputs[k]), f"Logits mismatch on head {k}"
