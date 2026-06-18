"""Neural QueryIR Training Optimization Framework.

Provides configurable training infrastructure for the Neural QueryIR Model:
- Feed-forward head blocks
- Activation / optimizer / scheduler factories
- Per-head loss weighting
- Checkpoint management and early stopping
- Training diagnostics
- Experiment runner for hyperparameter search
- Neural candidate ranker
"""

from __future__ import annotations

from .activation_factory import get_activation
from .checkpoint_manager import CheckpointManager
from .early_stopping import EarlyStopping
from .ffn_blocks import FeedForwardBlock
from .loss_weighter import MultiTaskLossWeighter
from .optimizer_factory import build_optimizer
from .scheduler_factory import build_scheduler
from .training_config import NeuralTrainingConfig, load_training_config

__all__ = [
    "CheckpointManager",
    "EarlyStopping",
    "FeedForwardBlock",
    "MultiTaskLossWeighter",
    "NeuralTrainingConfig",
    "build_optimizer",
    "build_scheduler",
    "get_activation",
    "load_training_config",
]
