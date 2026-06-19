from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from .attention_model import DEFAULT_V2_CONFIG, SchemaAwareOptionAIRModel
from .ir_label_encoder import IRLabelEncoder
from .model import DEFAULT_CONFIG, OptionAIRModel
from .vocab import Vocabulary


SCHEMA_AWARE_MODEL_VERSIONS = {"option_a_v2", "schema_aware_queryir_v1", "neural_queryir_v1"}


def save_model_bundle(model, vocab: Vocabulary, label_encoder: IRLabelEncoder, config: dict, output_dir) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path / "model.pt")
    vocab.save(str(output_path / "vocab.json"))
    label_encoder.save(str(output_path / "label_maps.json"))
    defaults = DEFAULT_V2_CONFIG if (config or {}).get("model_version") in SCHEMA_AWARE_MODEL_VERSIONS else DEFAULT_CONFIG
    (output_path / "config.yaml").write_text(yaml.safe_dump({**defaults, **(config or {})}, sort_keys=True), encoding="utf-8")


def load_model_bundle(model_dir) -> dict[str, Any]:
    model_path = Path(model_dir)
    vocab = Vocabulary.load(str(model_path / "vocab.json"))
    label_encoder = IRLabelEncoder.load(str(model_path / "label_maps.json"))
    config_path = model_path / "config.yaml"
    config = {**DEFAULT_CONFIG}
    if config_path.exists():
        config.update(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    if config.get("model_version") in SCHEMA_AWARE_MODEL_VERSIONS or config.get("architecture") == "schema_aware_queryir":
        config = {**DEFAULT_V2_CONFIG, **config}
        model = SchemaAwareOptionAIRModel(config=config, vocab_size=len(vocab), label_sizes=label_encoder.label_sizes)
    else:
        model = OptionAIRModel(config=config, vocab_size=len(vocab), label_sizes=label_encoder.label_sizes)
    state = torch.load(model_path / "model.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return {"model": model, "vocab": vocab, "label_encoder": label_encoder, "config": config}
