"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from tests.test_neural_ir_dataset import test_ir_training_dataset_loads_small_jsonl as _dataset_smoke


def test_ir_training_dataset_entrypoint(tmp_path) -> None:
    _dataset_smoke(tmp_path)
