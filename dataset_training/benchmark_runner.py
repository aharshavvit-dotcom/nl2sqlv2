"""Gold-replay-only benchmark runner.

This class always uses gold QueryIR as predictions (oracle upper bound).
It is NOT a real model benchmark. Output is explicitly marked
``is_valid_for_quality_gate = False`` and cannot be used for production
quality gates or promotion.

For real-model benchmark evaluation, use:
    python training/run_unseen_db_benchmark.py
    python training/evaluate_generic_models.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset_evaluator import DatasetScaleEvaluator
from .reporting import save_report_pair
from .utils import read_jsonl


class GoldReplayBenchmarkRunner:
    """Oracle/debug benchmark that replays gold QueryIR as predictions.

    All output is marked ``gold_replay_used = True`` and
    ``is_valid_for_quality_gate = False``.
    """

    def run(self, input_path: str | Path, output_path: str | Path, model_name: str = "gold_query_ir") -> dict[str, Any]:
        rows = read_jsonl(input_path)
        report = DatasetScaleEvaluator().evaluate_model(
            model_name,
            rows,
            evaluation_mode="explicit_gold_replay_baseline",
            model_artifact_source="gold_replay",
        )
        report["gold_replay_used"] = True
        report["is_valid_for_quality_gate"] = False
        save_report_pair(output_path, report, "Gold Replay Benchmark Report (Debug Only)")
        return report


# Backward-compatible alias — deprecated, use GoldReplayBenchmarkRunner
BenchmarkRunner = GoldReplayBenchmarkRunner
