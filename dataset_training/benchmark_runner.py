from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset_evaluator import DatasetScaleEvaluator
from .reporting import save_report_pair
from .utils import read_jsonl


class BenchmarkRunner:
    def run(self, input_path: str | Path, output_path: str | Path, model_name: str = "gold_query_ir") -> dict[str, Any]:
        rows = read_jsonl(input_path)
        mode = "explicit_gold_replay_baseline" if "gold" in model_name else "real_model_predictions"
        report = DatasetScaleEvaluator().evaluate_model(model_name, rows, evaluation_mode=mode)
        save_report_pair(output_path, report, "Generic Benchmark Report")
        return report
