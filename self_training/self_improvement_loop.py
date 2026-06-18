from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataset_training.utils import read_jsonl, write_json

from .correction_builder import CorrectionBuilder
from .hard_negative_miner import HardNegativeMiner
from .iteration_reporter import IterationReporter
from .ranking_trainer import RankingTrainer


class SelfImprovementLoop:
    def run(
        self,
        train_path: str | Path,
        validation_path: str | Path,
        retrieval_model_dir: str | Path,
        neural_model_dir: str | Path,
        output_dir: str | Path,
        iterations: int = 2,
        max_examples: int | None = 1000,
    ) -> dict[str, Any]:
        from training.evaluate_against_gold import evaluate_against_gold

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        reporter = IterationReporter()
        train_rows = read_jsonl(train_path)
        validation_rows = read_jsonl(validation_path)
        if max_examples:
            validation_rows = validation_rows[:max_examples]
        temp_validation = output / "_validation_subset.jsonl"
        _write_jsonl(temp_validation, validation_rows)
        summaries = []
        for iteration in range(1, iterations + 1):
            iteration_dir = output / f"iteration_{iteration:03d}"
            predictions = iteration_dir / "validation_predictions.jsonl"
            comparison_report = iteration_dir / "gold_comparison_report.json"
            args = _Args(
                input=temp_validation,
                retrieval_model_dir=Path(retrieval_model_dir),
                neural_model_dir=Path(neural_model_dir),
                output=predictions,
                report=comparison_report,
                max_examples=max_examples,
            )
            report = evaluate_against_gold(args)
            rows = read_jsonl(predictions)
            corrections = CorrectionBuilder().build(rows)
            negatives = HardNegativeMiner().mine(rows)
            _write_jsonl(iteration_dir / "correction_positive_examples.jsonl", corrections["correction_positive_examples"])
            _write_jsonl(iteration_dir / "mined_hard_negatives.jsonl", negatives["mined_hard_negatives"])
            ranker_report = RankingTrainer().train(rows, output / "adaptive_ranker")
            iteration_report = {
                "iteration": iteration,
                "summary": {
                    **report["summary"],
                    "training_examples": len(train_rows),
                    "corrections": corrections["summary"]["positive_examples"],
                    "hard_negatives": negatives["error_summary"]["total_errors"],
                    "ranker_candidates": ranker_report["candidates"],
                },
                "warnings": report.get("warnings", []),
            }
            reporter.write(iteration_dir, iteration_report, title=f"Self-Improvement Iteration {iteration:03d}")
            summaries.append(iteration_report)
        summary = {
            "iterations": iterations,
            "iteration_summaries": summaries,
            "improved": _improved(summaries),
        }
        write_json(output / "summary_report.json", summary)
        reporter.write(output, {"summary": {"iterations": iterations, "improved": summary["improved"]}}, title="Self-Improvement Summary")
        (output / "summary_report.md").write_text((output / "report.md").read_text(encoding="utf-8"), encoding="utf-8")
        return summary


class _Args:
    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _improved(summaries: list[dict[str, Any]]) -> bool:
    if len(summaries) < 2:
        return False
    first = summaries[0]["summary"].get("gold_comparison_score", 0.0)
    last = summaries[-1]["summary"].get("gold_comparison_score", 0.0)
    return last >= first
