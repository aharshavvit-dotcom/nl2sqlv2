from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_ir.ir_dataset import load_jsonl
from neural_ir.training_curriculum import CurriculumPlanner
from training_ir.train_option_a_model import train_option_a_model


DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "option_a_ir_model"


def train_option_a_curriculum(
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    epochs_per_phase: int = 2,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    max_examples_per_phase: int | None = 200,
    seed: int = 13,
) -> dict[str, Any]:
    rows = load_jsonl(train_path)
    validation_rows = load_jsonl(validation_path)
    phases = CurriculumPlanner().split(rows, max_examples_per_phase=max_examples_per_phase)
    if not phases:
        raise ValueError(f"No curriculum phases could be built from {train_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "_curriculum"
    temp_dir.mkdir(parents=True, exist_ok=True)
    validation_limited = validation_rows[: max(1, min(len(validation_rows), 200))]
    validation_temp = temp_dir / "validation.jsonl"
    _write_jsonl(validation_temp, validation_limited)

    phase_reports = []
    for phase_idx, phase in enumerate(phases, start=1):
        if not phase["rows"]:
            continue
        train_temp = temp_dir / f"{phase['name']}.jsonl"
        _write_jsonl(train_temp, phase["rows"])
        metrics = train_option_a_model(
            train_path=train_temp,
            validation_path=validation_temp,
            output_dir=output_dir,
            max_examples=None,
            epochs=epochs_per_phase,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + phase_idx,
        )
        phase_reports.append(
            {
                "phase": phase["name"],
                "example_count": phase["example_count"],
                "by_dataset": phase["by_dataset"],
                "by_complexity": phase["by_complexity"],
                "metrics": metrics,
            }
        )

    report = {
        "phases": phase_reports,
        "epochs_per_phase": epochs_per_phase,
        "batch_size": batch_size,
        "max_examples_per_phase": max_examples_per_phase,
        "output_dir": str(output_dir),
    }
    (output_dir / "curriculum_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Option A with a simple WikiSQL -> Spider -> BIRD Mini curriculum.")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "processed" / "ir_training_examples.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "data" / "processed" / "ir_validation_examples.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs-per-phase", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-examples-per-phase", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_examples = args.max_examples_per_phase if args.max_examples_per_phase > 0 else None
    report = train_option_a_curriculum(
        train_path=args.train,
        validation_path=args.validation,
        output_dir=args.output_dir,
        epochs_per_phase=args.epochs_per_phase,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_examples_per_phase=max_examples,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
