from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_ir.confidence_calibrator import OptionAConfidenceCalibrator
from neural_ir.predictor import OptionAIRPredictor
from neural_ir.schema_linearizer import schema_from_example


def calibrate_option_a_confidence(model_dir: Path, validation_path: Path, output_path: Path) -> dict:
    rows = _load_jsonl(validation_path)
    prediction_rows = []
    predictor = OptionAIRPredictor(str(model_dir)) if (model_dir / "model.pt").exists() else None
    for row in rows:
        if "raw_confidence" in row or "confidence" in row:
            prediction_rows.append(row)
            continue
        if predictor is None:
            continue
        if row.get("correct") is None and row.get("passed") is None:
            continue
        result = predictor.predict(row.get("question", ""), schema_from_example(row))
        prediction_rows.append(
            {
                "raw_confidence": result.get("raw_confidence", result.get("confidence", 0.0)),
                "passed": bool(row.get("correct", row.get("passed", False))),
                "prediction_debug": result.get("debug", {}),
            }
        )
    calibrator = OptionAConfidenceCalibrator()
    payload = calibrator.fit(prediction_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(str(output_path))
    return payload


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Option A confidence calibration.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = calibrate_option_a_confidence(args.model_dir, args.validation, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
