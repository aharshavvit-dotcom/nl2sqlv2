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


def _is_correct_ir(pred: dict | None, gold: dict | None) -> bool:
    if not pred or not gold:
        return False
    
    # Check intent and base table
    if pred.get("intent") != gold.get("intent"):
        return False
    if pred.get("base_table") != gold.get("base_table"):
        return False
    
    # Compare metric columns and aggregations
    pred_metrics = {(m.get("column"), m.get("aggregation")) for m in pred.get("metrics", []) if isinstance(m, dict)}
    gold_metrics = {(m.get("column"), m.get("aggregation")) for m in gold.get("metrics", []) if isinstance(m, dict)}
    if pred_metrics != gold_metrics:
        return False
        
    # Compare dimension columns
    pred_dims = {d.get("column") for d in pred.get("dimensions", []) if isinstance(d, dict)}
    gold_dims = {d.get("column") for d in gold.get("dimensions", []) if isinstance(d, dict)}
    if pred_dims != gold_dims:
        return False

    # Compare filter columns and operators
    pred_filters = {(f.get("column"), f.get("operator")) for f in pred.get("filters", []) if isinstance(f, dict)}
    gold_filters = {(f.get("column"), f.get("operator")) for f in gold.get("filters", []) if isinstance(f, dict)}
    if pred_filters != gold_filters:
        return False

    # Compare date filter columns and types
    pred_dates = {(df.get("date_column"), df.get("filter_type")) for df in pred.get("date_filters", []) if isinstance(df, dict)}
    gold_dates = {(df.get("date_column"), df.get("filter_type")) for df in gold.get("date_filters", []) if isinstance(df, dict)}
    if pred_dates != gold_dates:
        return False

    return True


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
        
        correct_val = row.get("correct")
        passed_val = row.get("passed")
        
        if correct_val is not None:
            passed = bool(correct_val)
        elif passed_val is not None:
            passed = bool(passed_val)
        elif "query_ir" in row:
            # Dynamically evaluate predicted QueryIR against ground-truth query_ir in the example
            result = predictor.predict(row.get("question", ""), schema_from_example(row))
            pred_ir = result.get("repaired_query_ir") or result.get("query_ir")
            gold_ir = row.get("query_ir")
            passed = _is_correct_ir(pred_ir, gold_ir)
            prediction_rows.append(
                {
                    "raw_confidence": result.get("raw_confidence", result.get("confidence", 0.0)),
                    "passed": passed,
                    "prediction_debug": result.get("debug", {}),
                }
            )
            continue
        else:
            continue
            
        result = predictor.predict(row.get("question", ""), schema_from_example(row))
        prediction_rows.append(
            {
                "raw_confidence": result.get("raw_confidence", result.get("confidence", 0.0)),
                "passed": passed,
                "prediction_debug": result.get("debug", {}),
            }
        )
    calibrator = OptionAConfidenceCalibrator()
    payload = calibrator.fit(prediction_rows, dataset_path=validation_path)
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
    default_model = ROOT / "artifacts" / "neural_ir_model"
    if not default_model.exists():
        default_model = ROOT / "artifacts" / "option_a_ir_model_v2"
    if not default_model.exists():
        default_model = ROOT / "artifacts" / "option_a_ir_model"
        
    parser.add_argument("--model-dir", type=Path, default=default_model)
    parser.add_argument("--validation", type=Path, default=ROOT / "data" / "processed" / "ir_validation_examples.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = args.model_dir / "option_a_calibration.json"
    return args


def main() -> int:
    args = parse_args()
    report = calibrate_option_a_confidence(args.model_dir, args.validation, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
