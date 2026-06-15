from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_ir.error_analysis import OptionAErrorAnalyzer
from neural_ir.ir_dataset import load_jsonl
from neural_ir.ir_label_encoder import IRLabelEncoder
from neural_ir.model_registry import load_model_bundle
from neural_ir.predictor import OptionAIRPredictor
from neural_ir.schema_linearizer import extract_schema_items, schema_from_example


DEFAULT_MODEL_DIR = ROOT / "artifacts" / "option_a_ir_model"


def analyze_option_a_errors(model_dir: Path, test_path: Path, output_path: Path, max_examples: int | None = 250) -> dict[str, Any]:
    rows = load_jsonl(test_path)
    if max_examples is not None:
        rows = rows[:max_examples]
    bundle = load_model_bundle(model_dir)
    label_encoder: IRLabelEncoder = bundle["label_encoder"]
    predictor = OptionAIRPredictor(str(model_dir))
    prediction_rows = []
    for row in rows:
        schema = schema_from_example(row)
        schema_items = extract_schema_items(schema)
        gold = label_encoder.decode(label_encoder.encode(row.get("query_ir") or {}, schema_items), schema_items)
        result = predictor.predict(row.get("question", ""), schema)
        prediction_rows.append(
            {
                "id": row.get("example_id"),
                "example_id": row.get("example_id"),
                "question": row.get("question"),
                "dataset_name": row.get("dataset_name") or _dataset_from_id(row.get("example_id")),
                "gold": gold,
                "prediction": (result.get("debug") or {}).get("decoded_prediction", {}),
                "ir_validation": result.get("ir_validation"),
                "sql_validation": result.get("sql_validation"),
                "confidence": result.get("confidence"),
            }
        )
    report = OptionAErrorAnalyzer().analyze(prediction_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _dataset_from_id(example_id: Any) -> str:
    raw = str(example_id or "unknown")
    return raw.split(":", 1)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Option A prediction errors by intent, dataset, and slot.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "processed" / "ir_test_examples.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR / "error_analysis_report.json")
    parser.add_argument("--max-examples", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze_option_a_errors(args.model_dir, args.test, args.output, max_examples=args.max_examples)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

