from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_ir.predictor import OptionAIRPredictor
from nl2sql_v1.schema import read_sqlite_schema


DEFAULT_MODEL_DIR = ROOT / "artifacts" / "option_a_ir_model"


def predict_with_option_a(model_dir: Path, db_path: Path, question: str) -> dict:
    schema = read_sqlite_schema(db_path)
    predictor = OptionAIRPredictor(str(model_dir))
    return predictor.predict(question, schema)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Option A QueryIR prediction.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--question", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = predict_with_option_a(args.model_dir, args.db, args.question)
    decoded = result.get("debug", {}).get("decoded_prediction", {})
    print(f"predicted intent: {decoded.get('intent')}")
    print("predicted slots:")
    print(json.dumps(decoded, indent=2))
    print("QueryIR:")
    print(json.dumps(result.get("query_ir"), indent=2))
    print("IR validation:")
    print(json.dumps(result.get("ir_validation"), indent=2))
    print("SQL:")
    print(result.get("sql") or "")
    print("SQL validation:")
    print(json.dumps(result.get("sql_validation"), indent=2))
    return 0 if result.get("sql_validation", {}).get("is_valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
