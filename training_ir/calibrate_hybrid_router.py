from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_ir.calibration import HybridRouterCalibrator
from neural_ir.predictor import OptionAIRPredictor
from nl2sql_v1.schema import read_sqlite_schema
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel


DEFAULT_MODEL_DIR = ROOT / "artifacts" / "option_a_ir_model"


def calibrate_hybrid_router(eval_cases: Path, db_path: Path, option_a_model_dir: Path, output_path: Path) -> dict[str, Any]:
    rows = _load_jsonl(eval_cases)
    schema = read_sqlite_schema(db_path)
    option_c_model = RetrievalNL2SQLModel.load(use_option_a_fallback=False, option_a_model_dir=option_a_model_dir)
    option_a_predictor = OptionAIRPredictor(str(option_a_model_dir)) if (option_a_model_dir / "model.pt").exists() else None
    option_c_results = []
    option_a_results = []
    for row in rows:
        option_c = option_c_model.predict(row["question"], schema, use_option_a_fallback=False)
        option_c_results.append(
            {
                "id": row.get("id"),
                "confidence": option_c.confidence,
                "validation": option_c.validation,
                "expected_source": row.get("expected_source"),
            }
        )
        if option_a_predictor is not None:
            option_a = option_a_predictor.predict(row["question"], schema)
        else:
            option_a = {"confidence": 0.0, "sql_validation": {"is_valid": False, "issues": ["Option A model missing"]}}
        option_a_results.append(
            {
                "id": row.get("id"),
                "confidence": option_a.get("confidence", 0.0),
                "sql_validation": option_a.get("sql_validation") or option_a.get("validation") or {},
                "expected_source": row.get("expected_source"),
            }
        )
    report = HybridRouterCalibrator().calibrate(option_c_results, option_a_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the Option C / Option A hybrid router.")
    parser.add_argument("--eval-cases", type=Path, default=ROOT / "evaluation" / "option_a_eval_cases.jsonl")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--option-a-model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR / "hybrid_calibration.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = calibrate_hybrid_router(args.eval_cases, args.db, args.option_a_model_dir, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

