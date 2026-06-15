from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_ir.benchmark import HybridBenchmark
from neural_ir.predictor import OptionAIRPredictor
from nl2sql_v1.schema import read_sqlite_schema
from retriever.retrieval_nl2sql_model import RetrievalNL2SQLModel


def run_hybrid_benchmark(eval_cases: Path, db_path: Path | None, option_a_model_dir: Path, output: Path) -> dict:
    cases = _load_jsonl(eval_cases)
    schema = read_sqlite_schema(db_path) if db_path else None
    option_c_model = RetrievalNL2SQLModel.load(use_option_a_fallback=False)
    hybrid_model = RetrievalNL2SQLModel.load(option_a_model_dir=option_a_model_dir, use_option_a_fallback=True)
    option_a = OptionAIRPredictor(str(option_a_model_dir)) if (option_a_model_dir / "model.pt").exists() else None

    benchmark = HybridBenchmark(
        option_c_predictor=(lambda question: option_c_model.predict(question, schema).model_dump()) if schema is not None else None,
        option_a_predictor=(lambda question: option_a.predict(question, schema)) if option_a is not None and schema is not None else None,
        hybrid_predictor=(lambda question: hybrid_model.predict(question, schema, use_option_a_fallback=True).model_dump()) if schema is not None else None,
    )
    report = benchmark.run(cases, db_path=str(db_path) if db_path else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _default_model_dir() -> Path:
    p = ROOT / "artifacts" / "neural_ir_model"
    if p.exists():
        return p
    p = ROOT / "artifacts" / "option_a_ir_model_v2"
    if p.exists():
        return p
    return ROOT / "artifacts" / "option_a_ir_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Option C, Option A V2, and the hybrid router.")
    default_model = _default_model_dir()
    
    parser.add_argument("--eval-cases", type=Path, default=ROOT / "evaluation" / "adaptive_router_benchmark_cases.jsonl")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "sample_retail.db")
    parser.add_argument("--option-a-model-dir", type=Path, default=default_model)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "adaptive_router_benchmark_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_hybrid_benchmark(args.eval_cases, args.db, args.option_a_model_dir, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
