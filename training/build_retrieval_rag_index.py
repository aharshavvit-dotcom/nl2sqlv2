from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.rag_index_builder import RAGIndexBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local RAG indexes for Retrieval QueryIR.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "processed" / "generic_ir_train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "retrieval_ir_model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = RAGIndexBuilder().build_from_jsonl(args.input, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
