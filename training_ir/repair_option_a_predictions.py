from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_ir.ir_repair import OptionAIRRepairer
from neural_ir.schema_linearizer import schema_from_example


def repair_predictions(input_path: Path, output_path: Path) -> dict:
    repairer = OptionAIRRepairer()
    rows_written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in _load_jsonl(input_path):
            query_ir = row.get("query_ir") or row.get("predicted_query_ir")
            if not query_ir:
                continue
            repaired = repairer.repair(query_ir, schema=row.get("schema") or schema_from_example(row), question=row.get("question", ""))
            out.write(json.dumps({**row, "repair": repaired, "repaired_query_ir": repaired.get("query_ir")}, ensure_ascii=False) + "\n")
            rows_written += 1
    return {"input": str(input_path), "output": str(output_path), "rows_written": rows_written}


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
    parser = argparse.ArgumentParser(description="Repair Option A QueryIR prediction JSONL rows.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = repair_predictions(args.input, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
