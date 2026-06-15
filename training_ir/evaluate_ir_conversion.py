from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT = ROOT / "artifacts" / "option_a_ir_data" / "ir_conversion_eval.json"


def evaluate_ir_conversion(input_path: Path, output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    by_dataset: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "successful": 0})
    by_intent = Counter()
    by_unsupported_reason = Counter()
    failures = []
    successful = 0
    ir_valid = 0
    sql_valid = 0
    roundtrip_valid = 0

    for row in rows:
        dataset = row.get("dataset_name") or "unknown"
        by_dataset[dataset]["total"] += 1
        unsupported_reason = row.get("unsupported_reason")
        if unsupported_reason:
            by_unsupported_reason[unsupported_reason] += 1
            failures.append(sample_failure(row, unsupported_reason))
            continue
        successful += 1
        by_dataset[dataset]["successful"] += 1
        by_intent[row.get("intent") or "unknown"] += 1
        ir_validation = row.get("ir_validation") or {}
        if ir_validation.get("is_valid", bool(row.get("query_ir"))):
            ir_valid += 1
        sql_validation = row.get("sql_validation") or {}
        if sql_validation.get("is_valid", False):
            sql_valid += 1
        roundtrip = row.get("roundtrip_validation") or {}
        if roundtrip.get("is_valid", False):
            roundtrip_valid += 1
        else:
            failures.append(sample_failure(row, "roundtrip_validation_failed"))

    payload = {
        "input": str(input_path),
        "total_examples": len(rows),
        "successful_examples": successful,
        "failed_examples": len(rows) - successful,
        "conversion_success_rate": successful / len(rows) if rows else 0.0,
        "ir_validation_rate": ir_valid / successful if successful else 0.0,
        "sql_validation_rate": sql_valid / successful if successful else 0.0,
        "roundtrip_validation_rate": roundtrip_valid / successful if successful else 0.0,
        "by_dataset": dict(by_dataset),
        "by_intent": dict(by_intent),
        "by_unsupported_reason": dict(by_unsupported_reason),
        "sample_failures": failures[:25],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def sample_failure(row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "example_id": row.get("example_id"),
        "dataset_name": row.get("dataset_name"),
        "db_id": row.get("db_id"),
        "reason": reason,
        "question": row.get("question"),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QueryIR conversion output.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = evaluate_ir_conversion(args.input, args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
