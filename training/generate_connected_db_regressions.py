from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connected_db_testing.schema_case_generator import SchemaCaseGenerator, write_cases_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate connected-database regression cases from a schema JSON file.")
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/connected_db_regressions/generated_cases.jsonl")
    parser.add_argument("--max-tables", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.schema.exists():
        raise FileNotFoundError(f"Schema file not found: {args.schema}")
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    cases = SchemaCaseGenerator().generate_cases(schema, max_tables=args.max_tables)
    write_cases_jsonl(str(args.output), cases)
    print(json.dumps({"output": str(args.output), "case_count": len(cases)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
