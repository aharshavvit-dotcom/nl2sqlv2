from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connected_db_testing.generated_case_runner import ConnectedDBRegressionReporter, ConnectedDBRegressionRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generated connected-database regression cases.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/connected_db_regressions/regression_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.cases.exists():
        raise FileNotFoundError(f"Cases file not found: {args.cases}")
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    schema_path = args.schema or ROOT / "artifacts/schema/current_schema.json"
    if not schema_path.exists():
        schema = _schema_from_cases(cases)
    else:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    report = ConnectedDBRegressionRunner().run(cases, schema)
    ConnectedDBRegressionReporter().write(report, args.output)
    print(json.dumps({"output": str(args.output), "case_pass_rate": report["summary"]["case_pass_rate"]}, indent=2, ensure_ascii=True))
    return 0 if report["summary"]["case_pass_rate"] >= 0.99 else 1


def _schema_from_cases(cases: list[dict]) -> dict:
    tables: dict[str, dict] = {}
    relationships = []
    for case in cases:
        table = (case.get("expected") or {}).get("base_table")
        if table and table not in tables:
            tables[table] = {"columns": {"id": {"type": "integer", "primary_key": True}, "name": {"type": "text"}}}
        if case.get("relationship"):
            relationships.append(case["relationship"])
    return {"dialect": "sqlite", "tables": tables, "relationships": relationships}


if __name__ == "__main__":
    raise SystemExit(main())
