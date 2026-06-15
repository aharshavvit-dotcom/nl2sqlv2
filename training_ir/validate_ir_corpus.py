from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ir.ir_to_sql_renderer import IRToSQLRenderer
from ir.ir_validator import IRValidator
from ir.query_ir_models import QueryIR
from validation.sql_validator import SQLValidator


DEFAULT_OUTPUT = ROOT / "artifacts" / "option_a_ir_data" / "ir_validation_report.json"


def validate_ir_corpus(input_path: Path, output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    renderer = IRToSQLRenderer()
    ir_validator = IRValidator()
    sql_validator = SQLValidator()
    results = []
    issues_by_type: Counter[str] = Counter()
    for index, row in enumerate(rows):
        issues: list[str] = []
        query_ir_payload = row.get("query_ir")
        rendered_sql = row.get("rendered_sql")
        query_ir = None

        required = ["question", "source_sql", "query_ir"]
        for key in required:
            if not row.get(key):
                issues.append(f"missing {key}")
        if query_ir_payload:
            try:
                query_ir = QueryIR(**query_ir_payload)
            except Exception as exc:
                issues.append(f"invalid QueryIR payload: {exc}")

        schema = schema_from_query_ir_payload(query_ir_payload)
        ir_validation = ir_validator.validate(query_ir, schema=schema) if query_ir else None
        if ir_validation and not ir_validation.is_valid:
            issues.extend(ir_validation.errors)
        if query_ir and not rendered_sql:
            rendered_sql = renderer.render(query_ir)
        sql_validation = sql_validator.validate(rendered_sql, schema=schema) if rendered_sql else {"is_valid": False, "issues": ["missing rendered_sql"]}
        if not sql_validation.get("is_valid"):
            issues.extend(str(issue) for issue in sql_validation.get("issues", []))
        if query_ir:
            if query_ir.limit <= 0:
                issues.append("missing or invalid QueryIR limit")
            if not query_ir.required_tables:
                issues.append("missing required tables")
            if query_ir.template_id in {"metric_summary", "metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension", "trend_by_date"} and not query_ir.metrics:
                issues.append("missing required metrics")
            if query_ir.template_id in {"metric_by_dimension", "top_n_metric_by_dimension", "bottom_n_metric_by_dimension", "count_by_dimension"} and not query_ir.dimensions:
                issues.append("missing required dimensions")
            source_sql = str(row.get("source_sql") or "").lower()
            if " where " in f" {source_sql} " and not query_ir.filters and not [item for item in query_ir.date_filters if item.filter_type != "grain"]:
                issues.append("missing filters for filtered source SQL")
            if "date" in source_sql and any(operator in source_sql for operator in [">=", "<=", ">", "<"]):
                date_ranges = [item for item in query_ir.date_filters if item.filter_type == "absolute_range"]
                if not date_ranges:
                    issues.append("missing date filter for date-filter source SQL")

        for issue in dict.fromkeys(issues):
            issues_by_type[classify_issue(issue)] += 1

        results.append(
            {
                "index": index,
                "example_id": row.get("example_id"),
                "is_valid": not issues,
                "issues": list(dict.fromkeys(issues)),
            }
        )

    valid_count = sum(1 for item in results if item["is_valid"])
    invalid_count = len(rows) - valid_count
    sample_failures = [item for item in results if not item["is_valid"]][:25]
    report = {
        "input_file": str(input_path),
        "total_rows": len(rows),
        "valid_rows": valid_count,
        "invalid_rows": invalid_count,
        "validity_rate": valid_count / len(rows) if rows else 0.0,
        "issues_by_type": dict(issues_by_type),
        "sample_failures": sample_failures,
        "input": str(input_path),
        "total_examples": len(rows),
        "valid_examples": valid_count,
        "invalid_examples": invalid_count,
        "validation_rate": valid_count / len(rows) if rows else 0.0,
        "failures": sample_failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def schema_from_query_ir_payload(query_ir_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not query_ir_payload:
        return None
    schema_context = (
        (query_ir_payload.get("metadata") or {})
        .get("validation_context", {})
        .get("schema_context", {})
    )
    tables = schema_context.get("tables")
    return {"tables": tables} if tables else None


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


def classify_issue(issue: str) -> str:
    lowered = issue.lower()
    if lowered.startswith("missing "):
        return lowered.replace("missing ", "missing_").split(" ", 1)[0]
    if "select *" in lowered:
        return "select_star"
    if "limit" in lowered:
        return "limit"
    if "sensitive" in lowered:
        return "sensitive_column"
    if "unknown table" in lowered:
        return "unknown_table"
    if "unknown column" in lowered:
        return "unknown_column"
    if "queryir" in lowered:
        return "query_ir"
    if "sql" in lowered:
        return "sql_validation"
    return lowered.split(":", 1)[0].replace(" ", "_")[:80]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a QueryIR training JSONL corpus.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_ir_corpus(args.input, args.output)
    print(json.dumps(report, indent=2))
    return 0 if report["invalid_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
