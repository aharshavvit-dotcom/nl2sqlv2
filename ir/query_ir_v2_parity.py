from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from execution_eval.sql_structure_comparator import SQLStructureComparator

from .ir_to_sql_renderer import IRToSQLRenderer
from .query_ir_migration import QueryIRCompatibilityError, coerce_query_ir_v1, migrate_v1_to_v2
from .query_ir_models import QueryIR
from .query_ir_v2_renderer_adapter import QueryIRV2RendererAdapter


def load_query_ir_examples_from_jsonl(paths: Iterable[str | Path]) -> list[QueryIR]:
    examples: list[QueryIR] = []
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            query_ir = row.get("query_ir")
            if isinstance(query_ir, dict):
                examples.append(coerce_query_ir_v1(query_ir))
    return examples


def run_query_ir_v2_renderer_parity(examples: Iterable[QueryIR | dict[str, Any]]) -> dict[str, Any]:
    renderer = IRToSQLRenderer()
    adapter = QueryIRV2RendererAdapter(renderer=IRToSQLRenderer())
    comparator = SQLStructureComparator()
    report: dict[str, Any] = {
        "total_migrated": 0,
        "total_parity_passed": 0,
        "total_migration_failures": 0,
        "total_sql_normalization_differences": 0,
        "unsupported_conversion_count": 0,
        "failures": [],
    }

    for index, raw in enumerate(examples):
        try:
            query_ir = coerce_query_ir_v1(raw) if isinstance(raw, dict) else raw
            v1_sql = renderer.render(query_ir)
            v2 = migrate_v1_to_v2(query_ir)
            report["total_migrated"] += 1
            v2_sql = adapter.render(v2)
            comparison = comparator.compare(v2_sql, v1_sql, dialect=query_ir.dialect)
            if comparison["structure_score"] >= 0.99 and not comparison["errors"]:
                report["total_parity_passed"] += 1
            else:
                report["failures"].append(
                    {
                        "index": index,
                        "query_ir_id": query_ir.query_ir_id,
                        "errors": comparison["errors"],
                        "structure_score": comparison["structure_score"],
                    }
                )
            if comparison["predicted"]["canonical_sql"] != comparison["gold"]["canonical_sql"]:
                report["total_sql_normalization_differences"] += 1
        except QueryIRCompatibilityError as exc:
            report["unsupported_conversion_count"] += 1
            report["failures"].append({"index": index, "error": exc.to_dict()})
        except Exception as exc:
            report["total_migration_failures"] += 1
            report["failures"].append({"index": index, "error": str(exc)})
    return report


__all__ = ["load_query_ir_examples_from_jsonl", "run_query_ir_v2_renderer_parity"]


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run QueryIR v1/v2 renderer parity over JSONL QueryIR examples.")
    parser.add_argument("paths", nargs="+", help="JSONL files containing a query_ir object per row.")
    args = parser.parse_args()
    examples = load_query_ir_examples_from_jsonl(args.paths)
    report = run_query_ir_v2_renderer_parity(examples)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
