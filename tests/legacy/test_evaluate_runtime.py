"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.create_sample_db import build_database
from scripts.evaluate_runtime import evaluate_runtime


def test_evaluate_runtime_writes_report(tmp_path: Path) -> None:
    db_path = tmp_path / "retail.db"
    output_path = tmp_path / "runtime_report.json"
    golden_path = tmp_path / "golden.jsonl"
    build_database(db_path)
    golden_path.write_text(
        json.dumps(
            {
                "id": "top_5_customers_by_sales",
                "question": "Top 5 customers by sales",
                "expected_template_id": "top_n_metric_by_dimension",
                "expected_intent": "top_n_metric_by_dimension",
                "expected_query_ir": {
                    "base_table": "orders",
                    "metric_expression_contains": "orders.amount",
                    "dimension_expression": "customers.customer_name",
                    "required_tables": ["orders", "customers"],
                    "join_conditions": ["orders.customer_id = customers.customer_id"],
                    "limit": 5,
                },
                "expected_sql_contains": ["SUM(orders.amount)", "LIMIT 5"],
                "expected_sql_not_contains": ["SELECT *"],
                "should_execute": True,
                "expected_result_columns": ["customer", "revenue"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_runtime(
        db_path=db_path,
        artifact_dir=tmp_path / "missing_artifact",
        golden_file=golden_path,
        output=output_path,
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["total_cases"] == 1
    assert payload["passed_cases"] == 1
    assert payload["query_ir_match_rate"] == 1.0
