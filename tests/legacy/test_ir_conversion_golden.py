from __future__ import annotations

import json
from pathlib import Path

from ir.sql_to_ir_converter import SQLToIRConverter
from nl2sql_v1.schema import read_sqlite_schema
from scripts.create_sample_db import build_database


ROOT = Path(__file__).resolve().parents[1]


def test_ir_conversion_golden_cases(tmp_path: Path) -> None:
    db_path = tmp_path / "retail.db"
    build_database(db_path)
    schema = read_sqlite_schema(db_path)
    converter = SQLToIRConverter()

    rows = [
        json.loads(line)
        for line in (ROOT / "evaluation" / "ir_conversion_golden.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows

    for row in rows:
        result = converter.convert(row["question"], row["sql"], schema, dataset_name="golden", example_id=row["id"])
        assert result["success"], result
        assert result["ir_validation"]["is_valid"], result
        assert result["roundtrip_sql"], result
        assert result["sql_validation"]["is_valid"], result
        assert result["roundtrip_validation"]["is_valid"], result
        query_ir = result["query_ir"]
        expected = row["expected"]
        assert query_ir["intent"] == expected["intent"]
        assert query_ir["limit"] == expected["limit"]
        if "base_table" in expected:
            assert query_ir["base_table"] == expected["base_table"]
        if "metric_expression_contains" in expected:
            assert expected["metric_expression_contains"] in query_ir["metrics"][0]["expression"]
        if "metric_expression" in expected:
            assert query_ir["metrics"][0]["expression"] == expected["metric_expression"]
        if "metric_aggregation" in expected:
            assert query_ir["metrics"][0]["aggregation"] == expected["metric_aggregation"]
        if "dimension_expression" in expected:
            assert query_ir["dimensions"][0]["expression"] == expected["dimension_expression"]
        if "join_condition" in expected:
            assert expected["join_condition"] in {join["condition"] for join in query_ir["joins"]}
        if "join_conditions" in expected:
            actual_conditions = {join["condition"] for join in query_ir["joins"]}
            for condition in expected["join_conditions"]:
                assert condition in actual_conditions
        if "date_grain" in expected:
            assert query_ir["date_filters"][0]["date_grain"] == expected["date_grain"]
            assert query_ir["date_filters"][0]["date_expression"] == expected["date_expression"]
        if "date_start" in expected:
            assert query_ir["date_filters"][0]["start_date"] == expected["date_start"]
            assert query_ir["date_filters"][0]["end_date"] == expected["date_end"]
        if "date_filter_start_date" in expected:
            assert query_ir["date_filters"][0]["start_date"] == expected["date_filter_start_date"]
            assert query_ir["date_filters"][0]["end_date"] == expected["date_filter_end_date"]
        if "filter_expression" in expected:
            filter_item = query_ir["filters"][0]
            assert filter_item["expression"] == expected["filter_expression"]
            assert filter_item["operator"] == expected["filter_operator"]
            assert filter_item["value"] == expected["filter_value"]
