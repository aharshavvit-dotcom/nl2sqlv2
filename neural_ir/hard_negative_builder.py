from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .candidate_builder import SchemaCandidateBuilder
from .schema_linearizer import schema_from_example


class HardNegativeBuilder:
    def __init__(self) -> None:
        self.candidate_builder = SchemaCandidateBuilder()

    def build_for_example(self, example: dict, schema_candidates: dict) -> list[dict]:
        gold_ir = deepcopy(example.get("query_ir") or example.get("gold_query_ir") or {})
        if not gold_ir:
            return []
        negatives = []
        for negative_type, mutator in [
            ("wrong_metric_column", self._wrong_metric_column),
            ("wrong_dimension_column", self._wrong_dimension_column),
            ("wrong_table", self._wrong_table),
            ("wrong_date_column", self._wrong_date_column),
            ("wrong_filter_column", self._wrong_filter_column),
            ("wrong_aggregation", self._wrong_aggregation),
            ("product_revenue_wrong_grain", self._product_revenue_wrong_grain),
        ]:
            negative_ir, reason = mutator(gold_ir, schema_candidates, example)
            if negative_ir and negative_ir != gold_ir:
                negatives.append(self._row(example, negative_type, gold_ir, negative_ir, reason))
        return negatives

    def build_file(self, input_path: str, output_path: str, max_negatives_per_example: int = 5):
        input_file = Path(input_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        rows_written = 0
        with output_file.open("w", encoding="utf-8") as out:
            for example in _load_jsonl(input_file):
                schema = schema_from_example(example)
                candidates = self.candidate_builder.build_candidates(schema, example.get("question", ""))
                negatives = self.build_for_example(example, candidates)[:max_negatives_per_example]
                for row in negatives:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_written += 1
        return {"input": str(input_file), "output": str(output_file), "hard_negative_rows": rows_written}

    @staticmethod
    def _row(example: dict, negative_type: str, gold_ir: dict, negative_ir: dict, reason: str) -> dict:
        return {
            "example_id": example.get("example_id") or example.get("id"),
            "negative_type": negative_type,
            "question": example.get("question", ""),
            "schema": example.get("schema") or schema_from_example(example),
            "gold_query_ir": gold_ir,
            "negative_query_ir": negative_ir,
            "gold_label": 1,
            "negative_label": 0,
            "metadata": {"reason": reason},
        }

    @staticmethod
    def _wrong_metric_column(gold_ir: dict, candidates: dict, example: dict) -> tuple[dict | None, str]:
        metrics = gold_ir.get("metrics") or []
        if not metrics:
            return None, ""
        current = metrics[0]
        replacement = _first_different_column(candidates.get("metric_candidates", []), current)
        if not replacement:
            return None, ""
        negative = deepcopy(gold_ir)
        metric = negative["metrics"][0]
        metric["table"] = replacement["table"]
        metric["column"] = replacement["column"]
        metric["expression"] = replacement["display"]
        metric["alias"] = replacement["column"]
        return negative, f"{replacement['display']} used instead of the gold metric column"

    @staticmethod
    def _wrong_dimension_column(gold_ir: dict, candidates: dict, example: dict) -> tuple[dict | None, str]:
        dimensions = gold_ir.get("dimensions") or []
        if not dimensions:
            return None, ""
        current = dimensions[0]
        replacement = _first_different_column(candidates.get("dimension_candidates", []), current)
        if not replacement:
            return None, ""
        negative = deepcopy(gold_ir)
        dimension = negative["dimensions"][0]
        dimension["table"] = replacement["table"]
        dimension["column"] = replacement["column"]
        dimension["expression"] = replacement["display"]
        dimension["alias"] = replacement["column"]
        return negative, f"{replacement['display']} used instead of the gold dimension column"

    @staticmethod
    def _wrong_table(gold_ir: dict, candidates: dict, example: dict) -> tuple[dict | None, str]:
        current = gold_ir.get("base_table")
        replacement = next((item for item in candidates.get("tables", []) if item.get("table") != current), None)
        if not replacement:
            return None, ""
        negative = deepcopy(gold_ir)
        negative["base_table"] = replacement["table"]
        negative["required_tables"] = list(dict.fromkeys([replacement["table"], *negative.get("required_tables", [])]))
        return negative, f"{replacement['table']} used as base table instead of {current}"

    @staticmethod
    def _wrong_date_column(gold_ir: dict, candidates: dict, example: dict) -> tuple[dict | None, str]:
        date_filters = gold_ir.get("date_filters") or []
        if not date_filters:
            return None, ""
        current = {"table": date_filters[0].get("date_table"), "column": date_filters[0].get("date_column")}
        replacement = _first_different_column(candidates.get("date_candidates", []), current)
        if not replacement:
            replacement = next((item for item in candidates.get("columns", []) if item.get("type") != "date"), None)
        if not replacement:
            return None, ""
        negative = deepcopy(gold_ir)
        date_filter = negative["date_filters"][0]
        date_filter["date_table"] = replacement["table"]
        date_filter["date_column"] = replacement["column"]
        date_filter["date_expression"] = replacement["display"]
        return negative, f"{replacement['display']} used where the gold date column is required"

    @staticmethod
    def _wrong_filter_column(gold_ir: dict, candidates: dict, example: dict) -> tuple[dict | None, str]:
        filters = gold_ir.get("filters") or []
        if not filters:
            return None, ""
        current = filters[0]
        replacement = _first_different_column(candidates.get("filter_candidates", []), current)
        if not replacement:
            return None, ""
        negative = deepcopy(gold_ir)
        item = negative["filters"][0]
        item["table"] = replacement["table"]
        item["column"] = replacement["column"]
        item["expression"] = replacement["display"]
        return negative, f"{replacement['display']} used instead of the gold filter column"

    @staticmethod
    def _wrong_aggregation(gold_ir: dict, candidates: dict, example: dict) -> tuple[dict | None, str]:
        metrics = gold_ir.get("metrics") or []
        if not metrics:
            return None, ""
        current = str(metrics[0].get("aggregation") or "SUM").upper()
        replacement = "AVG" if current != "AVG" else "SUM"
        negative = deepcopy(gold_ir)
        negative["metrics"][0]["aggregation"] = replacement
        return negative, f"{replacement} aggregation used instead of {current}"

    @staticmethod
    def _product_revenue_wrong_grain(gold_ir: dict, candidates: dict, example: dict) -> tuple[dict | None, str]:
        question = str(example.get("question") or "").lower()
        dimensions = " ".join(str(item.get("name") or item.get("column") or "") for item in gold_ir.get("dimensions") or []).lower()
        if "product" not in question and "product" not in dimensions:
            return None, ""
        if "revenue" not in question and "sales" not in question:
            return None, ""
        order_amount = next((item for item in candidates.get("metric_candidates", []) if item["display"] == "orders.amount"), None)
        if not order_amount:
            return None, ""
        negative = deepcopy(gold_ir)
        if not negative.get("metrics"):
            negative["metrics"] = [{"name": "revenue", "aggregation": "SUM", "alias": "revenue"}]
        metric = negative["metrics"][0]
        metric.update(
            {
                "name": "revenue",
                "aggregation": "SUM",
                "table": "orders",
                "column": "amount",
                "expression": "orders.amount",
                "alias": "revenue",
            }
        )
        return negative, "orders.amount used where order_items.quantity * order_items.price is required"


def _first_different_column(candidates: list[dict[str, Any]], current: dict[str, Any]) -> dict[str, Any] | None:
    current_ref = (current.get("table"), current.get("column"))
    return next((item for item in candidates if (item.get("table"), item.get("column")) != current_ref), None)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
