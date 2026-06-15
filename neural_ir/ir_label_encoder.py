from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


INTENTS = [
    "show_records",
    "count_records",
    "count_by_dimension",
    "metric_summary",
    "metric_by_dimension",
    "top_n_metric_by_dimension",
    "bottom_n_metric_by_dimension",
    "trend_by_date",
    "simple_filter",
]
METRIC_AGGREGATIONS = ["NONE", "COUNT", "SUM", "AVG", "MIN", "MAX"]
METRIC_EXPRESSION_TYPES = ["none", "column", "count_star", "product_revenue_expression"]
DATE_GRAINS = ["none", "month", "year"]
DATE_FILTER_TYPES = ["none", "last_month", "this_month", "last_year", "this_year", "last_30_days", "absolute_range"]
FILTER_OPERATORS = ["none", "equals", "not_equals", "greater_than", "greater_equal", "less_than", "less_equal", "contains", "in"]
ORDER_DIRECTIONS = ["none", "ASC", "DESC"]
LIMIT_BUCKETS = ["default_100", "top_5", "top_10", "top_20", "custom"]


class IRLabelEncoder:
    def __init__(self) -> None:
        self.last_warnings: list[str] = []
        self.label_maps = {
            "intent": _map(INTENTS),
            "metric_aggregation": _map(METRIC_AGGREGATIONS),
            "metric_expression_type": _map(METRIC_EXPRESSION_TYPES),
            "date_grain": _map(DATE_GRAINS),
            "date_filter_type": _map(DATE_FILTER_TYPES),
            "filter_operator": _map(FILTER_OPERATORS),
            "order_direction": _map(ORDER_DIRECTIONS),
            "limit_bucket": _map(LIMIT_BUCKETS),
        }

    @property
    def label_sizes(self) -> dict[str, int]:
        return {name: len(values) for name, values in self.label_maps.items()}

    def fit(self, examples: list[dict]) -> None:
        return None

    def encode(self, query_ir: dict[str, Any], schema_items: dict[str, Any]) -> dict[str, int]:
        self.last_warnings = []
        intent = query_ir.get("template_id") or query_ir.get("intent") or "show_records"
        metric = _first(query_ir.get("metrics"))
        dimension = _first(query_ir.get("dimensions"))
        date_filter = _first(query_ir.get("date_filters"))
        ir_filter = _first(query_ir.get("filters"))
        order_by = _first(query_ir.get("order_by"))
        metric_expression_type = _metric_expression_type(metric)
        aggregation = "COUNT" if metric_expression_type == "count_star" else str((metric or {}).get("aggregation") or "NONE").upper()
        base_table_index = _table_index(schema_items, query_ir.get("base_table"))
        metric_column_index = -1 if metric_expression_type in {"count_star", "product_revenue_expression"} else _column_index_from_any(
            schema_items,
            (metric or {}).get("table"),
            (metric or {}).get("column"),
            (metric or {}).get("expression"),
        )
        dimension_column_index = _column_index_from_any(
            schema_items,
            (dimension or {}).get("table"),
            (dimension or {}).get("column"),
            (dimension or {}).get("expression"),
        )
        date_column_index = _column_index_from_any(
            schema_items,
            (date_filter or {}).get("date_table"),
            (date_filter or {}).get("date_column"),
            (date_filter or {}).get("date_expression"),
        )
        filter_column_index = _column_index_from_any(
            schema_items,
            (ir_filter or {}).get("table"),
            (ir_filter or {}).get("column"),
            (ir_filter or {}).get("expression"),
        )
        self._warn_missing_pointer("base_table", query_ir.get("base_table"), base_table_index)
        if metric and metric_expression_type == "column":
            self._warn_missing_pointer("metric_column", (metric or {}).get("column") or (metric or {}).get("expression"), metric_column_index)
        if dimension:
            self._warn_missing_pointer("dimension_column", (dimension or {}).get("column") or (dimension or {}).get("expression"), dimension_column_index)
        if date_filter:
            self._warn_missing_pointer("date_column", (date_filter or {}).get("date_column") or (date_filter or {}).get("date_expression"), date_column_index)
        if ir_filter:
            self._warn_missing_pointer("filter_column", (ir_filter or {}).get("column") or (ir_filter or {}).get("expression"), filter_column_index)

        return {
            "intent_label": self._label("intent", intent, "show_records"),
            "base_table_index": base_table_index,
            "metric_aggregation_label": self._label("metric_aggregation", aggregation, "NONE"),
            "metric_column_index": metric_column_index,
            "metric_expression_type_label": self._label("metric_expression_type", metric_expression_type, "none"),
            "dimension_column_index": dimension_column_index,
            "date_column_index": date_column_index,
            "date_grain_label": self._label("date_grain", _date_grain(date_filter), "none"),
            "date_filter_type_label": self._label("date_filter_type", _date_filter_type(date_filter), "none"),
            "filter_column_index": filter_column_index,
            "filter_operator_label": self._label("filter_operator", (ir_filter or {}).get("operator") or "none", "none"),
            "order_direction_label": self._label("order_direction", (order_by or {}).get("direction") or "none", "none"),
            "limit_bucket_label": self._label("limit_bucket", _limit_bucket(query_ir.get("limit")), "default_100"),
        }

    def decode(self, predictions: dict[str, int], schema_items: dict[str, Any]) -> dict[str, Any]:
        intent = self._decode_label("intent", predictions.get("intent_label", 0))
        metric_aggregation = self._decode_label("metric_aggregation", predictions.get("metric_aggregation_label", 0))
        metric_expression_type = self._decode_label(
            "metric_expression_type",
            predictions.get("metric_expression_type_label", self.label_maps["metric_expression_type"]["none"]),
        )
        date_grain = self._decode_label("date_grain", predictions.get("date_grain_label", 0))
        date_filter_type = self._decode_label("date_filter_type", predictions.get("date_filter_type_label", 0))
        filter_operator = self._decode_label("filter_operator", predictions.get("filter_operator_label", 0))
        order_direction = self._decode_label("order_direction", predictions.get("order_direction_label", 0))
        limit_bucket = self._decode_label("limit_bucket", predictions.get("limit_bucket_label", 0))
        return {
            "intent": intent,
            "template_id": intent,
            "base_table": _table_at(schema_items, predictions.get("base_table_index", -1)),
            "metric_aggregation": metric_aggregation,
            "metric_column": _column_at(schema_items, predictions.get("metric_column_index", -1)),
            "metric_expression_type": metric_expression_type,
            "dimension_column": _column_at(schema_items, predictions.get("dimension_column_index", -1)),
            "date_column": _column_at(schema_items, predictions.get("date_column_index", -1)),
            "date_grain": date_grain,
            "date_filter_type": date_filter_type,
            "filter_column": _column_at(schema_items, predictions.get("filter_column_index", -1)),
            "filter_operator": filter_operator,
            "order_direction": order_direction,
            "limit_bucket": limit_bucket,
            "limit": _limit_from_bucket(limit_bucket),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"label_maps": self.label_maps}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "IRLabelEncoder":
        encoder = cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        encoder.label_maps = {
            name: {str(label): int(idx) for label, idx in values.items()}
            for name, values in payload.get("label_maps", encoder.label_maps).items()
        }
        return encoder

    def _label(self, name: str, value: Any, default: str) -> int:
        label = str(value or default)
        return self.label_maps[name].get(label, self.label_maps[name][default])

    def _decode_label(self, name: str, idx: Any) -> str:
        inverse = {value: key for key, value in self.label_maps[name].items()}
        return inverse.get(int(idx), next(iter(self.label_maps[name])))

    def _warn_missing_pointer(self, slot: str, value: Any, pointer: int) -> None:
        if value and pointer == -1:
            self.last_warnings.append(f"could not encode {slot} pointer for {value}")


def _map(labels: list[str]) -> dict[str, int]:
    return {label: idx for idx, label in enumerate(labels)}


def _first(values: Any) -> dict[str, Any] | None:
    if isinstance(values, list) and values:
        return values[0]
    return None


def _table_index(schema_items: dict[str, Any], table: Any) -> int:
    if not table:
        return -1
    try:
        return list(schema_items.get("tables", [])).index(str(table))
    except ValueError:
        return -1


def _column_index(schema_items: dict[str, Any], table: Any, column: Any) -> int:
    return _column_index_from_any(schema_items, table, column, None)


def _column_index_from_any(schema_items: dict[str, Any], table: Any, column: Any, expression: Any) -> int:
    if not table or not column or column == "*":
        refs = _expression_refs(expression)
        for ref_table, ref_column in refs:
            idx = _column_index(schema_items, ref_table, ref_column)
            if idx != -1:
                return idx
        if column and column != "*":
            return _column_index_by_name(schema_items, column)
        return -1
    parsed = _parse_qualified_column(str(column))
    if parsed:
        table, column = parsed
    target = (_clean_identifier(table), _clean_identifier(column))
    for idx, item in enumerate(schema_items.get("columns", [])):
        if (_clean_identifier(item.get("table")), _clean_identifier(item.get("column"))) == target:
            return idx
    by_name = _column_index_by_name(schema_items, column)
    if by_name != -1:
        return by_name
    for ref_table, ref_column in _expression_refs(expression):
        idx = _column_index(schema_items, ref_table, ref_column)
        if idx != -1:
            return idx
    return -1


def _column_index_by_name(schema_items: dict[str, Any], column: Any) -> int:
    cleaned = _clean_identifier(column)
    matches = [
        idx
        for idx, item in enumerate(schema_items.get("columns", []))
        if _clean_identifier(item.get("column")) == cleaned
    ]
    return matches[0] if len(matches) == 1 else -1


def _expression_refs(expression: Any) -> list[tuple[str, str]]:
    if not expression:
        return []
    cleaned = str(expression)
    cleaned = cleaned.replace('"', "").replace("`", "").replace("[", "").replace("]", "")
    return [
        (_clean_identifier(table), _clean_identifier(column))
        for table, column in re.findall(r"\b([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\b", cleaned)
    ]


def _parse_qualified_column(value: str) -> tuple[str, str] | None:
    cleaned = value.replace('"', "").replace("`", "").replace("[", "").replace("]", "")
    if "." not in cleaned:
        return None
    table, column = cleaned.rsplit(".", 1)
    return _clean_identifier(table), _clean_identifier(column)


def _clean_identifier(value: Any) -> str:
    return str(value or "").strip().strip('"`[]')


def _table_at(schema_items: dict[str, Any], idx: Any) -> str | None:
    tables = schema_items.get("tables", [])
    try:
        index = int(idx)
    except Exception:
        return None
    if 0 <= index < len(tables):
        return str(tables[index])
    return None


def _column_at(schema_items: dict[str, Any], idx: Any) -> dict[str, str] | None:
    columns = schema_items.get("columns", [])
    try:
        index = int(idx)
    except Exception:
        return None
    if 0 <= index < len(columns):
        item = columns[index]
        return {"table": str(item["table"]), "column": str(item["column"]), "type": str(item.get("type", ""))}
    return None


def _metric_expression_type(metric: dict[str, Any] | None) -> str:
    if not metric:
        return "none"
    expression = str(metric.get("expression") or "")
    if expression == "*" or str(metric.get("column") or "") == "*":
        return "count_star"
    lowered = expression.lower()
    if "order_items.quantity" in lowered and "order_items.price" in lowered:
        return "product_revenue_expression"
    if metric.get("column"):
        return "column"
    return "none"


def _date_grain(date_filter: dict[str, Any] | None) -> str:
    if not date_filter or date_filter.get("filter_type") != "grain":
        return "none"
    grain = str(date_filter.get("date_grain") or "month").lower()
    return "year" if grain == "year" else "month"


def _date_filter_type(date_filter: dict[str, Any] | None) -> str:
    if not date_filter or date_filter.get("filter_type") == "grain":
        return "none"
    if date_filter.get("filter_type") == "absolute_range":
        return "absolute_range"
    raw = str(date_filter.get("raw_text") or "").lower().replace(" ", "_")
    aliases = {
        "last_month": "last_month",
        "this_month": "this_month",
        "last_year": "last_year",
        "this_year": "this_year",
        "last_30_days": "last_30_days",
    }
    return aliases.get(raw, "none")


def _limit_bucket(limit: Any) -> str:
    try:
        value = int(limit or 100)
    except Exception:
        value = 100
    if value == 5:
        return "top_5"
    if value == 10:
        return "top_10"
    if value == 20:
        return "top_20"
    if value == 100:
        return "default_100"
    return "custom"


def _limit_from_bucket(bucket: str) -> int:
    return {"top_5": 5, "top_10": 10, "top_20": 20}.get(bucket, 100)
