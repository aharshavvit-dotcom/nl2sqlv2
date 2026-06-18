from __future__ import annotations

from typing import Any

from .schema_profiler import column_aliases_for, is_metric_column, normalize_schema, singularize, tokenize


METRIC_HINTS = {"amount", "total", "price", "quantity", "duration", "cost", "score", "rate", "balance", "weight"}


class MetricDetector:
    def detect(self, schema: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_schema(schema)
        metrics: dict[str, dict[str, Any]] = {}
        for table, info in normalized["tables"].items():
            stem = "_".join(singularize(token) for token in tokenize(table)) or table
            count_key = f"{stem}_count"
            metrics[count_key] = {
                "expression": "COUNT(*)",
                "base_table": table,
                "column": None,
                "aggregation": "COUNT",
                "aliases": [f"{stem.replace('_', ' ')} count", f"number of {table.replace('_', ' ')}", f"count {table.replace('_', ' ')}"],
                "confidence": 0.9,
            }
            for column in info.get("columns", []):
                if not is_metric_column(column):
                    continue
                key = f"{table}_{column['name']}".lower()
                aliases = column_aliases_for(column["name"])
                confidence = 0.9 if any(hint in column["name"].lower() for hint in METRIC_HINTS) else 0.75
                metrics[key] = {
                    "expression": f"{table}.{column['name']}",
                    "base_table": table,
                    "column": column["name"],
                    "aggregation": "SUM",
                    "aliases": aliases,
                    "confidence": confidence,
                }
        return {"metrics": metrics}
