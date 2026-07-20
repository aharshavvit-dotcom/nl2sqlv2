"""Safety dataset builder for neural safety supervision.

Generates controlled safety training data with provenance. The neural
safety head is an AUXILIARY signal only — deterministic AST-level and
read-only validation remains the production authority.

Categories:
- SAFE: Valid SELECT queries
- UNSAFE_DDL: CREATE, ALTER, DROP
- UNSAFE_DML: INSERT, UPDATE, DELETE
- UNSAFE_DCL: GRANT, REVOKE
- UNSAFE_TRUNCATE: TRUNCATE TABLE
- UNSAFE_INJECTION: SQL injection patterns

Usage::

    builder = SafetyDatasetBuilder()
    dataset = builder.build()
    builder.save(dataset, Path("data/processed/safety_training.jsonl"))
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Safety category definitions
# ---------------------------------------------------------------------------

SAFE_EXAMPLES = [
    {"question": "Show total sales by region", "sql": "SELECT region, SUM(sales) FROM orders GROUP BY region LIMIT 100"},
    {"question": "How many customers are there?", "sql": "SELECT COUNT(*) FROM customers"},
    {"question": "List all products", "sql": "SELECT name, price FROM products LIMIT 100"},
    {"question": "Show revenue by month", "sql": "SELECT month, SUM(revenue) FROM sales GROUP BY month LIMIT 100"},
    {"question": "What are the top 10 orders?", "sql": "SELECT * FROM orders ORDER BY total DESC LIMIT 10"},
    {"question": "Average order value", "sql": "SELECT AVG(total) FROM orders"},
    {"question": "Count orders by status", "sql": "SELECT status, COUNT(*) FROM orders GROUP BY status LIMIT 100"},
    {"question": "Find customers in New York", "sql": "SELECT name, email FROM customers WHERE city = 'New York' LIMIT 100"},
    {"question": "Show sales trend", "sql": "SELECT date, SUM(amount) FROM sales GROUP BY date ORDER BY date LIMIT 100"},
    {"question": "Total inventory", "sql": "SELECT SUM(quantity) FROM inventory"},
    {"question": "Revenue by category", "sql": "SELECT category, SUM(price * quantity) FROM order_items GROUP BY category LIMIT 100"},
    {"question": "Active users this month", "sql": "SELECT COUNT(DISTINCT user_id) FROM logins WHERE login_date >= '2024-01-01'"},
]

UNSAFE_DDL_EXAMPLES = [
    {"question": "Create a new table", "sql": "CREATE TABLE temp_data (id INT, name TEXT)", "category": "unsafe_ddl"},
    {"question": "Add a column", "sql": "ALTER TABLE users ADD COLUMN age INT", "category": "unsafe_ddl"},
    {"question": "Remove a table", "sql": "DROP TABLE customers", "category": "unsafe_ddl"},
    {"question": "Create an index", "sql": "CREATE INDEX idx_name ON users (name)", "category": "unsafe_ddl"},
    {"question": "Drop the database", "sql": "DROP DATABASE production", "category": "unsafe_ddl"},
]

UNSAFE_DML_EXAMPLES = [
    {"question": "Add a new customer", "sql": "INSERT INTO customers (name, email) VALUES ('John', 'john@example.com')", "category": "unsafe_dml"},
    {"question": "Update the price", "sql": "UPDATE products SET price = 99.99 WHERE id = 1", "category": "unsafe_dml"},
    {"question": "Delete old records", "sql": "DELETE FROM orders WHERE created_at < '2020-01-01'", "category": "unsafe_dml"},
    {"question": "Insert a record", "sql": "INSERT INTO logs (message) VALUES ('test')", "category": "unsafe_dml"},
    {"question": "Remove inactive users", "sql": "DELETE FROM users WHERE active = 0", "category": "unsafe_dml"},
]

UNSAFE_DCL_EXAMPLES = [
    {"question": "Grant access", "sql": "GRANT ALL PRIVILEGES ON orders TO public", "category": "unsafe_dcl"},
    {"question": "Revoke permissions", "sql": "REVOKE SELECT ON customers FROM user1", "category": "unsafe_dcl"},
]

UNSAFE_TRUNCATE_EXAMPLES = [
    {"question": "Clear the table", "sql": "TRUNCATE TABLE orders", "category": "unsafe_truncate"},
    {"question": "Empty the logs", "sql": "TRUNCATE TABLE system_logs", "category": "unsafe_truncate"},
]

UNSAFE_INJECTION_EXAMPLES = [
    {"question": "'; DROP TABLE users; --", "sql": "SELECT * FROM users WHERE name = ''; DROP TABLE users; --'", "category": "unsafe_injection"},
    {"question": "1 OR 1=1", "sql": "SELECT * FROM users WHERE id = 1 OR 1=1", "category": "unsafe_injection"},
    {"question": "admin'--", "sql": "SELECT * FROM users WHERE username = 'admin'--' AND password = 'x'", "category": "unsafe_injection"},
    {"question": "UNION attack", "sql": "SELECT name FROM users UNION SELECT password FROM credentials", "category": "unsafe_injection"},
]


class SafetyDatasetBuilder:
    """Build safety training data with provenance."""

    def __init__(self, augmentation_factor: int = 3):
        """Initialize builder.

        Parameters
        ----------
        augmentation_factor : int
            Number of schema-varied copies per template to generate.
        """
        self.augmentation_factor = augmentation_factor

    def build(self) -> list[dict[str, Any]]:
        """Build the complete safety training dataset.

        Returns
        -------
        list of training rows with safety labels and task masks.
        """
        rows: list[dict[str, Any]] = []

        # Safe examples
        for i, example in enumerate(SAFE_EXAMPLES):
            for aug_idx in range(self.augmentation_factor):
                rows.append(self._safe_row(example, f"safe_{i}_{aug_idx}"))

        # Unsafe examples by category
        all_unsafe = (
            UNSAFE_DDL_EXAMPLES
            + UNSAFE_DML_EXAMPLES
            + UNSAFE_DCL_EXAMPLES
            + UNSAFE_TRUNCATE_EXAMPLES
            + UNSAFE_INJECTION_EXAMPLES
        )
        for i, example in enumerate(all_unsafe):
            for aug_idx in range(self.augmentation_factor):
                rows.append(self._unsafe_row(example, f"unsafe_{i}_{aug_idx}"))

        return rows

    def _safe_row(self, example: dict, example_id: str) -> dict[str, Any]:
        return {
            "example_id": example_id,
            "dataset_name": "safety_synthetic",
            "source_dataset": "safety_synthetic",
            "question": example["question"],
            "source_sql": example["sql"],
            "is_safe": True,
            "safety_label": "safe",
            "safety_category": "safe_select",
            "eligible_for_training": True,
            "task_masks": {
                "safety": 1.0,
                # Other masks are 0 — safety-only supervision
                "intent": 0.0,
                "base_table": 0.0,
                "column": 0.0,
                "aggregation": 0.0,
                "filter": 0.0,
                "join_edge": 0.0,
                "complexity": 0.0,
            },
            "capability_annotation": {
                "task_masks": {"safety": 1.0},
                "safety_labels": ["safe"],
            },
            "metadata": {
                "source": "safety_dataset_builder",
                "category": "safe_select",
                "provenance": "synthetic",
            },
        }

    def _unsafe_row(self, example: dict, example_id: str) -> dict[str, Any]:
        category = example.get("category", "unsafe")
        return {
            "example_id": example_id,
            "dataset_name": "safety_synthetic",
            "source_dataset": "safety_synthetic",
            "question": example["question"],
            "source_sql": example["sql"],
            "is_safe": False,
            "safety_label": "unsafe",
            "safety_category": category,
            "eligible_for_training": True,
            "task_masks": {
                "safety": 1.0,
                "intent": 0.0,
                "base_table": 0.0,
                "column": 0.0,
                "aggregation": 0.0,
                "filter": 0.0,
                "join_edge": 0.0,
                "complexity": 0.0,
            },
            "capability_annotation": {
                "task_masks": {"safety": 1.0},
                "safety_labels": [category],
            },
            "metadata": {
                "source": "safety_dataset_builder",
                "category": category,
                "provenance": "synthetic",
            },
        }

    def save(self, dataset: list[dict[str, Any]], path: Path) -> dict[str, Any]:
        """Save dataset to JSONL with summary stats.

        Returns
        -------
        dict with summary statistics.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in dataset:
                f.write(json.dumps(row, default=str) + "\n")

        safe_count = sum(1 for r in dataset if r.get("is_safe"))
        unsafe_count = len(dataset) - safe_count
        categories = {}
        for r in dataset:
            cat = r.get("safety_category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        summary = {
            "total": len(dataset),
            "safe": safe_count,
            "unsafe": unsafe_count,
            "by_category": categories,
            "output_path": str(path),
        }
        return summary
