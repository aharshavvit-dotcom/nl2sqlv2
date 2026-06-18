from __future__ import annotations

from typing import Any, Callable

from generic_planner import SchemaProfile, TableIntentResolver
from ir.ir_to_sql_renderer import IRToSQLRenderer
from validation.sql_validator import SQLValidator


GENERIC_POSTGRES_SCHEMA = {
    "dialect": "postgres",
    "database": "regression",
    "schema_name": "public",
    "tables": {
        "users": {
            "columns": [
                {"name": "id", "type": "integer", "is_primary_key": True},
                {"name": "name", "type": "text"},
                {"name": "role", "type": "text"},
                {"name": "created_at", "type": "timestamp"},
                {"name": "password_hash", "type": "text"},
            ],
            "primary_keys": ["id"],
            "foreign_keys": [],
        },
        "berth_masters": {
            "columns": [
                {"name": "id", "type": "integer", "is_primary_key": True},
                {"name": "berth_name", "type": "text"},
                {"name": "berth_code", "type": "text"},
            ],
            "primary_keys": ["id"],
            "foreign_keys": [],
        },
        "assignments": {
            "columns": [
                {"name": "id", "type": "integer", "is_primary_key": True},
                {"name": "user_id", "type": "integer"},
                {"name": "berth_id", "type": "integer"},
                {"name": "assigned_date", "type": "date"},
                {"name": "status", "type": "text"},
            ],
            "primary_keys": ["id"],
            "foreign_keys": [
                {"column": "user_id", "references_table": "users", "references_column": "id"},
                {"column": "berth_id", "references_table": "berth_masters", "references_column": "id"},
            ],
        },
    },
    "relationships": [
        {"from_table": "assignments", "from_column": "user_id", "to_table": "users", "to_column": "id"},
        {"from_table": "assignments", "from_column": "berth_id", "to_table": "berth_masters", "to_column": "id"},
    ],
}


SAMPLE_RETAIL_SCHEMA = {
    "dialect": "sqlite",
    "tables": {
        "customers": {"columns": [{"name": "customer_id", "type": "integer"}, {"name": "customer_name", "type": "text"}]},
        "products": {"columns": [{"name": "product_id", "type": "integer"}, {"name": "product_name", "type": "text"}]},
        "orders": {"columns": [{"name": "order_id", "type": "integer"}, {"name": "customer_id", "type": "integer"}, {"name": "amount", "type": "real"}, {"name": "status", "type": "text"}, {"name": "order_date", "type": "date"}]},
        "order_items": {"columns": [{"name": "order_id", "type": "integer"}, {"name": "product_id", "type": "integer"}, {"name": "quantity", "type": "integer"}, {"name": "price", "type": "real"}]},
    },
}


DEFAULT_CASES = [
    {"case_id": "list_all_users", "question": "list all users", "schema_id": "generic_pg", "expected_intent": "show_records", "expected_base_table": "users", "expect_no_join": True},
    {"case_id": "list_all_berth_masters", "question": "list all berth_masters", "schema_id": "generic_pg", "expected_intent": "show_records", "expected_base_table": "berth_masters", "expect_no_join": True},
    {"case_id": "list_assignments", "question": "list assignments", "schema_id": "generic_pg", "expected_intent": "show_records", "expected_base_table": "assignments", "expect_no_join": True},
    {"case_id": "count_users", "question": "count users", "schema_id": "generic_pg", "expected_intent": "count_records", "expected_base_table": "users", "expect_no_join": True},
    {"case_id": "users_role_admin", "question": "show users where role is admin", "schema_id": "generic_pg", "expected_intent": "simple_filter", "expected_base_table": "users", "expect_no_join": True},
    {"case_id": "top_customers_sales", "question": "top 5 customers by sales", "schema_id": "sample_retail", "expected_intent": "top_n_metric_by_dimension"},
    {"case_id": "top_products_revenue", "question": "top products by revenue", "schema_id": "sample_retail", "expected_intent": "top_n_metric_by_dimension"},
    {"case_id": "sales_last_month", "question": "sales last month", "schema_id": "sample_retail", "expected_intent": "metric_summary"},
    {"case_id": "orders_completed", "question": "orders where status is completed", "schema_id": "sample_retail", "expected_intent": "simple_filter"},
]


Predictor = Callable[[str, dict[str, Any]], dict[str, Any]]


class RegressionSuite:
    def __init__(self, predictor: Predictor | None = None):
        self.predictor = predictor
        self.renderer = IRToSQLRenderer()
        self.validator = SQLValidator()

    def run(
        self,
        cases: list[dict[str, Any]] | None = None,
        feedback_safety_regressions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        rows = cases or DEFAULT_CASES
        case_results = [self._run_case(case) for case in rows]
        safety_results = [self._run_safety_case(case) for case in feedback_safety_regressions or []]
        blocking = [row for row in [*case_results, *safety_results] if row["status"] == "fail"]
        passed = [row for row in [*case_results, *safety_results] if row["status"] == "pass"]
        warnings = [row for row in case_results if row["status"] == "warning"]
        total = len(case_results) + len(safety_results)
        return {
            "passed": not blocking,
            "summary": {
                "total_cases": total,
                "passed": len(passed),
                "failed": len(blocking),
                "warnings": len(warnings),
                "pass_rate": len(passed) / total if total else 1.0,
            },
            "case_results": case_results,
            "safety_regression_results": safety_results,
            "blocking_failures": blocking,
        }

    def _run_case(self, case: dict[str, Any]) -> dict[str, Any]:
        schema = self._schema_for_case(case)
        prediction = self.predictor(case["question"], schema) if self.predictor else self._default_prediction(case["question"], schema)
        query_ir = prediction.get("query_ir") or prediction.get("selected_query_ir") or {}
        issues: list[str] = []
        if case.get("expected_intent") and query_ir.get("intent") != case["expected_intent"]:
            issues.append(f"expected intent {case['expected_intent']}, got {query_ir.get('intent')}")
        if case.get("expected_base_table") and query_ir.get("base_table") != case["expected_base_table"]:
            issues.append(f"expected base table {case['expected_base_table']}, got {query_ir.get('base_table')}")
        if case.get("expect_no_join") and query_ir.get("joins"):
            issues.append("unexpected join")
        sql = prediction.get("sql")
        if sql and case.get("expect_no_join") and "JOIN" in sql.upper():
            issues.append("unexpected join in SQL")
        status = "fail" if issues else "pass"
        if prediction.get("not_handled") and not any(case.get(key) for key in ["expected_base_table", "expect_no_join"]):
            status = "warning"
            issues.append("case not handled by local deterministic suite")
        return {
            "case_id": case.get("case_id"),
            "question": case.get("question"),
            "status": status,
            "issues": issues,
            "predicted_intent": query_ir.get("intent"),
            "predicted_base_table": query_ir.get("base_table"),
            "join_count": len(query_ir.get("joins") or []),
        }

    def _default_prediction(self, question: str, schema: dict[str, Any]) -> dict[str, Any]:
        result = TableIntentResolver(SchemaProfile(schema)).resolve(question)
        if not result.handled or result.query_ir is None:
            return {"not_handled": True, "query_ir": {"intent": None, "joins": []}}
        sql = self.renderer.render(result.query_ir, dialect=schema.get("dialect", "sqlite"))
        return {"query_ir": result.query_ir.model_dump(), "sql": sql}

    def _run_safety_case(self, case: dict[str, Any]) -> dict[str, Any]:
        sql = case.get("generated_sql") or case.get("sql") or ""
        validation = self.validator.validate(sql, schema=None)
        passed = not validation.get("is_valid", validation.get("ok", False))
        return {
            "case_id": case.get("case_id"),
            "question": case.get("question"),
            "status": "pass" if passed else "fail",
            "issues": [] if passed else ["unsafe feedback SQL validated as safe"],
            "validation": validation,
        }

    @staticmethod
    def _schema_for_case(case: dict[str, Any]) -> dict[str, Any]:
        if case.get("schema") and isinstance(case["schema"], dict):
            return case["schema"]
        if case.get("schema_id") == "sample_retail":
            return SAMPLE_RETAIL_SCHEMA
        return GENERIC_POSTGRES_SCHEMA
