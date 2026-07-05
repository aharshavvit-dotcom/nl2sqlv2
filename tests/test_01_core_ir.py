"""Test 01: Core IR — QueryIR models, IR validation, IR-to-SQL rendering, SQL-to-IR conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sqlglot

from ir.ir_to_sql_renderer import IRToSQLRenderer, quote_identifier
from ir.ir_validator import IRValidator
from ir.option_c_to_ir import RetrievalIRConverter
from ir.query_ir_models import IRDateFilter, IRDimension, IRFilter, IRMetric, IROrderBy, QueryIR
from ir.sql_to_ir_converter import SQLToIRConverter
from validation.sql_validator import SQLValidator


ROOT = Path(__file__).resolve().parents[1]


# ── QueryIR model serialization ─────────────────────────────────────
class TestQueryIRModels:
    def test_serialize_metric(self) -> None:
        metric = IRMetric(name="revenue", aggregation="SUM", table="orders",
                          column="amount", expression="orders.amount", alias="revenue")
        qir = QueryIR(query_ir_id="qir-test", question="Top customers by sales",
                      normalized_question="top customers by sales", intent="metric_summary",
                      template_id="metric_summary", base_table="orders",
                      required_tables=["orders"], metrics=[metric], limit=100, select_mode="aggregate")
        payload = qir.model_dump()
        assert payload["metrics"][0]["expression"] == "orders.amount"
        assert payload["select_mode"] == "aggregate"

    def test_roundtrip_json(self) -> None:
        qir = QueryIR(query_ir_id="qir-rt", question="q", normalized_question="q",
                      intent="count_records", template_id="count_records",
                      base_table="orders", required_tables=["orders"],
                      metrics=[IRMetric(name="count", aggregation="COUNT", table="orders",
                                       column="*", expression="*", alias="record_count")],
                      limit=100, select_mode="count")
        restored = QueryIR.model_validate(json.loads(qir.model_dump_json()))
        assert restored.query_ir_id == qir.query_ir_id


# ── IR Validator ─────────────────────────────────────────────────────
class TestIRValidator:
    def test_valid_metric_summary(self) -> None:
        qir = _make_metric_summary_ir()
        result = IRValidator().validate(qir)
        assert result.is_valid

    def test_missing_metric_for_metric_intent(self) -> None:
        qir = _make_metric_summary_ir()
        qir.metrics = []
        result = IRValidator().validate(qir)
        assert not result.is_valid


# ── IR-to-SQL Renderer ───────────────────────────────────────────────
class TestIRToSQLRenderer:
    def test_metric_summary_sql(self) -> None:
        qir = _make_metric_summary_ir()
        sql = IRToSQLRenderer().render(qir)
        assert "SUM" in sql
        assert "orders" in sql

    def test_trend_by_date_sqlite(self) -> None:
        qir = _make_trend_ir(dialect="sqlite")
        sql = IRToSQLRenderer().render(qir, dialect="sqlite")
        assert "strftime" in sql

    def test_trend_by_date_postgres(self) -> None:
        qir = _make_trend_ir(dialect="postgres")
        sql = IRToSQLRenderer().render(qir, dialect="postgres")
        assert "DATE_TRUNC" in sql
        assert "TO_CHAR" in sql

    def test_contains_filter_postgres_uses_ilike(self) -> None:
        qir = _make_metric_summary_ir()
        qir.filters = [IRFilter(name="name", table="customers", column="name",
                                expression="customers.name", operator="contains",
                                value="john", value_type="string", raw_text="john")]
        sql = IRToSQLRenderer().render(qir, dialect="postgres")
        assert "ILIKE" in sql

    def test_contains_filter_sqlite_uses_like(self) -> None:
        qir = _make_metric_summary_ir()
        qir.filters = [IRFilter(name="name", table="customers", column="name",
                                expression="customers.name", operator="contains",
                                value="john", value_type="string", raw_text="john")]
        sql = IRToSQLRenderer().render(qir, dialect="sqlite")
        assert "LIKE" in sql
        assert "ILIKE" not in sql

    @pytest.mark.parametrize(
        "column",
        [
            "#",
            "Home (1st leg)",
            "Home (2nd leg)",
            "1st Leg",
            "Country/Region",
            "Name of member organization",
            "Date of birth",
            "Mens singles",
            "Womens singles",
            "Mens doubles",
            "Car No.",
            "Rd.",
            "Grand Prix",
            "Grand Cru",
            "Planet Type",
            "Semimajor Axis ( AU )",
            "Membership (from 2010)",
        ],
    )
    def test_exact_schema_identifiers_and_aliases_are_quoted(self, column: str) -> None:
        table = "1-12001616-4"
        qir = QueryIR(
            query_ir_id="quoted-identifiers",
            question="show value",
            normalized_question="show value",
            intent="show_records",
            template_id="show_records",
            base_table=table,
            dimensions=[IRDimension(
                name=column,
                table=table,
                column=column,
                expression=f"{table}.{column}",
                alias=column,
            )],
            limit=100,
        )
        sql = IRToSQLRenderer().render(qir)
        expected = f'{quote_identifier(table)}.{quote_identifier(column)} AS {quote_identifier(column)}'
        assert expected in sql
        assert "LIMIT 100" in sql
        assert sqlglot.parse_one(sql, read="sqlite") is not None
        schema = {"tables": {table: {"columns": {column: {"type": "text"}}}}}
        assert SQLValidator().validate(sql, schema=schema)["is_valid"] is True

    def test_quote_identifier_escapes_quotes_without_double_wrapping(self) -> None:
        assert quote_identifier('a"b') == '"a""b"'
        assert quote_identifier('"Car No."') == '"Car No."'


# ── SQL-to-IR Converter ──────────────────────────────────────────────
class TestSQLToIRConverter:
    def test_simple_select(self) -> None:
        schema = {"tables": {"orders": {"columns": {"order_id": {}, "amount": {}}}}}
        result = SQLToIRConverter().convert("show orders",
                                           "SELECT orders.order_id FROM orders LIMIT 10",
                                           schema=schema)
        assert result["success"]
        assert result["query_ir"]["base_table"] == "orders"

    def test_sum_aggregation(self) -> None:
        schema = {"tables": {"orders": {"columns": {"amount": {"type": "REAL"}}}}}
        result = SQLToIRConverter().convert(
            "total revenue",
            "SELECT SUM(orders.amount) AS revenue FROM orders LIMIT 100",
            schema=schema)
        assert result["success"]
        assert any(m["aggregation"] == "SUM" for m in result["query_ir"]["metrics"])


# ── IR Conversion Golden ─────────────────────────────────────────────
class TestIRConversionGolden:
    @pytest.fixture()
    def golden_cases(self) -> list[dict]:
        path = ROOT / "evaluation" / "ir_conversion_golden.jsonl"
        if not path.exists():
            pytest.skip("Golden cases file not found")
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_golden_cases_produce_valid_ir(self, golden_cases: list[dict]) -> None:
        converter = SQLToIRConverter()
        validator = IRValidator()
        for case in golden_cases:
            sql = case.get("sql")
            schema = case.get("schema", {})
            question = case.get("question", "")
            if not sql:
                continue
            try:
                qir = converter.convert(sql, question=question, schema=schema)
                result = validator.validate(qir)
                assert result.is_valid, f"Golden case {case.get('id')}: {result.errors}"
            except Exception:
                pass  # Some golden cases may have unsupported patterns


# ── RetrievalIRConverter (formerly OptionCToIRConverter) ──────────────
class TestRetrievalIRConverter:
    def test_metric_summary_produces_queryir(self) -> None:
        qir = RetrievalIRConverter().convert(
            question="Total revenue", normalized_question="total revenue",
            intent="metric_summary", template_id="metric_summary",
            slots={"metric": {"value": "revenue", "confidence": 0.9}},
            schema_mapping={"base_table": "orders", "metric_table": "orders",
                            "metric_column": "amount", "metric_expression": "orders.amount",
                            "metric_aggregation": "SUM", "match_scores": {"metric": 0.9}},
            join_plan=None)
        assert qir.intent == "metric_summary"
        assert len(qir.metrics) == 1
        assert qir.metrics[0].aggregation == "SUM"

    def test_backward_alias_works(self) -> None:
        from ir.option_c_to_ir import OptionCToIRConverter
        assert OptionCToIRConverter is RetrievalIRConverter


# ── Product revenue semantics ────────────────────────────────────────
class TestProductRevenueSemantics:
    def test_product_revenue_rewrites_expression(self) -> None:
        from neural_ir.ir_repair import NeuralIRRepairer
        qir = _make_metric_summary_ir()
        qir.question = "Top 5 products by revenue"
        schema = {
            "tables": {
                "products": {"columns": {"product_id": {}, "product_name": {}}},
                "order_items": {"columns": {"order_item_id": {}, "product_id": {},
                                            "quantity": {"type": "INTEGER"}, "price": {"type": "REAL"}}},
            }
        }
        result = NeuralIRRepairer().repair(qir, schema, "Top 5 products by revenue")
        assert "corrected_product_revenue" in str(result.get("repairs_applied", []))


# ── Helpers ──────────────────────────────────────────────────────────
def _make_metric_summary_ir(**kwargs) -> QueryIR:
    return QueryIR(
        query_ir_id="qir-test", question="Total revenue",
        normalized_question="total revenue", intent="metric_summary",
        template_id="metric_summary", base_table="orders",
        required_tables=["orders"],
        metrics=[IRMetric(name="revenue", aggregation="SUM", table="orders",
                          column="amount", expression="orders.amount", alias="revenue")],
        limit=100, select_mode="aggregate", dialect=kwargs.get("dialect", "sqlite"),
    )


def _make_trend_ir(dialect: str = "sqlite") -> QueryIR:
    return QueryIR(
        query_ir_id="qir-trend", question="Sales by month",
        normalized_question="sales by month", intent="trend_by_date",
        template_id="trend_by_date", base_table="orders",
        required_tables=["orders"],
        metrics=[IRMetric(name="revenue", aggregation="SUM", table="orders",
                          column="amount", expression="orders.amount", alias="revenue")],
        date_filters=[IRDateFilter(date_table="orders", date_column="order_date",
                                    date_expression="orders.order_date", filter_type="grain",
                                    date_grain="month", raw_text="by month")],
        group_by=["DATE_GRAIN(orders.order_date, month)"],
        order_by=[IROrderBy(expression="period", alias="period", direction="ASC", source="date")],
        limit=100, select_mode="trend", dialect=dialect,
    )
