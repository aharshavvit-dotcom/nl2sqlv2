from __future__ import annotations

from ir.query_ir_models import IRDateFilter, IRDimension, IRFilter, IRJoin, IRMetric, IROrderBy, QueryIR


def make_v1_metric_summary() -> QueryIR:
    return QueryIR(
        query_ir_id="qir-v1-metric",
        question="Total revenue",
        normalized_question="total revenue",
        intent="metric_summary",
        template_id="metric_summary",
        base_table="orders",
        required_tables=["orders"],
        metrics=[IRMetric(name="revenue", aggregation="SUM", table="orders", column="amount", expression="orders.amount", alias="revenue")],
        limit=100,
        select_mode="aggregate",
    )


def make_v1_metric_by_dimension() -> QueryIR:
    return QueryIR(
        query_ir_id="qir-v1-metric-dim",
        question="Revenue by region",
        normalized_question="revenue by region",
        intent="metric_by_dimension",
        template_id="metric_by_dimension",
        base_table="orders",
        required_tables=["orders", "customers"],
        metrics=[IRMetric(name="revenue", aggregation="SUM", table="orders", column="amount", expression="orders.amount", alias="revenue")],
        dimensions=[IRDimension(name="region", table="customers", column="region", expression="customers.region", alias="region")],
        joins=[IRJoin(left_table="orders", left_column="customer_id", right_table="customers", right_column="customer_id", condition="orders.customer_id = customers.customer_id", path_order=0)],
        group_by=["customers.region"],
        order_by=[IROrderBy(expression="revenue", alias="revenue", direction="DESC", source="metric")],
        limit=100,
        select_mode="aggregate",
    )


def make_v1_count_by_dimension() -> QueryIR:
    return QueryIR(
        query_ir_id="qir-v1-count-dim",
        question="Count orders by status",
        normalized_question="count orders by status",
        intent="count_by_dimension",
        template_id="count_by_dimension",
        base_table="orders",
        required_tables=["orders"],
        metrics=[IRMetric(name="count", aggregation="COUNT", table=None, column="*", expression="*", alias="record_count")],
        dimensions=[IRDimension(name="status", table="orders", column="status", expression="orders.status", alias="status")],
        group_by=["orders.status"],
        order_by=[IROrderBy(expression="record_count", alias="record_count", direction="DESC", source="count")],
        limit=100,
        select_mode="count",
    )


def make_v1_filter() -> QueryIR:
    return QueryIR(
        query_ir_id="qir-v1-filter",
        question="Orders where status is completed",
        normalized_question="orders where status is completed",
        intent="simple_filter",
        template_id="simple_filter",
        base_table="orders",
        required_tables=["orders"],
        filters=[IRFilter(name="status", table="orders", column="status", expression="orders.status", operator="equals", value="completed")],
        dimensions=[IRDimension(name="order_id", table="orders", column="order_id", expression="orders.order_id", alias="order_id")],
        limit=100,
        select_mode="records",
    )


def make_v1_trend() -> QueryIR:
    return QueryIR(
        query_ir_id="qir-v1-trend",
        question="Sales by month",
        normalized_question="sales by month",
        intent="trend_by_date",
        template_id="trend_by_date",
        base_table="orders",
        required_tables=["orders"],
        metrics=[IRMetric(name="revenue", aggregation="SUM", table="orders", column="amount", expression="orders.amount", alias="revenue")],
        date_filters=[IRDateFilter(date_table="orders", date_column="order_date", date_expression="orders.order_date", filter_type="grain", date_grain="month", raw_text="by month")],
        group_by=["DATE_GRAIN(orders.order_date, month)"],
        order_by=[IROrderBy(expression="period", alias="period", direction="ASC", source="date")],
        limit=100,
        select_mode="trend",
    )


def make_v1_product_revenue() -> QueryIR:
    return QueryIR(
        query_ir_id="qir-v1-product-revenue",
        question="Top products by revenue",
        normalized_question="top products by revenue",
        intent="top_n_metric_by_dimension",
        template_id="top_n_metric_by_dimension",
        base_table="order_items",
        required_tables=["order_items", "products"],
        metrics=[IRMetric(name="revenue", aggregation="SUM", table=None, column=None, expression="order_items.quantity * order_items.price", alias="revenue")],
        dimensions=[IRDimension(name="product", table="products", column="product_name", expression="products.product_name", alias="product")],
        joins=[IRJoin(left_table="order_items", left_column="product_id", right_table="products", right_column="product_id", condition="order_items.product_id = products.product_id", path_order=0)],
        group_by=["products.product_name"],
        order_by=[IROrderBy(expression="revenue", alias="revenue", direction="DESC", source="metric")],
        limit=5,
        select_mode="aggregate",
    )


def supported_v1_examples() -> list[QueryIR]:
    return [
        make_v1_metric_summary(),
        make_v1_metric_by_dimension(),
        make_v1_count_by_dimension(),
        make_v1_filter(),
        make_v1_trend(),
        make_v1_product_revenue(),
    ]
