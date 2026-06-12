from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import (
    Column,
    Date,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "sample_retail.db"


def build_database(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    engine = create_engine(f"sqlite:///{db_path.as_posix()}", future=True)
    metadata = MetaData()

    customers = Table(
        "customers",
        metadata,
        Column("customer_id", Integer, primary_key=True),
        Column("customer_name", String, nullable=False),
        Column("segment", String, nullable=False),
        Column("city", String, nullable=False),
        Column("state", String, nullable=False),
        Column("region", String, nullable=False),
    )
    products = Table(
        "products",
        metadata,
        Column("product_id", Integer, primary_key=True),
        Column("product_name", String, nullable=False),
        Column("category", String, nullable=False),
        Column("brand", String, nullable=False),
        Column("unit_price", Float, nullable=False),
    )
    stores = Table(
        "stores",
        metadata,
        Column("store_id", Integer, primary_key=True),
        Column("store_name", String, nullable=False),
        Column("city", String, nullable=False),
        Column("state", String, nullable=False),
        Column("region", String, nullable=False),
    )
    sales_reps = Table(
        "sales_reps",
        metadata,
        Column("rep_id", Integer, primary_key=True),
        Column("rep_name", String, nullable=False),
        Column("region", String, nullable=False),
    )
    orders = Table(
        "orders",
        metadata,
        Column("order_id", Integer, primary_key=True),
        Column("customer_id", Integer, ForeignKey("customers.customer_id"), nullable=False),
        Column("store_id", Integer, ForeignKey("stores.store_id"), nullable=False),
        Column("rep_id", Integer, ForeignKey("sales_reps.rep_id"), nullable=False),
        Column("order_date", Date, nullable=False),
        Column("status", String, nullable=False),
        Column("amount", Float, nullable=False),
    )
    order_items = Table(
        "order_items",
        metadata,
        Column("order_item_id", Integer, primary_key=True),
        Column("order_id", Integer, ForeignKey("orders.order_id"), nullable=False),
        Column("product_id", Integer, ForeignKey("products.product_id"), nullable=False),
        Column("quantity", Integer, nullable=False),
        Column("unit_price", Float, nullable=False),
        Column("discount", Float, nullable=False),
        Column("profit", Float, nullable=False),
    )
    metadata.create_all(engine)

    customer_rows = [
        (1, "Acme Corp", "Enterprise", "San Francisco", "CA", "West"),
        (2, "Bluebird Retail", "SMB", "Los Angeles", "CA", "West"),
        (3, "Canyon Goods", "Mid-Market", "Austin", "TX", "South"),
        (4, "Delta Market", "Enterprise", "New York", "NY", "East"),
        (5, "Evergreen Co", "SMB", "Seattle", "WA", "West"),
        (6, "Futura Stores", "Mid-Market", "Chicago", "IL", "Central"),
        (7, "Granite Supply", "Enterprise", "Denver", "CO", "West"),
        (8, "Harbor Wholesale", "SMB", "Boston", "MA", "East"),
        (9, "Iris Direct", "Mid-Market", "Miami", "FL", "South"),
        (10, "Jupiter Partners", "Enterprise", "Dallas", "TX", "South"),
    ]
    product_rows = [
        (1, "Laptop Pro 14", "Electronics", "Northstar", 1299.0),
        (2, "Wireless Mouse", "Electronics", "Northstar", 39.0),
        (3, "Standing Desk", "Furniture", "WorkWell", 499.0),
        (4, "Office Chair", "Furniture", "WorkWell", 229.0),
        (5, "Coffee Beans 5lb", "Grocery", "Roastly", 58.0),
        (6, "Sparkling Water Case", "Grocery", "PureDrop", 24.0),
        (7, "Notebook Pack", "Office Supplies", "PaperTrail", 16.0),
        (8, "Ink Cartridge", "Office Supplies", "PrintCo", 74.0),
    ]
    store_rows = [
        (1, "SF Flagship", "San Francisco", "CA", "West"),
        (2, "Austin Central", "Austin", "TX", "South"),
        (3, "NY Downtown", "New York", "NY", "East"),
        (4, "Chicago Loop", "Chicago", "IL", "Central"),
    ]
    rep_rows = [
        (1, "Maya Singh", "West"),
        (2, "Luis Garcia", "South"),
        (3, "Nora Chen", "East"),
        (4, "Sam Patel", "Central"),
    ]

    order_rows = []
    item_rows = []
    order_id = 1
    item_id = 1
    start = date(2023, 1, 15)
    for idx in range(72):
        customer_id = (idx % len(customer_rows)) + 1
        store_id = (idx % len(store_rows)) + 1
        rep_id = (idx % len(rep_rows)) + 1
        order_date = start + timedelta(days=idx * 17)
        product_a = (idx % len(product_rows)) + 1
        product_b = ((idx + 3) % len(product_rows)) + 1
        quantity_a = (idx % 4) + 1
        quantity_b = ((idx + 1) % 3) + 1
        price_a = product_rows[product_a - 1][4]
        price_b = product_rows[product_b - 1][4]
        discount_a = round(price_a * quantity_a * (0.03 if idx % 5 == 0 else 0.0), 2)
        discount_b = round(price_b * quantity_b * (0.05 if idx % 7 == 0 else 0.0), 2)
        amount = round(price_a * quantity_a + price_b * quantity_b - discount_a - discount_b, 2)
        status = ["completed", "shipped", "pending", "cancelled"][idx % 4]
        order_rows.append(
            {
                "order_id": order_id,
                "customer_id": customer_id,
                "store_id": store_id,
                "rep_id": rep_id,
                "order_date": order_date,
                "status": status,
                "amount": amount,
            }
        )
        item_rows.append(
            {
                "order_item_id": item_id,
                "order_id": order_id,
                "product_id": product_a,
                "quantity": quantity_a,
                "unit_price": price_a,
                "discount": discount_a,
                "profit": round(price_a * quantity_a * 0.22 - discount_a, 2),
            }
        )
        item_id += 1
        item_rows.append(
            {
                "order_item_id": item_id,
                "order_id": order_id,
                "product_id": product_b,
                "quantity": quantity_b,
                "unit_price": price_b,
                "discount": discount_b,
                "profit": round(price_b * quantity_b * 0.18 - discount_b, 2),
            }
        )
        item_id += 1
        order_id += 1

    with engine.begin() as conn:
        conn.execute(customers.insert(), [dict(zip(["customer_id", "customer_name", "segment", "city", "state", "region"], row)) for row in customer_rows])
        conn.execute(products.insert(), [dict(zip(["product_id", "product_name", "category", "brand", "unit_price"], row)) for row in product_rows])
        conn.execute(stores.insert(), [dict(zip(["store_id", "store_name", "city", "state", "region"], row)) for row in store_rows])
        conn.execute(sales_reps.insert(), [dict(zip(["rep_id", "rep_name", "region"], row)) for row in rep_rows])
        conn.execute(orders.insert(), order_rows)
        conn.execute(order_items.insert(), item_rows)


if __name__ == "__main__":
    build_database()
    print(f"Created {DB_PATH}")
