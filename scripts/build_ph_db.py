"""
Transform the raw Olist SQLite database into a PH-localized 3-table schema
for the UST-DSA workshop.

Input:  datasets/olist.sqlite
Output: data/ecommerce_ph.db  (customers, orders, products)
"""

import hashlib
import random
import sqlite3
from pathlib import Path

import pandas as pd

SRC_DB = Path(__file__).resolve().parent.parent / "datasets" / "olist.sqlite"
DST_DB = Path(__file__).resolve().parent.parent / "data" / "ecommerce_ph.db"

BRL_TO_PHP = 9.5

PH_CITIES = [
    "Manila", "Quezon City", "Makati", "Cebu City", "Davao City",
    "Pasig", "Taguig", "Antipolo", "Caloocan", "Las Piñas",
    "Mandaluyong", "Marikina", "Muntinlupa", "Parañaque", "San Juan",
    "Valenzuela", "Pasay", "Malabon", "Navotas", "Bacoor",
    "Iloilo City", "Cagayan de Oro", "Zamboanga City", "Bacolod",
]

FIRST_NAMES = [
    "Juan", "Maria", "Jose", "Ana", "Pedro", "Rosa", "Carlos", "Elena",
    "Miguel", "Sofia", "Rafael", "Isabel", "Antonio", "Luz", "Francisco",
    "Carmen", "Luis", "Teresa", "Ramon", "Gloria", "Manuel", "Nena",
    "Ricardo", "Fe", "Eduardo", "Corazon", "Roberto", "Lourdes",
    "Andres", "Cristina", "Marco", "Patricia", "Gabriel", "Josephine",
    "Daniel", "Angelica", "Paolo", "Jasmine", "Kenneth", "Nicole",
    "Mark", "Trisha", "John", "Kimberly", "James", "Rhea", "Kevin",
    "Czarina", "Adrian", "Bianca",
]

LAST_NAMES = [
    "Santos", "Reyes", "Cruz", "Bautista", "Del Rosario", "Gonzales",
    "Ramos", "Aquino", "Garcia", "Mendoza", "Torres", "Villanueva",
    "Dela Cruz", "Rivera", "Fernandez", "Lopez", "Martinez", "Flores",
    "Castillo", "Soriano", "Tan", "Lim", "Sy", "Chua", "Go",
    "Ong", "Co", "Yu", "Ang", "Tiu", "Hernandez", "Pascual",
    "Aguilar", "Domingo", "Salvador", "Navarro", "Santiago", "De Leon",
    "Dizon", "Mercado",
]

STATUS_MAP = {
    "delivered": "Delivered",
    "shipped": "Shipped",
    "canceled": "Cancelled",
    "unavailable": "Unavailable",
    "invoiced": "Invoiced",
    "processing": "Processing",
    "created": "Created",
    "approved": "Approved",
}


def _seed_from_id(id_str: str) -> int:
    return int(hashlib.md5(id_str.encode()).hexdigest()[:8], 16)


def build_customers(src: sqlite3.Connection) -> pd.DataFrame:
    print("  Building customers...")

    customers = pd.read_sql(
        """
        SELECT
            c.customer_unique_id AS customer_id,
            MIN(o.order_purchase_timestamp) AS signup_date,
            c.customer_city AS orig_city
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_unique_id
        """,
        src,
    )

    customers["signup_date"] = pd.to_datetime(customers["signup_date"]).dt.date

    rng = random.Random(42)
    names = []
    cities = []
    for cid in customers["customer_id"]:
        seed = _seed_from_id(cid)
        r = random.Random(seed)
        names.append(f"{r.choice(FIRST_NAMES)} {r.choice(LAST_NAMES)}")
        cities.append(rng.choice(PH_CITIES))

    customers["name"] = names
    customers["city"] = cities
    customers = customers[["customer_id", "name", "city", "signup_date"]]

    print(f"    {len(customers)} customers")
    return customers


def build_orders(src: sqlite3.Connection) -> pd.DataFrame:
    print("  Building orders...")

    orders = pd.read_sql(
        """
        SELECT
            o.order_id,
            c.customer_unique_id AS customer_id,
            o.order_purchase_timestamp AS order_date,
            o.order_status AS status,
            COALESCE(oi.total, 0) AS total_amount_brl
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        LEFT JOIN (
            SELECT order_id, SUM(price + freight_value) AS total
            FROM order_items
            GROUP BY order_id
        ) oi ON o.order_id = oi.order_id
        """,
        src,
    )

    orders["order_date"] = pd.to_datetime(orders["order_date"])
    orders["total_amount"] = (orders["total_amount_brl"] * BRL_TO_PHP).round(2)
    orders["status"] = orders["status"].map(STATUS_MAP).fillna("Other")

    ref_date = orders["order_date"].max()
    last_order = orders.groupby("customer_id")["order_date"].transform("max")
    orders["days_since_last_order"] = (ref_date - last_order).dt.days

    orders = orders[
        ["order_id", "customer_id", "order_date", "total_amount", "status",
         "days_since_last_order"]
    ]

    print(f"    {len(orders)} orders")
    return orders


def build_products(src: sqlite3.Connection) -> pd.DataFrame:
    print("  Building products...")

    products = pd.read_sql(
        """
        SELECT
            p.product_id,
            COALESCE(t.product_category_name_english, p.product_category_name, 'other')
                AS category,
            oi.median_price
        FROM products p
        LEFT JOIN product_category_name_translation t
            ON p.product_category_name = t.product_category_name
        LEFT JOIN (
            SELECT product_id, AVG(price) AS median_price
            FROM order_items
            GROUP BY product_id
        ) oi ON p.product_id = oi.product_id
        """,
        src,
    )

    products["price"] = (products["median_price"].fillna(0) * BRL_TO_PHP).round(2)

    cat_counters: dict[str, int] = {}
    names = []
    for cat in products["category"]:
        cat_counters[cat] = cat_counters.get(cat, 0) + 1
        pretty = cat.replace("_", " ").title()
        names.append(f"{pretty} #{cat_counters[cat]}")
    products["name"] = names

    products = products[["product_id", "name", "category", "price"]]

    print(f"    {len(products)} products")
    return products


def main():
    print(f"Source: {SRC_DB}")
    print(f"Dest:   {DST_DB}")

    DST_DB.parent.mkdir(parents=True, exist_ok=True)
    if DST_DB.exists():
        DST_DB.unlink()

    src = sqlite3.connect(str(SRC_DB))

    customers = build_customers(src)
    orders = build_orders(src)
    products = build_products(src)

    src.close()

    dst = sqlite3.connect(str(DST_DB))
    customers.to_sql("customers", dst, index=False)
    orders.to_sql("orders", dst, index=False)
    products.to_sql("products", dst, index=False)

    dst.execute("CREATE INDEX idx_orders_customer ON orders(customer_id)")
    dst.execute("CREATE INDEX idx_customers_id ON customers(customer_id)")
    dst.execute("CREATE INDEX idx_products_id ON products(product_id)")
    dst.commit()
    dst.close()

    print("\nDone! Verifying...")
    conn = sqlite3.connect(str(DST_DB))
    for table in ["customers", "orders", "products"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")
    sample_cities = [r[0] for r in conn.execute(
        "SELECT DISTINCT city FROM customers LIMIT 5"
    ).fetchall()]
    print(f"  Sample cities: {sample_cities}")
    avg_amount = conn.execute("SELECT AVG(total_amount) FROM orders").fetchone()[0]
    print(f"  Avg order amount (PHP): {avg_amount:.2f}")
    conn.close()


if __name__ == "__main__":
    main()
