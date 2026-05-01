"""
Build a PH-localized SQLite database from the Kaggle e-commerce churn dataset.

Reads:  datasets/data_ecommerce_customer_churn.csv  (3,941 rows with proper churn labels)
Writes: data/ecommerce_churn.db  (customers + orders tables)

Customer identity (Filipino names, PH cities) is synthesized.
Order history is synthesized to be consistent with each customer's Kaggle features.
Olist distributions are used for realistic status/payment/review sampling.
"""

import hashlib
import math
import random
import sqlite3
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_CSV = Path(__file__).resolve().parent.parent / "datasets" / "data_ecommerce_customer_churn.csv"
DST_DB = Path(__file__).resolve().parent.parent / "data" / "ecommerce_churn.db"

REFERENCE_DATE = date(2025, 3, 1)

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

STATUS_CHOICES = ["Delivered", "Shipped", "Cancelled", "Unavailable", "Invoiced", "Processing", "Created"]
STATUS_WEIGHTS = [0.970, 0.011, 0.006, 0.006, 0.003, 0.003, 0.001]

PAYMENT_CHOICES = ["Credit Card", "Boleto", "Voucher", "Debit Card"]
PAYMENT_WEIGHTS = [0.740, 0.190, 0.056, 0.015]

INSTALLMENT_CHOICES = [1, 2, 3, 4, 5, 6, 7, 8, 10]
INSTALLMENT_WEIGHTS = [0.52, 0.12, 0.10, 0.07, 0.05, 0.04, 0.02, 0.04, 0.05]

CATEGORY_MAP = {
    "Laptop & Accessory": ["computers", "computers_accessories", "tablets_printing_image", "electronics"],
    "Mobile Phone": ["telephony", "fixed_telephony", "electronics"],
    "Mobile": ["telephony", "fixed_telephony", "electronics"],
    "Fashion": ["fashion_bags_accessories", "fashion_shoes", "fashion_female_clothing",
                "fashion_male_clothing", "fashion_sport", "fashion_underwear_beach"],
    "Grocery": ["food", "food_drink", "drinks"],
    "Others": ["housewares", "health_beauty", "sports_leisure", "bed_bath_table",
               "cool_stuff", "watches_gifts", "perfumery", "baby", "toys",
               "garden_tools", "auto", "stationery"],
}

ALL_CATEGORIES = sorted({cat for cats in CATEGORY_MAP.values() for cat in cats})


def _seed_from_id(id_str: str) -> int:
    return int(hashlib.md5(id_str.encode()).hexdigest()[:8], 16)


def generate_customer_id(index: int) -> str:
    return hashlib.sha256(f"kaggle-churn-{index}".encode()).hexdigest()[:32]


def impute_nulls(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Tenure", "WarehouseToHome", "DaySinceLastOrder"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val).astype(int)
    return df


def build_customers(df: pd.DataFrame) -> pd.DataFrame:
    print("  Building customers...")

    city_rng = random.Random(42)
    records = []

    for idx, row in df.iterrows():
        cid = generate_customer_id(idx)
        r = random.Random(_seed_from_id(cid))
        name = f"{r.choice(FIRST_NAMES)} {r.choice(LAST_NAMES)}"
        city = city_rng.choice(PH_CITIES)
        signup_date = REFERENCE_DATE - timedelta(days=int(row["Tenure"]) * 30)

        records.append({
            "customer_id": cid,
            "name": name,
            "city": city,
            "signup_date": signup_date.isoformat(),
            "tenure_months": int(row["Tenure"]),
            "warehouse_to_home": int(row["WarehouseToHome"]),
            "num_devices": int(row["NumberOfDeviceRegistered"]),
            "preferred_category": row["PreferedOrderCat"],
            "satisfaction_score": int(row["SatisfactionScore"]),
            "marital_status": row["MaritalStatus"],
            "num_addresses": int(row["NumberOfAddress"]),
            "complain": int(row["Complain"]),
            "days_since_last_order": int(row["DaySinceLastOrder"]),
            "cashback_amount": float(row["CashbackAmount"]),
            "churn": int(row["Churn"]),
        })

    customers = pd.DataFrame(records)
    print(f"    {len(customers)} customers")
    return customers


def synthesize_orders(customers: pd.DataFrame) -> pd.DataFrame:
    print("  Synthesizing orders...")

    all_orders = []

    for _, cust in customers.iterrows():
        cid = cust["customer_id"]
        rng = random.Random(_seed_from_id(cid) + 1)
        np_rng = np.random.RandomState(_seed_from_id(cid) + 2)

        tenure = cust["tenure_months"]
        days_since = cust["days_since_last_order"]
        cashback = cust["cashback_amount"]
        pref_cat = cust["preferred_category"]
        satisfaction = cust["satisfaction_score"]

        # Number of orders
        if tenure == 0:
            num_orders = 1
        else:
            jitter = max(0.5, rng.gauss(1.0, 0.3))
            num_orders = max(1, min(20, round(tenure / 3 * jitter)))

        # Date boundaries
        last_order_date = REFERENCE_DATE - timedelta(days=days_since)
        signup_date = REFERENCE_DATE - timedelta(days=tenure * 30)

        if signup_date >= last_order_date:
            signup_date = last_order_date - timedelta(days=max(1, num_orders))

        # Order dates
        if num_orders == 1:
            order_dates = [last_order_date]
        else:
            span_days = (last_order_date - signup_date).days
            if span_days <= 0:
                span_days = num_orders
            points = sorted(np_rng.uniform(0, span_days, num_orders - 1).tolist())
            order_dates = [signup_date + timedelta(days=int(p)) for p in points]
            order_dates.append(last_order_date)
            order_dates.sort()

        # Order amounts
        base_amount = max(cashback * 6.0, 200.0)
        log_mean = math.log(base_amount)

        # Category pool
        preferred_cats = CATEGORY_MAP.get(pref_cat, CATEGORY_MAP["Others"])

        for i, od in enumerate(order_dates):
            order_id = hashlib.sha256(f"{cid}-order-{i}".encode()).hexdigest()[:32]

            amount = max(91, min(15000, np_rng.lognormal(log_mean, 0.4)))

            # 70% preferred, 30% random
            if rng.random() < 0.7:
                category = rng.choice(preferred_cats)
            else:
                category = rng.choice(ALL_CATEGORIES)

            status = rng.choices(STATUS_CHOICES, weights=STATUS_WEIGHTS, k=1)[0]

            payment = rng.choices(PAYMENT_CHOICES, weights=PAYMENT_WEIGHTS, k=1)[0]

            # Review score (only for delivered, ~90% chance)
            review_score = None
            if status == "Delivered" and rng.random() < 0.9:
                weights = [math.exp(-abs(s - satisfaction)) for s in range(1, 6)]
                total_w = sum(weights)
                weights = [w / total_w for w in weights]
                review_score = rng.choices([1, 2, 3, 4, 5], weights=weights, k=1)[0]

            all_orders.append({
                "order_id": order_id,
                "customer_id": cid,
                "order_date": datetime(od.year, od.month, od.day,
                                       rng.randint(6, 23), rng.randint(0, 59), rng.randint(0, 59)).isoformat(),
                "total_amount": round(amount, 2),
                "status": status,
                "payment_type": payment,
                "review_score": review_score,
            })

    orders = pd.DataFrame(all_orders)
    print(f"    {len(orders)} orders")
    return orders


def main():
    print(f"Source: {KAGGLE_CSV}")
    print(f"Dest:   {DST_DB}")

    df = pd.read_csv(KAGGLE_CSV)
    print(f"  Loaded {len(df)} rows from Kaggle CSV")

    df = impute_nulls(df)

    customers = build_customers(df)
    orders = synthesize_orders(customers)

    DST_DB.parent.mkdir(parents=True, exist_ok=True)
    if DST_DB.exists():
        DST_DB.unlink()

    dst = sqlite3.connect(str(DST_DB))
    customers.to_sql("customers", dst, index=False)
    orders.to_sql("orders", dst, index=False)

    dst.execute("CREATE INDEX idx_orders_customer ON orders(customer_id)")
    dst.execute("CREATE INDEX idx_customers_id ON customers(customer_id)")
    dst.commit()
    dst.close()

    print("\nDone! Verifying...")
    conn = sqlite3.connect(str(DST_DB))
    for table in ["customers", "orders"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    churned = conn.execute("SELECT SUM(churn) FROM customers").fetchone()[0]
    print(f"  Churned customers: {churned}")

    cities = conn.execute("SELECT COUNT(DISTINCT city) FROM customers").fetchone()[0]
    print(f"  Distinct cities: {cities}")

    nulls = conn.execute(
        "SELECT COUNT(*) FROM customers WHERE tenure_months IS NULL OR warehouse_to_home IS NULL OR days_since_last_order IS NULL"
    ).fetchone()[0]
    print(f"  Null imputed columns: {nulls}")

    avg_orders = conn.execute(
        "SELECT AVG(cnt) FROM (SELECT COUNT(*) as cnt FROM orders GROUP BY customer_id)"
    ).fetchone()[0]
    print(f"  Avg orders per customer: {avg_orders:.1f}")

    avg_amount = conn.execute("SELECT AVG(total_amount) FROM orders").fetchone()[0]
    print(f"  Avg order amount (PHP): {avg_amount:.2f}")

    sample_cities = [r[0] for r in conn.execute(
        "SELECT DISTINCT city FROM customers LIMIT 5"
    ).fetchall()]
    print(f"  Sample cities: {sample_cities}")

    conn.close()


if __name__ == "__main__":
    main()
