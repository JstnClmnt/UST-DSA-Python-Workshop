import sqlite3

import anthropic
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
)


@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")


@st.cache_resource
def get_db():
    return sqlite3.connect("data/ecommerce_ph.db", check_same_thread=False)


def get_customer_ids(conn):
    query = "SELECT customer_id, name FROM customers ORDER BY name LIMIT 200"
    return pd.read_sql(query, conn)


def get_customer_features(conn, customer_id):
    query = """
    SELECT
        customer_id,
        COUNT(*) AS total_orders,
        AVG(total_amount) AS avg_order_value,
        MIN(days_since_last_order) AS days_since_last_order,
        SUM(CASE WHEN status = 'Cancelled' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS cancellation_rate
    FROM orders
    WHERE customer_id = ?
    GROUP BY customer_id
    """
    return pd.read_sql(query, conn, params=[customer_id])


def get_customer_profile(conn, customer_id):
    query = "SELECT * FROM customers WHERE customer_id = ?"
    return pd.read_sql(query, conn, params=[customer_id])


def explain_churn_risk(customer_data: dict, risk_score: float) -> str:
    client = anthropic.Anthropic()
    prompt = f"""A customer has a {risk_score:.0%} churn risk score.
Their profile:
- Days since last order: {customer_data['days_since_last_order']}
- Total orders: {customer_data['total_orders']}
- Avg order value: ₱{customer_data['avg_order_value']:,.2f}
- Cancellation rate: {customer_data['cancellation_rate']:.0%}

Write a brief, actionable recommendation for the business owner.
Be concise and specific. 2-3 sentences max."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# --- UI ---

st.title("📊 Customer Churn Predictor")
st.markdown("Look up a customer and get an AI-powered churn risk assessment.")

model = load_model()
conn = get_db()

customer_list = get_customer_ids(conn)
options = dict(zip(
    customer_list["customer_id"],
    customer_list.apply(lambda r: f"{r['name']} ({r['customer_id'][:8]}…)", axis=1),
))

selected_id = st.selectbox(
    "Select a customer",
    options=list(options.keys()),
    format_func=lambda x: options[x],
)

if selected_id:
    profile = get_customer_profile(conn, selected_id)
    features = get_customer_features(conn, selected_id)

    if features.empty:
        st.warning("No orders found for this customer.")
    else:
        row = features.iloc[0]
        feature_values = row[["days_since_last_order", "total_orders",
                              "avg_order_value", "cancellation_rate"]]
        risk_score = model.predict_proba(feature_values.values.reshape(1, -1))[0][1]

        # Customer profile
        st.subheader(profile.iloc[0]["name"])
        col1, col2, col3 = st.columns(3)
        col1.metric("City", profile.iloc[0]["city"])
        col2.metric("Member Since", profile.iloc[0]["signup_date"])
        col3.metric("Total Orders", int(row["total_orders"]))

        st.divider()

        # Churn risk
        risk_pct = risk_score * 100
        if risk_pct >= 70:
            risk_color = "🔴"
            risk_label = "High Risk"
        elif risk_pct >= 40:
            risk_color = "🟡"
            risk_label = "Medium Risk"
        else:
            risk_color = "🟢"
            risk_label = "Low Risk"

        st.subheader(f"{risk_color} Churn Risk: {risk_pct:.1f}%")
        st.caption(risk_label)
        st.progress(min(risk_score, 1.0))

        # Feature details
        st.divider()
        st.subheader("Customer Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Days Since Last Order", int(row["days_since_last_order"]))
        c2.metric("Total Orders", int(row["total_orders"]))
        c3.metric("Avg Order Value", f"₱{row['avg_order_value']:,.2f}")
        c4.metric("Cancellation Rate", f"{row['cancellation_rate']:.0%}")

        # AI explanation
        st.divider()
        st.subheader("🤖 AI Recommendation")
        with st.spinner("Generating recommendation..."):
            customer_data = {
                "days_since_last_order": int(row["days_since_last_order"]),
                "total_orders": int(row["total_orders"]),
                "avg_order_value": float(row["avg_order_value"]),
                "cancellation_rate": float(row["cancellation_rate"]),
            }
            explanation = explain_churn_risk(customer_data, risk_score)
        st.info(explanation)
