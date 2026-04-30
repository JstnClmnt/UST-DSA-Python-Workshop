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
    layout="wide",
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


def get_city_metrics(conn):
    query = """
    SELECT
        c.city,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(o.order_id) AS total_orders,
        ROUND(SUM(o.total_amount), 2) AS total_revenue,
        ROUND(AVG(o.total_amount), 2) AS avg_order_value,
        SUM(CASE WHEN o.status = 'Cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
        ROUND(SUM(CASE WHEN o.status = 'Cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(o.order_id), 2) AS cancellation_rate
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.city
    ORDER BY c.city
    """
    return pd.read_sql(query, conn)


def get_city_churn_rates(conn):
    query = """
    SELECT
        c.city,
        COUNT(*) AS total_customers,
        SUM(CASE WHEN o.days_since_last_order > 90 THEN 1 ELSE 0 END) AS churned_customers,
        ROUND(SUM(CASE WHEN o.days_since_last_order > 90 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS churn_rate
    FROM customers c
    JOIN (
        SELECT customer_id, MIN(days_since_last_order) AS days_since_last_order
        FROM orders
        GROUP BY customer_id
    ) o ON c.customer_id = o.customer_id
    GROUP BY c.city
    ORDER BY churn_rate DESC
    """
    return pd.read_sql(query, conn)


def explain_city_insights(city_data: str) -> str:
    client = anthropic.Anthropic()
    prompt = f"""Here is city-level analytics data for an e-commerce business in the Philippines:

{city_data}

Analyze the data and provide:
1. Which cities need the most attention and why
2. 2-3 actionable recommendations for reducing churn

Be concise and specific. 4-5 sentences max."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# --- UI ---

st.title("📊 Customer Churn Predictor")
st.markdown("AI-powered churn risk assessment — by customer or by city.")

model = load_model()
conn = get_db()

tab1, tab2 = st.tabs(["Customer Lookup", "City Analytics"])

with tab1:
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

            st.subheader(profile.iloc[0]["name"])
            col1, col2, col3 = st.columns(3)
            col1.metric("City", profile.iloc[0]["city"])
            col2.metric("Member Since", profile.iloc[0]["signup_date"])
            col3.metric("Total Orders", int(row["total_orders"]))

            st.divider()

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

            st.divider()
            st.subheader("Customer Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Days Since Last Order", int(row["days_since_last_order"]))
            c2.metric("Total Orders", int(row["total_orders"]))
            c3.metric("Avg Order Value", f"₱{row['avg_order_value']:,.2f}")
            c4.metric("Cancellation Rate", f"{row['cancellation_rate']:.0%}")

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

with tab2:
    metrics_df = get_city_metrics(conn)
    churn_df = get_city_churn_rates(conn)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cities", len(churn_df))
    col2.metric("Avg Churn Rate", f"{churn_df['churn_rate'].mean():.1f}%")
    col3.metric("Total Revenue", f"₱{metrics_df['total_revenue'].sum():,.0f}")
    col4.metric("Total Orders", f"{metrics_df['total_orders'].sum():,}")

    st.divider()

    st.subheader("Churn Rate by City (%)")
    st.bar_chart(churn_df.set_index("city")["churn_rate"])

    st.subheader("Total Revenue by City (₱)")
    st.bar_chart(metrics_df.set_index("city")["total_revenue"])

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Avg Order Value by City (₱)")
        st.bar_chart(metrics_df.set_index("city")["avg_order_value"])
    with col_right:
        st.subheader("Cancellation Rate by City (%)")
        st.bar_chart(metrics_df.set_index("city")["cancellation_rate"])

    st.divider()
    st.subheader("🤖 AI City Insights")
    if st.button("Generate City Insights"):
        with st.spinner("Analyzing city data..."):
            merged = churn_df.merge(metrics_df, on="city")
            summary = merged[["city", "churn_rate", "total_revenue", "avg_order_value", "cancellation_rate"]].to_string(index=False)
            insights = explain_city_insights(summary)
        st.info(insights)
