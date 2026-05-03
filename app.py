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
    artifact = joblib.load("models/churn_model.pkl")
    return (
        artifact["model"],
        artifact["feature_columns"],
        artifact["impute_medians"],
        artifact.get("scaler"),
        artifact.get("model_name", "Unknown"),
    )


@st.cache_resource
def get_db():
    return sqlite3.connect(
        "file:data/ecommerce_churn.db?mode=ro", uri=True, check_same_thread=False
    )


def get_customer_ids(conn):
    query = "SELECT customer_id, name FROM customers ORDER BY name LIMIT 200"
    return pd.read_sql(query, conn)


def get_customer_profile(conn, customer_id):
    query = "SELECT * FROM customers WHERE customer_id = ?"
    return pd.read_sql(query, conn, params=[customer_id])


def get_customer_orders(conn, customer_id):
    query = """
    SELECT order_date, total_amount, status, payment_type, review_score
    FROM orders WHERE customer_id = ? ORDER BY order_date DESC
    """
    return pd.read_sql(query, conn, params=[customer_id])


def build_feature_vector(profile_row, feature_columns):
    row_data = {
        "Tenure": profile_row["tenure_months"],
        "WarehouseToHome": profile_row["warehouse_to_home"],
        "NumberOfDeviceRegistered": profile_row["num_devices"],
        "SatisfactionScore": profile_row["satisfaction_score"],
        "NumberOfAddress": profile_row["num_addresses"],
        "Complain": profile_row["complain"],
        "DaySinceLastOrder": profile_row["days_since_last_order"],
        "CashbackAmount": profile_row["cashback_amount"],
        "PreferedOrderCat": profile_row["preferred_category"],
        "MaritalStatus": profile_row["marital_status"],
    }
    df = pd.DataFrame([row_data])
    df = pd.get_dummies(df, columns=["PreferedOrderCat", "MaritalStatus"], drop_first=True)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def explain_churn_risk(customer_data: dict, risk_score: float) -> str:
    client = anthropic.Anthropic()
    prompt = f"""A customer has a {risk_score:.0%} churn risk score.
Their profile:
- Tenure: {customer_data['tenure_months']} months
- Satisfaction Score: {customer_data['satisfaction_score']}/5
- Preferred Category: {customer_data['preferred_category']}
- Marital Status: {customer_data['marital_status']}
- Days Since Last Order: {customer_data['days_since_last_order']}
- Cashback Amount: ₱{customer_data['cashback_amount']:,.2f}
- Filed Complaint: {'Yes' if customer_data['complain'] else 'No'}
- Devices Registered: {customer_data['num_devices']}
- Addresses on File: {customer_data['num_addresses']}
- Warehouse Distance: {customer_data['warehouse_to_home']} km

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
        ROUND(AVG(c.satisfaction_score), 1) AS avg_satisfaction
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.city
    ORDER BY c.city
    """
    return pd.read_sql(query, conn)


def get_city_churn_rates(conn):
    query = """
    SELECT
        city,
        COUNT(*) AS total_customers,
        SUM(churn) AS churned_customers,
        ROUND(SUM(churn) * 100.0 / COUNT(*), 1) AS churn_rate
    FROM customers
    GROUP BY city
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

model, feature_columns, impute_medians, scaler, model_name = load_model()
conn = get_db()

st.sidebar.markdown(f"**Active Model:** {model_name}")
st.sidebar.markdown(f"**Scaling:** {'Yes' if scaler is not None else 'No'}")

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
        row = profile.iloc[0]

        features = build_feature_vector(row, feature_columns)
        if scaler is not None:
            features = pd.DataFrame(
                scaler.transform(features),
                columns=features.columns,
                index=features.index,
            )
        risk_score = model.predict_proba(features)[0][1]

        st.subheader(row["name"])
        col1, col2, col3 = st.columns(3)
        col1.metric("City", row["city"])
        col2.metric("Member Since", row["signup_date"])
        col3.metric("Tenure", f"{row['tenure_months']} months")

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
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Satisfaction", f"{row['satisfaction_score']}/5")
        c2.metric("Days Since Order", int(row["days_since_last_order"]))
        c3.metric("Cashback", f"₱{row['cashback_amount']:,.2f}")
        c4.metric("Complaint", "Yes" if row["complain"] else "No")
        c5.metric("Devices", int(row["num_devices"]))

        c6, c7, c8, c9 = st.columns(4)
        c6.metric("Preferred Category", row["preferred_category"])
        c7.metric("Marital Status", row["marital_status"])
        c8.metric("Addresses", int(row["num_addresses"]))
        c9.metric("Warehouse Distance", f"{row['warehouse_to_home']} km")

        with st.expander("Order History"):
            orders = get_customer_orders(conn, selected_id)
            if orders.empty:
                st.info("No orders found.")
            else:
                st.dataframe(orders, use_container_width=True)

        st.divider()
        st.subheader("🤖 AI Recommendation")
        if st.button("Generate AI Recommendation", key="recommend"):
            with st.spinner("Generating recommendation..."):
                customer_data = {
                    "tenure_months": int(row["tenure_months"]),
                    "satisfaction_score": int(row["satisfaction_score"]),
                    "preferred_category": row["preferred_category"],
                    "marital_status": row["marital_status"],
                    "days_since_last_order": int(row["days_since_last_order"]),
                    "cashback_amount": float(row["cashback_amount"]),
                    "complain": int(row["complain"]),
                    "num_devices": int(row["num_devices"]),
                    "num_addresses": int(row["num_addresses"]),
                    "warehouse_to_home": int(row["warehouse_to_home"]),
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
        st.subheader("Avg Satisfaction by City")
        st.bar_chart(metrics_df.set_index("city")["avg_satisfaction"])

    st.divider()
    st.subheader("🤖 AI City Insights")
    if st.button("Generate City Insights"):
        with st.spinner("Analyzing city data..."):
            merged = churn_df.merge(metrics_df, on="city")
            summary = merged[["city", "churn_rate", "total_revenue", "avg_order_value", "avg_satisfaction"]].to_string(index=False)
            insights = explain_city_insights(summary)
        st.info(insights)
