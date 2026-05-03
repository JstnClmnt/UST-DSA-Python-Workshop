# Streamlit App Architecture

Paste the Mermaid diagram below into Claude web and ask it to visualize it.

```mermaid
flowchart TB
    subgraph sources["Data Sources"]
        DB[("SQLite DB<br/>ecommerce_churn.db<br/><i>3,941 customers · 13,998 orders</i>")]
        PKL["Model Artifact<br/>churn_model.pkl"]
    end

    subgraph loading["Cached Loading (runs once)"]
        direction LR
        CONN["get_db()<br/><i>Read-only connection</i>"]
        MODEL["load_model()<br/><i>model, feature_columns,<br/>impute_medians, scaler</i>"]
    end

    DB --> CONN
    PKL --> MODEL

    subgraph tab1["Tab 1 — Customer Lookup"]
        direction TB
        Q1["Query customer list<br/><i>SELECT customer_id, name<br/>FROM customers</i>"]
        SELECT["Dropdown selector<br/><i>st.selectbox</i>"]
        Q2["Query customer profile<br/><i>SELECT * FROM customers<br/>WHERE customer_id = ?</i>"]
        FEAT["Build feature vector<br/><i>Map DB columns → model features<br/>One-hot encode categories<br/>Reindex to training columns</i>"]
        SCALE{"Scaler<br/>exists?"}
        TRANSFORM["StandardScaler<br/>transform"]
        PREDICT["model.predict_proba()<br/><i>→ churn probability</i>"]
        RISK["Display risk score<br/><i>🟢 Low · 🟡 Medium · 🔴 High</i>"]
        METRICS["Display customer metrics<br/><i>Tenure, satisfaction, cashback,<br/>complaints, devices, etc.</i>"]
        Q3["Query order history<br/><i>SELECT order_date, total_amount, ...<br/>FROM orders WHERE customer_id = ?</i>"]
        ORDERS["Expandable order table"]
        BTN1{{"Button: Generate AI Recommendation"}}
        CLAUDE1["Claude API<br/><i>claude-sonnet-4-6</i><br/>→ Retention recommendation"]

        Q1 --> SELECT --> Q2
        Q2 --> FEAT --> SCALE
        SCALE -- Yes --> TRANSFORM --> PREDICT
        SCALE -- No --> PREDICT
        PREDICT --> RISK
        Q2 --> METRICS
        Q2 --> Q3 --> ORDERS
        BTN1 --> CLAUDE1
    end

    subgraph tab2["Tab 2 — City Analytics"]
        direction TB
        Q4["Query city metrics<br/><i>Revenue, orders, avg order value,<br/>satisfaction by city</i>"]
        Q5["Query city churn rates<br/><i>Churn rate by city</i>"]
        KPI["Summary KPIs<br/><i>Total cities · Avg churn rate<br/>Total revenue · Total orders</i>"]
        CHARTS["Bar charts<br/><i>Churn rate · Revenue<br/>Avg order value · Satisfaction</i>"]
        BTN2{{"Button: Generate City Insights"}}
        CLAUDE2["Claude API<br/><i>claude-sonnet-4-6</i><br/>→ City-level analysis"]

        Q4 --> KPI
        Q5 --> KPI
        Q4 --> CHARTS
        Q5 --> CHARTS
        BTN2 --> CLAUDE2
    end

    CONN --> Q1
    CONN --> Q2
    CONN --> Q3
    CONN --> Q4
    CONN --> Q5
    MODEL --> FEAT
    MODEL --> SCALE
```

## Text Description

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│  SQLite DB (ecommerce_churn.db)    Model Artifact (.pkl)        │
│  • customers table (3,941 rows)    • Random Forest model        │
│  • orders table (13,998 rows)      • feature_columns            │
│                                    • impute_medians             │
│                                    • StandardScaler             │
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────┐       ┌──────────────────────────┐
│  get_db()            │       │  load_model()            │
│  Cached DB connection│       │  Cached model + scaler   │
│  (read-only mode)    │       │  + feature columns       │
└──────────┬───────────┘       └────────────┬─────────────┘
           │                                │
     ┌─────┴──────────────────┬─────────────┘
     ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  TAB 1: CUSTOMER LOOKUP                                         │
│                                                                 │
│  1. Query customer list → Dropdown selector                     │
│  2. Query selected customer profile                             │
│  3. Build feature vector:                                       │
│     • Map DB columns → model feature names                      │
│     • One-hot encode (PreferedOrderCat, MaritalStatus)          │
│     • Reindex to match training columns (fill_value=0)          │
│  4. Scale features (if scaler exists)                           │
│  5. model.predict_proba() → churn probability                   │
│  6. Display: risk score (color-coded), customer metrics          │
│  7. Query + display order history (expandable)                  │
│  8. [Button] → Claude API → AI retention recommendation         │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  TAB 2: CITY ANALYTICS                                          │
│                                                                 │
│  1. Query city-level metrics (revenue, orders, satisfaction)    │
│  2. Query city churn rates                                      │
│  3. Display: summary KPIs + bar charts (4 charts)               │
│  4. [Button] → Claude API → AI city insights                    │
└─────────────────────────────────────────────────────────────────┘
```
