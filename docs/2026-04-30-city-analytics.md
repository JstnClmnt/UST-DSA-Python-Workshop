# City-Level Churn Analytics Dashboard

## Context

The current Streamlit app (`app.py`) only supports **individual customer-level** churn prediction — user picks a customer, sees their risk score, gets an AI recommendation. We want to add a **city-level analytics view** so users can compare churn and revenue metrics across the 24 Philippine cities in the database. This makes the workshop demo more insightful ("which markets are at risk?") without adding complexity.

**Scope:** Modify `app.py` only. No new files, no new dependencies, no database changes.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Navigation | `st.tabs` (2 tabs) | Simplest approach, keeps everything in one page |
| Layout | Change to `layout="wide"` | City charts need horizontal space |
| 4th metric | Cancellation rate by city (not top categories) | No `order_items` table exists to join orders to products |
| Charts | `st.bar_chart` | Built-in, zero extra imports, good for workshop |
| AI integration | "Generate City Insights" button | Prevents auto API calls on every rerender |
| Caching | No `@st.cache_data` on query functions | Follows existing pattern in the app |

## 4 City Metrics

1. **Churn rate** — % of customers with `churn = 1` (from Kaggle ground-truth label)
2. **Total revenue** — `SUM(total_amount)` grouped by city
3. **Average order value** — `AVG(total_amount)` grouped by city
4. **Cancellation rate** — % of orders with `status = 'Cancelled'`

## Implementation Steps

### Step 1: Update page config

**File:** `app.py:11-15`

Change `layout="centered"` to `layout="wide"` and update the subtitle markdown (line 77) to reflect both views.

### Step 2: Add `get_city_metrics(conn)` function

**File:** `app.py` — insert after `explain_churn_risk` (after ~line 70), before `# --- UI ---`

SQL (single query for revenue, AOV, cancellation rate):
```sql
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
```

### Step 3: Add `get_city_churn_rates(conn)` function

**File:** `app.py` — insert after `get_city_metrics`

SQL (uses Kaggle ground-truth `churn` column directly):
```sql
SELECT
    city,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(SUM(churn) * 100.0 / COUNT(*), 1) AS churn_rate
FROM customers
GROUP BY city
ORDER BY churn_rate DESC
```

### Step 4: Add `explain_city_insights(city_data: str)` function

**File:** `app.py` — insert after `get_city_churn_rates`

Mirrors existing `explain_churn_risk` pattern. Calls `claude-sonnet-4-6` with `max_tokens=400`. Prompt asks for which cities need attention and 2-3 actionable recommendations.

### Step 5: Restructure UI with tabs

**File:** `app.py` — UI section (~line 73 onward)

- Add `tab1, tab2 = st.tabs(["Customer Lookup", "City Analytics"])` after loading model/conn
- Wrap all existing customer lookup UI code inside `with tab1:`
- No logic changes to existing code — just indentation

### Step 6: Build City Analytics tab

**File:** `app.py` — inside `with tab2:`

Layout:
```
Row 1: 4 summary st.metric cards (total cities, avg churn rate, total revenue, total orders)
---
Row 2: Churn Rate by City bar chart (full width)
Row 3: Total Revenue by City bar chart (full width)
---
Row 4: Two columns
  Left:  Avg Order Value by City bar chart
  Right: Cancellation Rate by City bar chart
---
Row 5: "Generate City Insights" button -> Claude API call -> st.info()
```

## Claude Code Live Demo Prompt (Segment 4)

On the `workshop` branch (no `app.py`), use this prompt to kick off Claude Code in plan mode:

```
I have a Jupyter notebook that trains three ML models (Random Forest, MLP, SVM) for
churn prediction and saves the best one. Convert it into a Streamlit app that lets me
look up customers, see their churn risk score, and get AI recommendations using Claude.
The app should handle the model artifact format (which may include a scaler) and show
which model won in the sidebar.
```

**Expected flow:**
1. Claude Code reads the notebook and database to understand the schema, model artifact format, and scaler
2. It enters plan mode and proposes an implementation plan
3. Presenter reviews and approves the plan
4. Claude Code generates `app.py`
5. Run `streamlit run app.py` live

## Verification

1. Run `streamlit run app.py`
2. Confirm the "Customer Lookup" tab works exactly as before
3. Switch to "City Analytics" tab
4. Verify 4 summary metrics display at the top
5. Verify all 4 bar charts render with 24 cities each
6. Click "Generate City Insights" and confirm Claude returns analysis
7. Switch back to "Customer Lookup" — confirm no regressions
