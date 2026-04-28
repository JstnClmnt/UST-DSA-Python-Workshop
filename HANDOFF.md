# Workshop Handoff — AI-Powered Data Apps: From Jupyter to Production
> This file is the single source of truth for Claude Code (or any LLM assistant) 
> to understand the full context of this workshop project before generating any code, 
> diagrams, scripts, or assets.

---

## 1. Context

**Event:** Python Workshop for BS Data Science and Analytics students  
**University:** University of Santo Tomas (UST), Manila, Philippines  
**Audience:** Mixed year levels — 1st to 4th year  
**Duration:** 90 minutes  
**Assumption:** Students already have foundational ML/data processing knowledge. Do not over-explain basics.

---

## 2. Workshop Title & Narrative

**Title:** AI-Powered Data Apps: From Jupyter to Production

**Core Narrative:**
> "You already know how to build models. Today we'll show you how professionals 
> ship them — and how AI-assisted coding tools are changing the way that happens."

The workshop tells a single end-to-end story:
```
Real Data → Notebook (prototype) → Diagram (mental model) → Claude Code (generate script) → Running App
```

---

## 3. Workshop Timeline

| # | Segment | Duration | Format | Description |
|---|---|---|---|---|
| 1 | Hook | 10 mins | Live demo | Show the finished Streamlit app running. Let audience interact. "Here's what we're building." |
| 2 | Notebook | 10 mins | Jupyter | Scroll through notebook, run it once. Not a deep dive — "this is our foundation." |
| 3 | Diagram | 5 mins | Visual/Slide | Show how notebook maps to a production script. Bridge before Claude Code. |
| 4 | Claude Code (live) | 20 mins | Terminal | Switch to clean git branch (no script). Prompt Claude Code to generate Streamlit script. Review output together. |
| 5 | Live Run | 8 mins | Terminal | `streamlit run app.py` — run the generated script live |
| 6 | Wrap-up | 10 mins | Talk | GitHub repo link, year-level takeaways, Q&A |

**Total: ~63 mins** (leaves ~27 min buffer for pacing/questions)

---

## 4. Dataset

**Source:** Brazilian E-Commerce Public Dataset by Olist  
**Kaggle link:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce  
**SQLite version:** https://www.kaggle.com/datasets/terencicp/e-commerce-dataset-by-olist-as-an-sqlite-database  
**Size:** ~100k real orders, 2016–2018  
**License:** Public / CC BY-NC-SA 4.0

**Why Olist:** Real commercial data with authentic noise, outliers, and churn signals. 
Not synthetic — students are told this explicitly during the workshop.

### Localization (PH Adaptation)
The dataset is Brazilian. Before the workshop, run a preprocessing/remapping script to adapt it:

- **Cities:** Remap Brazilian cities → Philippine cities (Manila, Quezon City, Makati, Cebu, Davao, Pasig, Taguig, etc.)
- **Currency:** Convert BRL → PHP (multiply by ~9.5)
- **Categories:** Use the included `product_category_name_translation.csv` for English names
- **Output:** Save the adapted data back into a SQLite `.db` file

### Final Database Schema (after adaptation)
Use a simplified 3-table schema for the workshop:

```sql
customers (
    customer_id     TEXT PRIMARY KEY,
    name            TEXT,
    city            TEXT,       -- PH cities
    signup_date     DATE
)

orders (
    order_id                TEXT PRIMARY KEY,
    customer_id             TEXT,
    order_date              TIMESTAMP,
    total_amount            DECIMAL,    -- in PHP
    status                  TEXT,       -- Delivered, Cancelled, Shipped, etc.
    days_since_last_order   INT         -- pre-computed
)

products (
    product_id      TEXT PRIMARY KEY,
    name            TEXT,
    category        TEXT,
    price           DECIMAL     -- in PHP
)
```

> Note: `days_since_last_order` is pre-computed during preprocessing so no 
> feature engineering is needed live during the workshop.

---

## 5. ML Use Case

**Task:** Customer Churn Prediction (binary classification)  
**Definition of churn:** Customer has not placed an order in the last 90 days  
**Target column:** `is_churned` (1 = churned, 0 = active)

### Features
```python
features = [
    'days_since_last_order',
    'total_orders',
    'avg_order_value',      # in PHP
    'cancellation_rate',    # proportion of cancelled orders
]
target = 'is_churned'
```

### Model
- Use `RandomForestClassifier` or `LogisticRegression` (fast to train, easy to explain)
- Train/test split: 80/20
- Save with `joblib.dump(model, 'churn_model.pkl')`
- **Model is pre-trained before the workshop.** Training code is shown in the notebook but the `.pkl` file is already saved. The script loads the pre-trained model — it does not retrain.

---

## 6. Application Stack

| Layer | Tool | Notes |
|---|---|---|
| Data | SQLite | Single `.db` file, no server needed |
| Data manipulation | pandas | `pd.read_sql()` — not `read_csv()` |
| ML | scikit-learn | Pre-trained, loaded via joblib |
| Frontend | Streamlit | Main UI |
| AI layer | Anthropic Python SDK | Direct API call — NOT LangChain |
| AI model | `claude-sonnet-4-20250514` | Sonnet 4.6 |
| Dev assistant | Claude Code | Used live on stage to generate the script |

### Anthropic API Usage
The app calls Claude to generate a **natural language explanation** of each churn prediction.

```python
import anthropic

client = anthropic.Anthropic()

def explain_churn_risk(customer_data: dict, risk_score: float) -> str:
    prompt = f"""
    A customer has a {risk_score:.0%} churn risk score.
    Their profile:
    - Days since last order: {customer_data['days_since_last_order']}
    - Total orders: {customer_data['total_orders']}
    - Avg order value: ₱{customer_data['avg_order_value']:,.2f}
    - Cancellation rate: {customer_data['cancellation_rate']:.0%}

    Write a brief, actionable recommendation for the business owner.
    Be concise and specific. 2-3 sentences max.
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

---

## 7. Git Branch Strategy

| Branch | Contains | Purpose |
|---|---|---|
| `main` | Everything (notebook + script + model) | Full working reference, shown in Hook |
| `workshop` | Notebook + model only, NO script | Used during Claude Code live demo |

During the workshop:
1. Open `main` → run the finished app for the Hook
2. Switch to `workshop` branch → no `app.py` exists here
3. Prompt Claude Code to generate `app.py` live

---

## 8. Diagram to Generate

Claude Code should generate a **visual diagram** (can be a Mermaid diagram, ASCII, or SVG) 
showing how the notebook maps to the production script:

```
NOTEBOOK                              SCRIPT (app.py)
──────────────────────                ─────────────────────────
① Load SQLite DB       ──────────►   utils.py / inline db load
② Feature Engineering  ──────────►   precomputed in .db file
③ Train & Save Model   ──────────►   churn_model.pkl (pre-loaded)
④ Evaluate Model       ──────────►   (offline, not in app)
                                      app.py
                                        ├── load model (joblib)
                                        ├── connect to SQLite
                                        ├── customer lookup UI
                                        ├── run prediction
                                        └── Claude API → explanation
```

This diagram is shown at Segment 3 (the 5-min bridge before Claude Code).

---

## 9. Repo Structure (target)

```
workshop-repo/
│
├── data/
│   └── ecommerce_ph.db          # Adapted Olist SQLite DB
│
├── notebooks/
│   └── 01_churn_model.ipynb     # Full notebook: load → features → train → save
│
├── models/
│   └── churn_model.pkl          # Pre-trained model
│
├── app.py                       # Streamlit app (generated by Claude Code live)
├── requirements.txt
└── README.md
```

---

## 10. Claude Code Prompt (to use on stage)

When on the `workshop` branch (no `app.py`), use this prompt for Claude Code:

```
I have a Jupyter notebook that:
1. Loads a SQLite database (ecommerce_ph.db) with 3 tables: customers, orders, products
2. Computes churn features: days_since_last_order, total_orders, avg_order_value, cancellation_rate
3. Trains a RandomForestClassifier to predict customer churn (is_churned)
4. Saves the model to models/churn_model.pkl

Convert this into a Streamlit app (app.py) that:
- Lets the user look up a customer by ID
- Loads the pre-trained model from models/churn_model.pkl
- Shows the customer's churn risk score as a percentage
- Calls the Anthropic API (claude-sonnet-4-20250514) to generate a plain-English 
  explanation and recommendation based on the prediction
- Displays everything cleanly in the Streamlit UI
- Uses pd.read_sql() to query the SQLite DB (not read_csv)
- Uses the anthropic Python SDK directly (not LangChain)

Keep the code clean, well-commented, and beginner-readable.
```

---

## 11. Key Decisions & Rationale (for context)

| Decision | Choice | Why |
|---|---|---|
| LangChain vs Anthropic SDK | Direct SDK | Fewer abstractions to explain, more transferable skill |
| Synthetic vs real data | Real (Olist) | Authentic ML patterns, "100k real orders" lands better |
| CSV vs SQLite | SQLite | Teaches `pd.read_sql()`, mirrors production patterns |
| Train live vs pre-trained | Pre-trained | Saves 10-15 mins, training code still shown in notebook |
| Dedicated AI section vs woven in | Claude Code IS the AI section | More authentic than a separate slide-based demo |

---

## 12. What Claude Code Should NOT Do

- Do not use LangChain or any other LLM orchestration framework
- Do not use `pd.read_csv()` — always use `pd.read_sql()` with SQLite connection
- Do not retrain the model inside the Streamlit app
- Do not over-engineer — this is a workshop demo, keep it readable
- Do not add authentication, multi-user support, or caching complexity
- Do not add `order_items` table — schema is intentionally simplified to 3 tables
