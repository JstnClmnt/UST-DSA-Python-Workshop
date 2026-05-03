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

**Source:** E-Commerce Customer Churn Dataset  
**Kaggle link:** https://www.kaggle.com/datasets/samuelsemaya/e-commerce-customer-churn  
**Size:** 3,941 customers with 10 features and a binary churn label (~17% churn rate)  
**License:** Public

**Why this dataset:** Pre-labeled churn column independent of features — no data leakage. 
Small enough to train quickly, large enough to be realistic.

### Database Schema

The raw Kaggle CSV is transformed by `scripts/build_churn_db.py` into a SQLite database 
at `data/ecommerce_churn.db` with 2 tables:

```sql
customers (
    customer_id             TEXT PRIMARY KEY,
    name                    TEXT,           -- synthetic PH names
    city                    TEXT,           -- PH cities (Manila, Cebu, Davao, etc.)
    signup_date             DATE,
    tenure_months           INT,
    warehouse_to_home       INT,
    num_devices             INT,
    preferred_category      TEXT,           -- Laptop, Mobile, Fashion, Grocery, Others
    satisfaction_score      INT,            -- 1-5
    marital_status          TEXT,           -- Single, Married, Divorced
    num_addresses           INT,
    complain                INT,            -- 0 or 1
    days_since_last_order   INT,
    cashback_amount         REAL,
    churn                   INT             -- 0 = active, 1 = churned
)

orders (
    order_id        TEXT PRIMARY KEY,
    customer_id     TEXT,
    order_date      TIMESTAMP,
    total_amount    REAL,           -- in PHP
    status          TEXT,           -- Delivered, Shipped, Cancelled, etc.
    payment_type    TEXT,           -- Credit Card, Boleto, Voucher, Debit
    review_score    INT             -- 1-5, nullable
)
```

> Note: Customer names and cities are synthetic (generated from hashed IDs). 
> Orders are synthesized to match customer profiles.

---

## 5. ML Use Case

**Task:** Customer Churn Prediction (binary classification)  
**Target column:** `churn` (1 = churned, 0 = active)  
**Churn rate:** ~17%

### Features (10 raw → 15 after one-hot encoding)
```python
numeric_features = [
    'Tenure',                       # tenure_months
    'WarehouseToHome',              # warehouse_to_home
    'NumberOfDeviceRegistered',     # num_devices
    'SatisfactionScore',            # satisfaction_score
    'NumberOfAddress',              # num_addresses
    'Complain',                     # complain
    'DaySinceLastOrder',            # days_since_last_order
    'CashbackAmount',              # cashback_amount
]
categorical_features = [
    'PreferedOrderCat',            # preferred_category (one-hot, drop_first)
    'MaritalStatus',               # marital_status (one-hot, drop_first)
]
target = 'Churn'
```

### Null Imputation
Three columns have nulls, imputed with median before training:
- Tenure: median 9.0 months
- WarehouseToHome: median 14 km
- DaySinceLastOrder: median 3 days

### Models
The notebook trains and compares three models using `GridSearchCV` with 5-fold CV and F2 scoring:

| Model | CV F2 | Test F2 | Test Accuracy | Test Recall |
|-------|-------|---------|---------------|-------------|
| **Random Forest** (winner) | 0.72 | 0.74 | 93.5% | 70% |
| SVM | 0.65 | 0.69 | 91.4% | 64% |
| MLP | 0.59 | 0.63 | 87.7% | 46% |

**F2 scoring rationale:** Recall is weighted 4x more than precision — missing a churner is costlier than a false alarm.

### Model Artifact
Saved at `models/churn_model.pkl` as a joblib dict:
```python
{
    "model": best_model,               # RandomForestClassifier
    "feature_columns": [...],           # 15 encoded feature names
    "impute_medians": {...},            # null imputation values
    "scaler": None,                     # RF doesn't need scaling
    "model_name": "Random Forest",
}
```

**Model is pre-trained before the workshop.** Training code is shown in the notebook 
but the `.pkl` file is already saved. The app loads the pre-trained model — it does not retrain.

---

## 6. Application Stack

| Layer | Tool | Notes |
|---|---|---|
| Data | SQLite | Single `.db` file, no server needed |
| Data manipulation | pandas | `pd.read_sql()` — not `read_csv()` |
| ML | scikit-learn | Pre-trained, loaded via joblib |
| Frontend | Streamlit | Main UI |
| AI layer | Anthropic Python SDK | Direct API call — NOT LangChain |
| AI model | `claude-sonnet-4-6` | Sonnet 4.6 |
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
    - Tenure: {customer_data['tenure_months']} months
    - Satisfaction Score: {customer_data['satisfaction_score']}/5
    - Preferred Category: {customer_data['preferred_category']}
    - Days Since Last Order: {customer_data['days_since_last_order']}
    - Cashback Amount: ₱{customer_data['cashback_amount']:,.2f}
    - Filed Complaint: {'Yes' if customer_data['complain'] else 'No'}

    Write a brief, actionable recommendation for the business owner.
    Be concise and specific. 2-3 sentences max.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

---

## 7. Git Branch Strategy

| Branch | Contains | Purpose |
|---|---|---|
| `main` | Everything (notebook + app + model) | Full working reference, shown in Hook |
| `workshop` | Same as main, minus `app.py` | Used during Claude Code live demo |

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
① Load SQLite DB       ──────────►   inline db load (sqlite3)
② EDA & Preprocessing  ──────────►   (offline, not in app)
③ Train & Compare      ──────────►   churn_model.pkl (pre-loaded)
④ SHAP Explanations    ──────────►   (offline, not in app)
                                      app.py
                                        ├── load model (joblib)
                                        ├── connect to SQLite
                                        ├── customer lookup UI
                                        ├── build feature vector
                                        ├── run prediction
                                        ├── Claude API → explanation
                                        └── city analytics tab
```

This diagram is shown at Segment 3 (the 5-min bridge before Claude Code).

---

## 9. Repo Structure (target)

```
workshop-repo/
│
├── data/
│   └── ecommerce_churn.db      # Kaggle churn SQLite DB
│
├── notebooks/
│   └── 01_churn_model.ipynb    # Full notebook: load → EDA → 3 models → SHAP → save
│
├── models/
│   └── churn_model.pkl         # Pre-trained Random Forest model
│
├── app.py                      # Streamlit app (generated by Claude Code live)
├── requirements.txt
└── README.md
```

---

## 10. Claude Code Prompt (to use on stage)

When on the `workshop` branch (no `app.py`), use this prompt for Claude Code:

```
I have a Jupyter notebook that:
1. Loads a Kaggle CSV (datasets/data_ecommerce_customer_churn.csv) with 3,941
   customers and 10 features like tenure, satisfaction, complaints, cashback,
   preferred category, marital status, etc.
2. Trains 3 models (Random Forest, MLP, SVM) with GridSearchCV and F2 scoring
   to predict customer churn
3. Selects the best model (Random Forest) and saves it to models/churn_model.pkl
   as a dict with model, feature_columns, impute_medians, scaler, and model_name

I also have a SQLite database (data/ecommerce_churn.db) with 2 tables — customers
and orders — built by scripts/build_churn_db.py from the same CSV plus synthetic
Filipino names and cities.

Convert this into a Streamlit app (app.py) that:
- Lets the user look up a customer by name from a dropdown
- Loads the pre-trained model from models/churn_model.pkl
- Shows the customer's churn risk score as a percentage with a color-coded badge
- Displays customer metrics (tenure, satisfaction, cashback, complaint status, etc.)
- Shows the customer's order history in an expandable section
- Calls the Anthropic API (claude-sonnet-4-6) to generate a plain-English
  explanation and recommendation based on the prediction and customer profile
- Adds a second tab for city-level analytics (churn rates, revenue, satisfaction)
- Uses pd.read_sql() to query the SQLite DB (not read_csv)
- Uses the anthropic Python SDK directly (not LangChain)

Keep the code clean and beginner-readable.
```

---

## 11. Key Decisions & Rationale (for context)

| Decision | Choice | Why |
|---|---|---|
| LangChain vs Anthropic SDK | Direct SDK | Fewer abstractions to explain, more transferable skill |
| Data source | Kaggle customer churn (3,941 rows, pre-labeled) | Fixes leakage problem; proper churn label independent of features |
| CSV vs SQLite | SQLite | Teaches `pd.read_sql()`, mirrors production patterns |
| Train live vs pre-trained | Pre-trained | Saves 10-15 mins, training code still shown in notebook |
| Scoring metric | F2 (recall 4x precision) | Churn miss is costlier than false alarm |
| Model architecture | 3 models with GridSearchCV | Demonstrates model selection; SHAP interpretability |
| Dedicated AI section vs woven in | Claude Code IS the AI section | More authentic than a separate slide-based demo |

---

## 12. What Claude Code Should NOT Do

- Do not use LangChain or any other LLM orchestration framework
- Do not use `pd.read_csv()` — always use `pd.read_sql()` with SQLite connection
- Do not retrain the model inside the Streamlit app
- Do not over-engineer — this is a workshop demo, keep it readable
- Do not add authentication, multi-user support, or caching complexity
