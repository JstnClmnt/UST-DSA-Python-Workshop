# UST DSA Python Workshop — Live Demo Branch

> This is the **workshop branch** — a stripped-down version of the repo for the live Claude Code demo. Looking for the complete app? Switch to [`main`](../../tree/main).

## What's Different Here

This branch has the **notebook and pre-trained model** but **no `app.py`**. During the workshop, [Claude Code](https://claude.com/claude-code) generates the Streamlit app live on stage.

| | `main` branch | `workshop` branch |
|---|---|---|
| Dataset | Kaggle churn (3,941 customers) | Olist PH-localized (~96K customers) |
| Database | `ecommerce_churn.db` (3 MB) | `ecommerce_ph.db` (31 MB) |
| Schema | 2 tables (customers, orders) | 3 tables (customers, orders, products) |
| ML features | 10 features + one-hot encoding | 4 derived features |
| Models | RF + MLP + SVM with GridSearchCV | Random Forest only |
| `app.py` | Included (285 lines, 2 tabs) | **Not included** — generated live |
| SHAP | Yes | No |

## Prerequisites

- [ ] Python 3.12+ installed
- [ ] [uv](https://docs.astral.sh/uv/) installed
- [ ] [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) installed (`npm install -g @anthropic-ai/claude-code`)
- [ ] An [Anthropic API key](https://console.anthropic.com/) or Claude Subscription
- [ ] Git installed

## Setup

```bash
git clone https://github.com/JstnClmnt/UST-DSA-Python-Workshop.git
cd UST-DSA-Python-Workshop
git checkout workshop

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## What's in This Branch

```
UST-DSA-Python-Workshop/
├── data/
│   └── ecommerce_ph.db          # PH-localized Olist SQLite DB
│       ├── customers (96,096)    #   customer_id, name, city, signup_date
│       ├── orders (99,441)       #   order_id, customer_id, order_date, total_amount, status, days_since_last_order
│       └── products (32,951)     #   product_id, name, category, price
├── models/
│   └── churn_model.pkl           # Pre-trained Random Forest model
├── notebooks/
│   └── 01_churn_model.ipynb      # ML pipeline: load → EDA → features → train → save
├── scripts/
│   └── build_ph_db.py            # How the database was built (reference only)
├── requirements.txt
├── .env.example
├── HANDOFF.md                    # Full context doc — Claude Code reads this
└── README.md
```

## Workshop Flow (90 minutes)

| # | Segment | Duration | What Happens |
|---|---------|----------|-------------|
| 1 | Hook | 10 min | Switch to `main`, run the finished app: `streamlit run app.py` |
| 2 | Notebook | 10 min | Walk through the ML pipeline in Jupyter |
| 3 | Diagram | 5 min | How notebook concepts map to a production app |
| 4 | Claude Code | 20 min | On this branch, prompt Claude Code to generate `app.py` |
| 5 | Live Run | 8 min | Run the generated app: `streamlit run app.py` |
| 6 | Wrap-up | 10 min | Q&A, GitHub link, takeaways |

## The Claude Code Demo

With no `app.py` on this branch, we prompt Claude Code to generate it:

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
- Calls the Anthropic API (claude-sonnet-4-6) to generate a plain-English
  explanation and recommendation based on the prediction
- Displays everything cleanly in the Streamlit UI
- Uses pd.read_sql() to query the SQLite DB (not read_csv)
- Uses the anthropic Python SDK directly (not LangChain)

Keep the code clean, well-commented, and beginner-readable.
```

Claude Code reads the notebook, database schema, and `HANDOFF.md` to understand the full context, then generates a working `app.py`.

## After the Workshop

To see the complete reference implementation with multi-model comparison, SHAP explanations, and the finished 2-tab Streamlit app:

```bash
git checkout main
streamlit run app.py
```

## Acknowledgments

- **Dataset:** [Brazilian E-Commerce by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle, CC BY-NC-SA 4.0)
- **Workshop:** Python Workshop for BS Data Science & Analytics, University of Santo Tomas, Manila
- **Built with:** [Claude Code](https://claude.com/claude-code) and [Streamlit](https://streamlit.io/)
