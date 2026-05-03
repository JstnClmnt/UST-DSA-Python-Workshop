# UST DSA Python Workshop - Live Demo Branch

> This is the **workshop branch**. It has the same dataset, notebook, model, scripts, and docs as `main`, but intentionally does **not** include `app.py`. Looking for the complete app? Switch to [`main`](../../tree/main).

## What's Different Here

This branch is set up for the live coding segment of the workshop. Students can inspect the completed ML pipeline and supporting files, then build the Streamlit app live on stage using AI-assisted coding.

## Prerequisites

- [ ] Python 3.12+ installed
- [ ] [uv](https://docs.astral.sh/uv/) installed
- [ ] An [Anthropic API key](https://console.anthropic.com/) or Claude subscription
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

```text
UST-DSA-Python-Workshop/
|-- data/
|   `-- ecommerce_churn.db        # SQLite DB (3,941 customers + 13,998 orders)
|-- models/
|   `-- churn_model.pkl           # Pre-trained Random Forest model
|-- notebooks/
|   `-- 01_churn_model.ipynb      # Full ML pipeline (EDA -> 3 models -> SHAP)
|-- scripts/
|   |-- build_churn_db.py         # Builds ecommerce_churn.db from Kaggle CSV
|   `-- build_ph_db.py            # Legacy: builds Olist-based DB
|-- datasets/
|   |-- customer_profiles.csv     # Synthetic Filipino names and cities
|   `-- data_ecommerce_customer_churn.csv
|-- docs/
|   |-- app-architecture.md       # Mermaid app architecture diagram
|   |-- Customer Churn Prediction-2026-05-03-173204.png
|   `-- other design/review docs
|-- requirements.txt
|-- .env.example
|-- HANDOFF.md
`-- README.md
```

**Missing:** `app.py` - this is built during the live demo.

## Workshop Flow

| # | Segment | Duration | What Happens |
|---|---------|----------|-------------|
| 1 | Hook | 10 min | Switch to `main`, run the finished app: `streamlit run app.py` |
| 2 | Notebook | 10 min | Walk through the ML pipeline in Jupyter |
| 3 | Diagram | 5 min | Show `docs/app-architecture.md` and the PNG architecture diagram |
| 4 | Live Coding | 20 min | On this branch, generate `app.py` using AI-assisted coding |
| 5 | Live Run | 8 min | Run the generated app: `streamlit run app.py` |
| 6 | Wrap-up | 10 min | Q&A, GitHub link, takeaways |

## The Live Coding Demo

With no `app.py` on this branch, use the following prompt to generate it:

```text
I have a Jupyter notebook that:
1. Loads a Kaggle CSV (datasets/data_ecommerce_customer_churn.csv) with 3,941
   customers and 10 features like tenure, satisfaction, complaints, cashback,
   preferred category, marital status, etc.
2. Trains 3 models (Random Forest, MLP, SVM) with GridSearchCV and F2 scoring
   to predict customer churn.
3. Selects the best model (Random Forest) and saves it to models/churn_model.pkl
   as a dict with model, feature_columns, impute_medians, scaler, and model_name.

I also have a SQLite database (data/ecommerce_churn.db) with 2 tables - customers
and orders - built by scripts/build_churn_db.py from the same CSV plus synthetic
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

The AI-assisted coding tool can read the notebook, database schema, `docs/app-architecture.md`, and `HANDOFF.md` to understand the full context, then generate a working `app.py`.

## After the Workshop

To see the complete reference implementation with the finished 2-tab Streamlit app:

```bash
git checkout main
streamlit run app.py
```

## Acknowledgments

- **Dataset:** [E-Commerce Customer Churn](https://www.kaggle.com/datasets/samuelsemaya/e-commerce-customer-churn) by Samuel Semaya (Kaggle)
- **Workshop:** Python Workshop for BS Data Science & Analytics, University of Santo Tomas, Manila
- **Built with:** [Streamlit](https://streamlit.io/) and [Anthropic Claude API](https://docs.anthropic.com/)
