# UST DSA Python Workshop — Live Demo Branch

> This is the **workshop branch** — it has everything from `main` except `app.py`. Looking for the complete app? Switch to [`main`](../../tree/main).

## What's Different Here

This branch has the **same dataset, notebook, and pre-trained model** as `main` but **no `app.py`**. During the workshop, [Claude Code](https://claude.com/claude-code) generates the Streamlit app live on stage.

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
│   └── ecommerce_churn.db        # SQLite DB (3,941 customers + 13,998 orders)
├── models/
│   └── churn_model.pkl           # Pre-trained Random Forest model
├── notebooks/
│   └── 01_churn_model.ipynb      # Full ML pipeline (EDA → 3 models → SHAP)
├── scripts/
│   ├── build_churn_db.py         # Builds ecommerce_churn.db from Kaggle CSV
│   └── build_ph_db.py           # Legacy: builds Olist-based DB
├── datasets/
│   ├── customer_profiles.csv     # Synthetic Filipino names and cities
│   └── data_ecommerce_customer_churn.csv  # Kaggle source CSV
├── docs/                         # Design specs and research
├── requirements.txt
├── .env.example
├── HANDOFF.md                    # Full context doc — Claude Code reads this
└── README.md
```

**Missing:** `app.py` — this is what Claude Code generates during the live demo.

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
1. Loads a SQLite database (ecommerce_churn.db) with 2 tables: customers and orders
2. Trains 3 models (Random Forest, MLP, SVM) with GridSearchCV and F2 scoring
   to predict customer churn using 10 features (tenure, satisfaction, complaints,
   cashback, preferred category, marital status, etc.)
3. Selects the best model (Random Forest) and saves it to models/churn_model.pkl
   as a dict with model, feature_columns, impute_medians, scaler, and model_name

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

Claude Code reads the notebook, database schema, and `HANDOFF.md` to understand the full context, then generates a working `app.py`.

## After the Workshop

To see the complete reference implementation with the finished 2-tab Streamlit app:

```bash
git checkout main
streamlit run app.py
```

## Acknowledgments

- **Dataset:** [E-Commerce Customer Churn](https://www.kaggle.com/datasets/samuelsemaya/e-commerce-customer-churn) by Samuel Semaya (Kaggle)
- **Workshop:** Python Workshop for BS Data Science & Analytics, University of Santo Tomas, Manila
- **Built with:** [Claude Code](https://claude.com/claude-code) and [Streamlit](https://streamlit.io/)
