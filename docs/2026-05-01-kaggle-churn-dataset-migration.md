# Kaggle Churn Dataset Migration — Design Spec

**Date:** 2026-05-01
**Status:** Draft

## Context

The current pipeline has data leakage: churn is defined as `days_since_last_order > 90`, and the RandomForest model places 99.92% feature importance on that same column, yielding a fake 100% accuracy. This is pedagogically harmful for a workshop teaching real ML.

We're replacing the data layer with a Kaggle dataset ([E-Commerce Customer Churn](https://www.kaggle.com/datasets/samuelsemaya/e-commerce-customer-churn)) that has a proper, independently-labeled `Churn` column (3,941 rows, ~17% churn rate). We synthesize PH-localized identity and order history around it, sampling distributions from the Olist dataset for realism.

## Pipeline

```
datasets/data_ecommerce_customer_churn.csv (3,941 rows, proper churn label)
    ↓
scripts/build_churn_db.py (NEW)
    ↓
data/ecommerce_churn.db (NEW — 2 tables: customers + orders)
    ↓
notebooks/01_churn_model.ipynb (REWRITTEN — Kaggle features + GridSearchCV)
    ↓
models/churn_model.pkl (REPLACED — new model artifact with feature config)
    ↓
app.py (REWRITTEN — new DB, new features, new queries)
```

Old files (`build_ph_db.py`, `ecommerce_ph.db`) are kept for reference, not deleted.

---

## 1. Build Script — `scripts/build_churn_db.py`

### Input
- `datasets/data_ecommerce_customer_churn.csv` (3,941 rows — ML features + churn labels)
- `datasets/customer_profiles.csv` (3,941 rows — synthesized Filipino names & PH cities)

### Output
- `data/ecommerce_churn.db` (SQLite, 2 tables)

### Schema

**`customers`** (3,941 rows)
```sql
CREATE TABLE customers (
    customer_id         TEXT PRIMARY KEY,
    name                TEXT,
    city                TEXT,
    signup_date         DATE,
    tenure_months       INTEGER,
    warehouse_to_home   INTEGER,
    num_devices         INTEGER,
    preferred_category  TEXT,
    satisfaction_score  INTEGER,
    marital_status      TEXT,
    num_addresses       INTEGER,
    complain            INTEGER,
    days_since_last_order INTEGER,
    cashback_amount     REAL,
    churn               INTEGER
);
```

**`orders`** (synthesized, ~10K-20K rows)
```sql
CREATE TABLE orders (
    order_id      TEXT PRIMARY KEY,
    customer_id   TEXT,
    order_date    TIMESTAMP,
    total_amount  REAL,
    status        TEXT,
    payment_type  TEXT,
    review_score  INTEGER
);
CREATE INDEX idx_orders_customer ON orders(customer_id);
```

### Null Imputation (before synthesis)

| Column | Nulls | Strategy |
|--------|-------|----------|
| Tenure | 194 | Median (9.0) |
| WarehouseToHome | 169 | Median (14.0) |
| DaySinceLastOrder | 213 | Median (3.0) |

### Customer Identity

Name and city come from `datasets/customer_profiles.csv` (pre-generated, checked into git).

| Field | Derivation |
|-------|------------|
| `customer_id` | `sha256(f"kaggle-churn-{row_index}")[:32]` |
| `name` | Read from `customer_profiles.csv` |
| `city` | Read from `customer_profiles.csv` |
| `signup_date` | `REFERENCE_DATE - timedelta(days=Tenure * 30)` where `REFERENCE_DATE = 2025-03-01` |

### Order History Synthesis

For each customer, generate orders consistent with their Kaggle features:

**Number of orders:** `max(1, round(Tenure / 3 * jitter))` where jitter ~ `N(1.0, 0.3)`, clamped to `[1, 20]`.

**Order dates:**
- Last order = `REFERENCE_DATE - DaySinceLastOrder days`
- First order = `signup_date`
- Intermediate orders distributed uniformly with random perturbation

**Order amounts (PHP):**
- Base amount = `max(CashbackAmount * 6, 200)`
- Per-order: `lognormal(ln(base), 0.4)`, clamped to `[91, 15000]` PHP

**Product categories → preferred category mapping:**
- `"Laptop & Accessory"` → computers, electronics, tablets
- `"Mobile Phone"` / `"Mobile"` → telephony, electronics
- `"Fashion"` → fashion_bags, fashion_shoes, fashion_female/male_clothing
- `"Grocery"` → food, food_drink, drinks
- `"Others"` → housewares, health_beauty, sports_leisure, etc.
- 70% from preferred pool, 30% random from all categories

**Order status** (from Olist distribution):
- Delivered: 97.0%, Shipped: 1.1%, Cancelled: 0.6%, other: 1.3%

**Review scores:**
- Only for Delivered orders (~90% get a review)
- Gaussian weights centered on customer's `SatisfactionScore`: `w[i] = exp(-|i+1 - score|)`, normalized

**Payment type** (from Olist distribution):
- Credit Card: 74%, Boleto: 19%, Voucher: 5.6%, Debit Card: 1.5%

### Reuse from `build_ph_db.py`
- `_seed_from_id()` function
- Overall script structure (constants → build functions → main → verify)
- Name/city lists moved to `customer_profiles.csv` (no longer inline in the build script)

---

## 2. Notebook — `notebooks/01_churn_model.ipynb`

### Data Source
Reads `datasets/data_ecommerce_customer_churn.csv` directly (NOT from the SQLite DB — keeps training independent of synthesized data).

### Features (10 → ~15 after encoding)

**Numeric (8):**
- Tenure, WarehouseToHome, NumberOfDeviceRegistered, SatisfactionScore
- NumberOfAddress, Complain, DaySinceLastOrder, CashbackAmount

**Categorical (2, one-hot encoded with `drop_first=True`):**
- PreferedOrderCat (6 values → 5 dummies)
- MaritalStatus (3 values → 2 dummies)

**Target:** Churn (0/1)

### Preprocessing
1. Impute nulls with median (Tenure, WarehouseToHome, DaySinceLastOrder)
2. One-hot encode categoricals with `pd.get_dummies(drop_first=True)`
3. Train/test split: 80/20, stratified, `random_state=42`
4. `StandardScaler` fitted on training set only (for MLP and SVM; RF uses unscaled data)

### Three Models — GridSearchCV with F2 Scoring

All models use `scoring=make_scorer(fbeta_score, beta=2)` — F2 weighs recall 4× more than precision, appropriate for churn where missing a churner is costlier than a false alarm.

**Random Forest** (108 candidates × 5 folds = 540 fits, unscaled data):
- `n_estimators`: [100, 200, 300], `max_depth`: [None, 10, 20, 30], `min_samples_split`: [2, 5, 10], `min_samples_leaf`: [1, 2, 4]

**MLP Neural Network** (36 candidates × 5 folds = 180 fits, scaled data):
- `hidden_layer_sizes`: [(64,32), (128,64), (100,)], `activation`: [relu, tanh], `alpha`: [0.0001, 0.001, 0.01], `learning_rate`: [constant, adaptive]
- `max_iter=500`, `early_stopping=True`

**SVM** (12 candidates × 5 folds = 60 fits, scaled data):
- `C`: [0.1, 1, 10], `kernel`: [rbf, linear], `gamma`: [scale, auto]
- `probability=True` for `predict_proba`

### Evaluation & Comparison
- Classification report per model
- ROC curves overlay (all 3 models + AUC in legend)
- Bar chart comparing Accuracy, F1, F2, Precision, Recall
- Side-by-side confusion matrices (1×3 subplot)
- SHAP feature importance bar chart (all 3 models) + beeswarm plot for winner

### Model Artifact

Best model selected by highest F2 on test set. Saved as a dict:

```python
artifact = {
    'model': best_model,
    'feature_columns': list(X.columns),
    'impute_medians': {
        'Tenure': df['Tenure'].median(),
        'WarehouseToHome': df['WarehouseToHome'].median(),
        'DaySinceLastOrder': df['DaySinceLastOrder'].median(),
    },
    'scaler': fitted_scaler_or_None,  # None if RF wins
    'model_name': 'Random Forest' | 'MLP' | 'SVM',
}
joblib.dump(artifact, '../models/churn_model.pkl')
```

---

## 3. App — `app.py`

### Changes

**DB connection:** `data/ecommerce_churn.db`

**Model loading:** Unpack artifact dict → `model`, `feature_columns`, `impute_medians`, `scaler`, `model_name`

**Customer features:** Read directly from `customers` table (no SQL aggregation from orders):
```sql
SELECT tenure_months, warehouse_to_home, num_devices, preferred_category,
       satisfaction_score, marital_status, num_addresses, complain,
       days_since_last_order, cashback_amount, churn
FROM customers WHERE customer_id = ?
```

**Feature vector for prediction:** Apply same `get_dummies` encoding as training, reindex to match `feature_columns` with `fill_value=0`.

**Customer metrics display (Tab 1):** Show all 10 features:
- Tenure, Satisfaction Score, Preferred Category, Marital Status
- Days Since Last Order, Cashback Amount, Complaint (Yes/No)
- Devices Registered, Addresses, Warehouse Distance

**Order history drill-down:** Expander showing synthesized orders:
```sql
SELECT order_date, total_amount, status, payment_type, review_score
FROM orders WHERE customer_id = ? ORDER BY order_date DESC
```

**Claude API prompt:** Updated to reference all 10 features.

**City analytics (Tab 2):**
- Churn rate from `customers.churn` directly (no more `days_since > 90` derivation)
- Revenue/order metrics via `customers JOIN orders`
- Add avg satisfaction score per city

---

## 4. File Changes Summary

| File | Action |
|------|--------|
| `scripts/build_churn_db.py` | CREATE (~200 lines, reads profiles CSV) |
| `datasets/customer_profiles.csv` | CREATE (3,941 rows — name, city) |
| `data/ecommerce_churn.db` | CREATE (generated by script) |
| `notebooks/01_churn_model.ipynb` | REWRITE (3 models, SHAP, F2 scoring) |
| `models/churn_model.pkl` | REPLACE (new artifact with scaler + model_name) |
| `app.py` | REWRITE (scaler support, sidebar model info) |
| `requirements.txt` | UPDATE (added shap) |
| `scripts/build_ph_db.py` | Keep (reference) |
| `data/ecommerce_ph.db` | Keep (reference) |

---

## 5. Verification

### Build script
- `SELECT COUNT(*) FROM customers` = 3,941
- `SELECT SUM(churn) FROM customers` = 674 (matches Kaggle exactly)
- `SELECT COUNT(DISTINCT city) FROM customers` = 24
- No NULLs in imputed columns
- Spot check: customer with Tenure=15 has orders spanning ~15 months

### Notebook
- Model accuracy is NOT 100% (leakage is gone)
- No single feature has > 50% importance
- All 3 models (RF, MLP, SVM) train and evaluate
- F2 scoring used for GridSearchCV and best-model selection
- SHAP visualizations render for all 3 models
- ROC curves, metrics bar chart, confusion matrices all display
- `churn_model.pkl` saved with scaler + model_name in artifact

### App
- `streamlit run app.py` starts without errors
- Sidebar shows winning model name and scaling status
- Customer dropdown shows Filipino names
- 10 feature metrics displayed (not old 4)
- Scaler applied before prediction when model requires it
- Churn risk is a realistic probability
- Claude recommendation references new features
- City analytics uses ground truth churn column
- Order history expander shows synthesized orders
