# Code Review and Security Review Findings

## Finding 1: Vulnerable dependency set

**Location:** `requirements.txt:13-15`  
**Priority:** P1

`pip-audit -r requirements.txt` found known CVEs in the resolved stack: `streamlit==1.38.0` has CVE-2026-33682 fixed in 1.54.0, `python-dotenv==1.0.1` has CVE-2026-28684 fixed in 1.2.2, and transitive `pillow==10.4.0` has CVE-2026-25990/CVE-2026-40192 fixed by 12.2.0.

The current compatible-release pins prevent at least the Streamlit and dotenv fixes, so raise those lower bounds before publishing or demoing from an exposed machine.

## Finding 2: Customer details are sent to Anthropic on every rerun

**Location:** `app.py:233-249`  
**Priority:** P1

The customer recommendation call is executed whenever `selected_id` is truthy, which means the first page render and ordinary Streamlit reruns can transmit customer-level profile fields to Anthropic and spend API quota without a deliberate user action.

Gate this behind a button, show a clear opt-in/fallback when `ANTHROPIC_API_KEY` is missing, and consider caching by customer/risk score.

## Finding 3: Binary model load is an executable trust boundary

**Location:** `app.py:20`  
**Priority:** P2

`joblib.load()` can execute code embedded in a malicious pickle. In this repo the artifact is committed and not user-uploaded, but for a public workshop repo a binary PR or compromised artifact can hide executable behavior outside normal code review.

Prefer regenerating from the notebook in setup/CI, storing a checksum, or using a safer model serialization format such as `skops` with trusted types.

## Finding 4: Test data leaks into preprocessing

**Location:** `notebooks/01_churn_model.ipynb:589-626`  
**Priority:** P2

The notebook computes imputation medians and one-hot columns on the full dataframe before `train_test_split`, so test-set distribution information influences the trained artifact and reported metrics.

Split first, then fit imputation/encoding on `X_train` only, ideally through a scikit-learn `Pipeline`/`ColumnTransformer` saved with the model.

## Finding 5: App opens the SQLite DB read-write and shares one connection

**Location:** `app.py:30-32`  
**Priority:** P3

The app only reads from the DB, but it caches a process-wide connection opened in default read-write mode with `check_same_thread=False`.

For a Streamlit app, open the database as `file:...?...mode=ro` with `uri=True`, or create short-lived read-only connections, so runtime bugs cannot mutate the committed DB and concurrent sessions are less fragile.
