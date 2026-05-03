# Code Review & Security Audit — Pre-Publication

**Date:** 2026-05-03  
**Branches reviewed:** `main` (commit 5726a76), `workshop` (commit e426c30)  
**Scope:** Security, binary hygiene, code quality, workshop readiness

---

## Summary

The repository is **safe to publish on GitHub**. All customer data is synthetic, no secrets exist in git history, `.env` is properly gitignored, and SQL queries are parameterized. The main items to address are tracked binary files adding ~46 MB to clone size, a legacy database on `main` that only the `workshop` branch uses, and an outdated `HANDOFF.md`.

---

## Findings

| # | Finding | Severity | Recommendation |
|---|---------|----------|----------------|
| 1 | `data/ecommerce_ph.db` (31 MB) tracked on `main` — legacy, unused by any code on `main` | MEDIUM | Remove from `main`; it only belongs on `workshop` |
| 2 | ~46 MB total tracked binaries (`ecommerce_ph.db` 31 MB + `churn_model.pkl` 11.7 MB + `ecommerce_churn.db` 3 MB + CSVs ~0.4 MB) | MEDIUM | Acceptable — under GitHub's 100 MB push limit. Consider Git LFS if the repo grows |
| 3 | `HANDOFF.md` on `main` is outdated — still describes Olist dataset, 3-table schema, 4 features, `claude-sonnet-4-20250514` | MEDIUM | Update to reflect Kaggle churn dataset, 2-table schema, 10+ features, `claude-sonnet-4-6` |
| 4 | No `LICENSE` file | LOW | Add MIT (or preferred license) before publishing |
| 5 | `.env` contains a real Anthropic API key locally | INFO | Properly gitignored — not in repo history. Rotate the key before any public demo |
| 6 | `datasets/data_ecommerce_customer_churn.csv` tracked despite `.gitignore` pattern (`datasets/*`) | LOW | Likely force-added. Accept — it's 207 KB of public Kaggle data |

---

## Security Assessment

### Secrets
- `.env` is in `.gitignore` and has **never been committed** to git history
- `.env.example` contains only the placeholder `your-api-key-here`
- Full history scanned (`git log -p --all -S`): no API keys, tokens, or credentials found
- **Action needed:** Rotate the local API key before any public demo where the `.env` file might be visible on screen

### Data Privacy / PII
- All customer names and cities are **synthetic** — generated deterministically by `scripts/build_churn_db.py` using SHA-256 hashing of customer IDs
- The Kaggle CSV (`data_ecommerce_customer_churn.csv`) contains only numeric features and a churn label — no PII
- Developer email (`justinegc28@yahoo.com`) is visible in git commit metadata — standard for public repositories

### SQL Injection
- `app.py` uses parameterized queries throughout: `params=[customer_id]`
- No string interpolation in any SQL query

### Dependencies
- `requirements.txt` uses compatible-release pinning (`~=`) — good practice
- All packages are mainstream (pandas, scikit-learn, streamlit, anthropic, shap)
- No known vulnerable package versions detected

---

## Code Quality

### `app.py` (285 lines)
- Clean separation: imports, cached loaders, query functions, Claude API functions, UI
- `@st.cache_resource` used correctly for model loading and DB connection
- Feature vector construction handles one-hot encoding via `pd.get_dummies` + `reindex(fill_value=0)` — correctly aligns with training columns
- Claude API uses direct Anthropic SDK (not LangChain) with correct model ID `claude-sonnet-4-6`

### `notebooks/01_churn_model.ipynb` (36 cells)
- Well-organized with markdown section headers
- Three models compared (Random Forest, MLP, SVM) with `GridSearchCV` and F2 scoring
- SHAP explanations generated for all three models
- Model artifact saved as a dict containing model, scaler, feature columns, impute medians, and model name
- **Known issue:** Imputation medians and one-hot encoding are computed on the full dataset before `train_test_split`, which leaks test-set statistics into training. The correct approach is to fit on `X_train` only (e.g., via a scikit-learn `Pipeline`). Impact is small given the dataset size, but reported metrics are slightly optimistic. Deferred for pedagogical simplicity in a 90-minute workshop.

### `scripts/build_churn_db.py`
- Deterministic order synthesis via seeded RNG per customer ID
- Verification queries at the end confirm row counts and data integrity
- Null imputation values documented in code and saved in model artifact

### `scripts/build_ph_db.py` (legacy)
- Only relevant to `workshop` branch (builds `ecommerce_ph.db` from Olist data)
- Still tracked on `main` — no harm, but contributes to clutter

---

## Recommendations Summary

### Before publishing (required)
1. Rotate the Anthropic API key if it was ever at risk of exposure
2. Add a `LICENSE` file (MIT recommended for a workshop repo)

### Before publishing (recommended)
3. Remove `data/ecommerce_ph.db` from `main` — it's 31 MB of dead weight that only `workshop` uses
4. Update `HANDOFF.md` on `main` to reflect the Kaggle churn dataset migration

### Nice to have
5. Add a `.gitignore` exception for `datasets/data_ecommerce_customer_churn.csv` to make the tracking intentional rather than implicit
6. Consider Git LFS for binary files if the repository grows beyond its current ~46 MB

---

## Acknowledged Risks (accepted for workshop scope)

### `joblib.load()` pickle trust boundary
`joblib.load()` at `app.py:20` can execute arbitrary code embedded in a malicious pickle. In this repo the model artifact is committed and not user-uploaded, so the risk is limited to a compromised PR or supply-chain attack on the `.pkl` file. For a public workshop repo, this is an acceptable tradeoff — switching to a safer format like `skops` would add complexity without pedagogical benefit.

### Preprocessing data leakage
The notebook computes imputation medians and one-hot encoding on the full dataset before `train_test_split` (cells 10-11). This means test-set statistics influence the trained model, making reported metrics slightly optimistic. The correct approach is to fit transformers on `X_train` only via a scikit-learn `Pipeline`. Fixing this requires restructuring the notebook and retraining the model — deferred because the impact is small and the simpler code is easier to teach in 90 minutes.
