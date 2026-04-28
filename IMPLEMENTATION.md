# Implementation Roadmap

Build-out plan for the **AI-Powered Data Apps: From Jupyter to Production** workshop. Reads `HANDOFF.md` as the source of truth — this file is the *how* and *in what order*, not the *what*.

---

## Goal Recap

Ship an end-to-end demo for UST BS-DSA students: a working Streamlit churn-prediction app over a PH-localized Olist e-commerce SQLite DB, with a Jupyter notebook showing the prototype path and a clean `workshop` branch where Claude Code generates `app.py` live. Per `HANDOFF.md` §1–§3.

---

## Phased Build Order

Each phase produces a checkpoint. Don't start the next phase until the current one passes its verification line.

### Phase 0 — Environment (user)

User runs manually:

```bash
uv venv
uv pip install -r requirements.txt
```

**Done when:** `python -c "import streamlit, anthropic, pandas, sklearn, joblib"` succeeds inside the venv.

### Phase 1 — Data acquisition + PH localization

- Download Olist SQLite from `https://www.kaggle.com/datasets/terencicp/e-commerce-dataset-by-olist-as-an-sqlite-database` (HANDOFF §4).
- Write `scripts/build_ph_db.py` that:
  - Loads the Brazilian source DB.
  - Remaps `geolocation_*` / customer city to PH cities (Manila, Quezon City, Makati, Cebu, Davao, Pasig, Taguig, …).
  - Multiplies BRL prices by the FX rate (HANDOFF says ×9.5 — **confirm before workshop**).
  - Joins `product_category_name_translation.csv` for English category names.
  - Pre-computes `days_since_last_order` per customer.
  - Writes the simplified 3-table schema (`customers`, `orders`, `products`) per HANDOFF §4 to `data/ecommerce_ph.db`.
- Add `data/` to `.gitignore` if the DB exceeds GitHub's 100 MB limit; otherwise commit it for one-shot clone-and-run.

**Done when:** `sqlite3 data/ecommerce_ph.db ".tables"` lists exactly `customers`, `orders`, `products`, and a `SELECT COUNT(*) FROM orders` returns ~100k rows.

### Phase 2 — Notebook

`notebooks/01_churn_model.ipynb`:

1. Load DB with `pd.read_sql()` (never `read_csv` — HANDOFF §12).
2. Brief EDA: order-volume histogram, churn class balance, category mix. Use `matplotlib`/`seaborn`.
3. Build feature frame: `days_since_last_order`, `total_orders`, `avg_order_value`, `cancellation_rate`, target `is_churned` (no orders in last 90 days). Per HANDOFF §5.
4. 80/20 split, train `RandomForestClassifier` (or `LogisticRegression` if the talk track wants something more interpretable).
5. Print accuracy / confusion matrix / feature importances.
6. `joblib.dump(model, 'models/churn_model.pkl')`.

**Done when:** notebook runs top-to-bottom with no errors and `models/churn_model.pkl` exists.

### Phase 3 — Pre-trained model artifact

Just the output of Phase 2's final cell. Commit `models/churn_model.pkl` (small, <5 MB expected).

**Done when:** `joblib.load('models/churn_model.pkl').predict_proba(...)` returns a valid probability for a sample feature row.

### Phase 4 — `app.py` on `main`

Streamlit app (HANDOFF §6, §10):

- Customer-ID lookup via `st.text_input` or `st.selectbox` from a sample list.
- `pd.read_sql()` to fetch that customer's order history; compute the 4 features inline.
- Load `models/churn_model.pkl`; call `predict_proba` → risk score.
- Display score as a percentage with a colored badge.
- Call `anthropic.Anthropic().messages.create(...)` with the prompt template from HANDOFF §6 to generate a recommendation. Direct SDK only — **no LangChain** (HANDOFF §12).
- Read `ANTHROPIC_API_KEY` via `python-dotenv` from a `.env` file (gitignored).
- Keep the file beginner-readable: clear sections, short comments, no premature abstraction.

**Model ID — open decision:** HANDOFF §6 specifies `claude-sonnet-4-20250514` and labels it "Sonnet 4.6", but that ID is Sonnet **4.0**. The current Sonnet 4.6 ID is `claude-sonnet-4-6`. Default to **`claude-sonnet-4-6`** unless the user prefers 4.0 for cost reasons; flag the choice in the notebook talk track.

**Done when:** `streamlit run app.py` opens, a customer lookup returns a churn % and a Claude-generated recommendation in <5 s on a typical wifi connection.

### Phase 5 — Branch split

```bash
git checkout main
# ensure everything works on main
git checkout -b workshop
git rm app.py
git commit -m "Workshop branch: remove app.py for live Claude Code demo"
git push -u origin workshop
git checkout main
```

Per HANDOFF §7. The `workshop` branch keeps the notebook, model, data, and `requirements.txt` — only `app.py` is missing.

**Done when:** `git checkout workshop` shows the repo without `app.py` but every other artifact intact; `git checkout main` restores it.

### Phase 6 — Dry-run rehearsal

Walk the full workshop timeline (HANDOFF §3) end-to-end on a fresh clone:

1. Clone repo → `uv venv` → `uv pip install -r requirements.txt`.
2. On `main`: `streamlit run app.py` — verify the Hook demo works.
3. Open notebook, run top-to-bottom — verify Segment 2.
4. Switch to `workshop` branch — confirm `app.py` is gone.
5. Run Claude Code with the HANDOFF §10 prompt — confirm it generates a working `app.py`.
6. `streamlit run app.py` on the generated file — confirm Segment 5 works.

**Done when:** all six steps pass with no manual fix-ups.

---

## File-by-file Targets

Final layout (HANDOFF §9):

```
UST-DSA-Python-Workshop/
├── data/
│   └── ecommerce_ph.db
├── notebooks/
│   └── 01_churn_model.ipynb
├── models/
│   └── churn_model.pkl
├── scripts/
│   └── build_ph_db.py          # Phase 1 helper, not in HANDOFF but needed to reproduce data
├── app.py                       # exists on main; absent on workshop branch
├── requirements.txt             # already created
├── .env.example                 # template; real .env is gitignored
├── .gitignore
├── HANDOFF.md
├── IMPLEMENTATION.md            # this file
└── README.md
```

---

## Critical Decisions to Lock Before Phase 1

| Decision | Recommendation | Notes |
|---|---|---|
| Python version | **3.12** | All pinned packages support it; widely available via UV. |
| Olist source | **SQLite mirror** linked in HANDOFF §4 | Skips a CSV→SQLite conversion step. |
| FX rate (BRL→PHP) | Confirm before workshop | HANDOFF says ×9.5 — verify against a current FX feed on workshop morning. |
| Anthropic model ID | **`claude-sonnet-4-6`** | HANDOFF §6 has a label/ID mismatch (see Phase 4 above). |
| Commit `data/` to git? | Decide based on file size | If `ecommerce_ph.db` < 100 MB, commit it for one-shot clones. Otherwise gitignore + add a download script. |

---

## What Stays Out (per HANDOFF §12)

- No LangChain or any LLM orchestration framework.
- No `pd.read_csv` — always `pd.read_sql` against SQLite.
- No retraining inside the Streamlit app.
- No auth, multi-user support, or caching layers.
- No `order_items` table — schema is intentionally flat at 3 tables.
- No premature abstractions; the code must read cleanly to a 1st–4th year DSA student.

---

## Verification Checklist

A future Claude Code session (or the user) can pick this up by running this list top-to-bottom:

- [ ] `uv venv && uv pip install -r requirements.txt` succeeds.
- [ ] `data/ecommerce_ph.db` exists with 3 tables and ~100k orders.
- [ ] `notebooks/01_churn_model.ipynb` runs end-to-end and writes `models/churn_model.pkl`.
- [ ] `streamlit run app.py` on `main` works and calls Claude successfully (with `.env` populated).
- [ ] `workshop` branch exists, lacks `app.py`, retains everything else.
- [ ] Full dry-run of the HANDOFF §3 timeline passes on a fresh clone.
