# Alternative Dataset Options

Candidates to replace the current Olist-derived dataset with something more complex and feature-rich for churn prediction.

## Current Dataset (Olist)

- 3 simplified tables: customers, orders, products
- ~100k orders, ~96k customers, ~33k products
- Only 4 derived churn features: days_since_last_order, total_orders, avg_order_value, cancellation_rate
- No order_items join table, no reviews, no payment details, no browsing behavior, no demographics

---

## Candidate 1: REES46 Multi-Category Store (Recommended)

- **Source:** https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
- **Size:** ~42.4M event rows (~5 GB), spans Oct 2019 - Apr 2020
- **Columns:** event_time, event_type, product_id, category_id, category_code (hierarchical, e.g., electronics.smartphone), brand, price, user_id, user_session
- **Why it's better:** Full clickstream/browsing behavior (view, add-to-cart, remove-from-cart, purchase), session-level tracking, product brand and category hierarchy. Can derive richer churn features: session frequency, browse-to-buy ratio, cart abandonment rate, brand loyalty, category affinity, time-of-day patterns, inter-session gaps.
- **License:** CC BY-SA 4.0
- **Caveat:** Single flat event table (no separate customer demographics), but sessions and events can be pivoted into relational tables. Data is real, from the REES46 Marketing Platform.

## Candidate 2: Instacart Market Basket Analysis

- **Source:** https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis
- **Size:** 6 relational tables, ~3.4M orders, ~32.4M order-product rows, 200k+ users, 50k products
- **Tables:** orders (7 cols), order_products_prior (32.4M rows), order_products_train (1.4M rows), products (50k rows), aisles (134 rows), departments (21 rows)
- **Why it's better:** True multi-table relational schema with order_items-level granularity. Reorder flags, add-to-cart sequence, day-of-week and hour-of-day patterns, days-between-orders, 3-level product hierarchy (department > aisle > product).
- **License:** Custom (Instacart competition license, non-commercial research use)
- **Caveat:** Grocery domain only. No customer demographics, reviews, or payment info. Anonymized.

## Candidate 3: Retailrocket Recommender System Dataset

- **Source:** https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
- **Size:** ~2.76M events, large item properties file, hierarchical category tree
- **Columns:** timestamp, visitorid, event (view/addtocart/transaction), itemid, transactionid; item properties in key-value format; category tree with parent-child relationships
- **Why it's better:** Full behavioral funnel, rich item property metadata, hierarchical category tree. Can derive conversion funnels, item attribute preferences, category browsing depth.
- **License:** CC BY-NC-SA 4.0
- **Caveat:** All values are hashed for confidentiality — limits readability for a workshop setting.

## Candidate 4: Ecommerce Customer Churn (Ankit Verma)

- **Source:** https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
- **Size:** 5,630 rows, ~20 columns, single table
- **Columns:** CustomerID, Churn (target), Tenure, CityTier, WarehouseToHome, HoursSpendOnApp, NumberOfDeviceRegistered, PreferredLoginDevice, PreferredPaymentMode, PreferredOrderCategory, SatisfactionScore, MaritalStatus, Gender, NumberOfAddress, Complain, OrderAmountHikeFromLastYear, CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount
- **Why it's better:** Pre-labeled churn target. Rich customer-level features: demographics (gender, marital status, city tier), behavioral (app usage, device count), transactional (payment mode, coupon usage, cashback), and satisfaction/complaint signals.
- **License:** CC0 (Public Domain)
- **Caveat:** Only 5,630 rows. Single flat table, no order_items or product details.

## Candidate 5: REES46 Churn Dataset (Derived)

- **Source:** https://www.kaggle.com/datasets/fridrichmrtn/e-commerce-churn-dataset-rees46
- **Size:** Derived from Candidate 1, pre-processed for churn prediction
- **Why it's better:** Ready-made churn labels and pre-engineered features from real behavioral data, traceable back to the raw event stream.
- **License:** CC BY-SA 4.0
- **Caveat:** Being a derived dataset, fewer raw columns than the source.

---

## Recommendation

**REES46 Multi-Category Store (Candidate 1)** is the strongest pick — massive scale, real behavioral data, permissive license, and the most potential for deriving complex churn features beyond what Olist offers. Trade-off is rebuilding `scripts/build_ph_db.py` to transform event-level data into the SQLite schema.
