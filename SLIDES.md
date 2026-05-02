# Slide #1
- Title Card,
- Introduction, Speaker picture

# Slide #2
- General Overview of the Workshop

# Slide #3 — Data Split Strategy

- 80/20 stratified split: **Train (3,152)** vs **Test (789)**
- GridSearchCV uses 5-fold CV on the training set:
  - Each iteration: 64% train / 16% validation / 20% test (held out)
  - Validation fold rotates across all 5 folds
- Test set is **never** seen during training or tuning — only used once for final evaluation
- Stratified split preserves the ~17% churn rate in both sets

# Slide #4 — Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| Tenure | Numeric | Months the customer has been on the platform |
| WarehouseToHome | Numeric | Distance (km) from warehouse to customer's home |
| NumberOfDeviceRegistered | Numeric | Devices registered to the customer's account |
| SatisfactionScore | Numeric (1–5) | Customer satisfaction rating |
| NumberOfAddress | Numeric | Addresses saved on the customer's account |
| Complain | Binary (0/1) | Whether the customer filed a complaint last month |
| DaySinceLastOrder | Numeric | Days since the customer's most recent order |
| CashbackAmount | Numeric (₱) | Average cashback received |
| PreferedOrderCat | Categorical (6) | Most frequently ordered category (Laptop & Accessory, Mobile Phone, Mobile, Fashion, Grocery, Others) |
| MaritalStatus | Categorical (3) | Single, Married, or Divorced |
| **Churn** | **Binary (Target)** | **1 = churned, 0 = retained (~17% churn rate)** |

Source: [Kaggle — E-Commerce Customer Churn](https://www.kaggle.com/datasets/samuelsemaya/e-commerce-customer-churn)

# Slide #5


# What Claude did not catch during this AI-assisted coding session:
- Shapley Values, Claude used Permutation Feature Importance, can't even make Shap library work I intervened and did it for like ~3minutes and it worked.
- Did not visualize Complaint and Marital Status.
