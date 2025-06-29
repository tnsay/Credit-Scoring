## Credit Scoring Business Understanding.
### 1. Why is model interpretability important in a regulated financial environment?

Under frameworks like the Basel II Capital Accord, banks are required to demonstrate how they assess credit risk and make loan decisions. This makes **model interpretability critical**. Financial institutions must be able to **explain their risk models** to regulators, customers, and internal auditors. Transparent models also help build trust and fairness, especially in decisions that affect credit access.

For this reason, interpretable models such as **Logistic Regression with Weight of Evidence (WoE)** are widely used, since they clearly show how features contribute to a risk score.

---

### 2. Why is a proxy target necessary in this context?

The dataset lacks a direct label indicating whether a customer has defaulted. However, to train a model, we need a binary outcome variable. Therefore, we create a **proxy target variable** based on customer behavior patterns such as **Recency**, **Frequency**, and **Monetary (RFM)** metrics.

Using **KMeans clustering**, we can group customers into segments and label the least active, least valuable segment as “high-risk.” This lets us simulate the idea of default using **behavioral proxies**.

However, there are **business risks** to using a proxy target — if the proxy is inaccurate, the model may make incorrect decisions (e.g., rejecting a good borrower).

---

### 3. Trade-offs: Interpretable vs. Complex Models

| Model Type            | Pros                                   | Cons                                  |
|-----------------------|----------------------------------------|---------------------------------------|
| Logistic Regression + WoE | Interpretable, Auditable, Regulatory friendly | May lack accuracy on complex patterns |
| Gradient Boosting (e.g., XGBoost, LightGBM) | High accuracy, captures nonlinear interactions | Poor interpretability, hard to justify to regulators |

In regulated industries like banking, it's often necessary to start with interpretable models. But in less regulated, high-innovation environments, complex models may be favored for performance.

---
