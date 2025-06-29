##  Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord emphasizes rigorous **risk measurement, transparency, and accountability** for financial institutions. Banks must not only assess credit risk accurately but also **explain and justify** the models they use for decision-making, especially in scenarios like loan approvals or rejections.

This creates a strong need for models that are **interpretable and well-documented**. A model’s logic and inputs must be clearly understandable to internal stakeholders, auditors, and regulators. Therefore, traditional techniques like **Logistic Regression with Weight of Evidence (WoE)** are preferred in many banking environments. These models provide clear insights into how each variable affects credit decisions and allow for robust documentation and validation.

---

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In this project, the dataset does not contain a direct indicator of whether a customer defaulted on a loan. However, supervised machine learning requires labeled data. Therefore, we must create a **proxy target variable** to represent likely defaulters.

To do this, we use **customer behavior patterns**, particularly RFM (Recency, Frequency, Monetary) metrics, and apply **KMeans clustering** to segment customers. The least active, least valuable cluster is then labeled as "high-risk," serving as a **proxy for default**.

While this enables model training, it introduces **business risks**:
- The proxy label may not accurately reflect actual credit behavior.
- The model may misclassify reliable customers as high risk, potentially leading to lost revenue or customer dissatisfaction.
- Conversely, it may approve risky customers due to proxy misalignment.

Hence, it's important to **validate the proxy definition carefully** and understand its limitations when making real-world credit decisions.

---

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

| Model Type                         | Pros                                                  | Cons                                                   |
|------------------------------------|-------------------------------------------------------|--------------------------------------------------------|
| Logistic Regression + WoE          | High interpretability, regulatory compliance, easy to document | May underperform on complex, nonlinear data             |
| Gradient Boosting (e.g., XGBoost)  | High predictive power, handles complex interactions   | Poor interpretability, harder to audit and explain     |

In highly regulated sectors like finance, **interpretability and explainability often take priority over raw performance**. Logistic Regression allows institutions to maintain transparency and explain decisions, which aligns with regulatory expectations.

On the other hand, Gradient Boosting methods may offer better predictive accuracy, especially with complex patterns in customer behavior. However, their "black-box" nature makes them difficult to justify in regulatory audits unless paired with explainability tools (e.g., SHAP, LIME). 

A common industry strategy is to **start with an interpretable baseline model** and only move to more complex models with appropriate justification, explainability techniques, and internal validation.

---
