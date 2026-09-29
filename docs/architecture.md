# Architecture

The Streamlit UI is being developed to call modular data, prediction, explainability, NLP, decision, and audit components.

Planned flow: load and validate structured data; clean and engineer features; estimate churn risk and revenue at risk; explain scores with SHAP and retrieve relevant CRM evidence; build a playbook recommendation; require an identified human approver before execution; record predictions, decisions, approvals, and outcomes.

The current backend includes structured data preparation, an XGBoost predictor, revenue-at-risk calculation, SHAP explanations, and account-scoped MiniLM retrieval from a local model. The available synthetic conversation corpus lacks account identifiers and dates and is excluded from account retrieval. Decision playbooks, human approval integration, persistent audit storage, and action execution remain planned. Any eventual action execution must require explicit human approval.
