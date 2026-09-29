# GARUDA-X

**AI-Powered Customer Risk & Revenue Decision Engine**

GARUDA-X is a modular foundation for analyzing structured business data, estimating customer churn risk and revenue at risk, explaining model outputs, and presenting evidence-backed recommendations for human review.

## Purpose and architecture

The planned system combines structured account signals and CRM evidence to produce interpretable risk assessments and playbook recommendations. Modules are separated by responsibility: `config/` for settings; `src/data/` for loading and feature preparation; `src/prediction/` for churn risk; `src/explainability/` and `src/nlp/` for SHAP explanations and CRM evidence; `src/decision/` for playbooks and approval; `src/audit/` for audit events; and `ui/` for the Streamlit interface.

The ML plan uses XGBoost, SHAP, and a lightweight Transformer. The Stage 3 XGBoost baseline is trained only on the synthetic monthly churn dataset. Generated model artifacts are local and ignored by Git. Its probabilities are not calibrated. Recommendations remain advisory: human approval is mandatory before any action can be executed.

## Current status

Stages 1–6 are implemented: the project foundation, structured data pipeline, XGBoost next-month churn baseline, revenue-at-risk calculation, SHAP explanations, and local MiniLM CRM evidence retrieval. The baseline uses chronological train, validation, and test partitions; its artifacts and metrics are described in `models/xgboost_model/model_metadata.json`. CRM retrieval is account-scoped and uses only the local MiniLM model. The available conversation corpus has no account IDs or dates, so it is not attached to modeled accounts and returns no matches. Decision playbooks, mandatory human approval, persistent audit storage, and the complete Streamlit workflow remain planned.

## Planned modules

1. Evaluate whether probability calibration is needed as a separate stage.
2. Build decision playbooks and mandatory human approval workflow.
3. Add durable audit logging, monitoring, and deployment controls.

## Run

Install dependencies from `requirements.txt`. Run `streamlit run app.py` to open the current placeholder UI. Run the test suite with `python -m pytest`; CRM retrieval loads the checked-in MiniLM directory locally and does not download a model. See `docs/` for architecture, schema, decision logic, and evaluation notes.
