# GARUDA-X

**AI-Powered Customer Risk & Revenue Decision Engine**

GARUDA-X predicts next-month customer churn risk from structured business data, estimates revenue at risk, explains model signals, retrieves CRM evidence when it can be matched to an account, and creates deterministic playbook recommendations for human review. It is a business decision engine, not a chatbot. It never performs a customer-facing action.

## Architecture and current status

Stages 1–8 provide modular data loading, validation and feature preparation; a saved XGBoost churn predictor; revenue-at-risk calculations; TreeSHAP explanations; local MiniLM retrieval; deterministic decision playbooks; mandatory human approval; and append-only audit events. Stage 9/10 adds the Streamlit Command Center, Accounts, Decisions, Data and Audit Log views, with analysis orchestration in `src/application.py`.

The probability output is uncalibrated. SHAP contributions are in the model's raw-margin log-odds space; they are not probability contributions or causal findings. Decisions and audit records are held in memory for the active Streamlit session. Human approval is mandatory before execution authorization can be recorded; GARUDA-X does not execute the approved action.

The current CRM corpus does not contain `user_id` or `account_id`. No interactions are matched by row order, similarity alone, or any invented mapping. Account-scoped CRM evidence may therefore show **No matching CRM evidence found**. This means evidence is unavailable for that account, not that the account has no issues.

## Repository structure

```text
app.py                         Streamlit entrypoint
config/                        Settings and decision thresholds
data/                          Dataset documentation; business data is supplied by the operator
models/xgboost_model/           Saved churn model, preprocessor and metadata
models/transformer/             Local MiniLM deployment model
src/data/                       Loaders, validators, cleaning, feature preparation
src/prediction/                 Churn scoring and revenue-at-risk
src/explainability/             TreeSHAP and structured evidence
src/nlp/                        Local Transformer and scoped CRM retrieval
src/decision/                   Recommendation, playbooks and human approval
src/audit/                      In-memory append-only audit events
src/application.py              Monthly analysis orchestration
ui/                             Streamlit pages and reusable components
tests/                          Backend and UI-foundation tests
docs/                           Architecture, schema, decision and evaluation docs
```

## Local setup

Use Python 3.12, the default version currently selected by Streamlit Community Cloud. Create an isolated environment and install the root dependency file:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.

Run the app from the repository root:

```bash
streamlit run app.py
```

The initial page is lightweight. Upload a CSV or Parquet user-month dataset in **Data**, inspect the schema report, choose a reporting month and select **Run analysis**. GARUDA-X loads saved models on demand and reuses them within the Streamlit process. No model is trained at app startup.

## Dataset requirements

Inference expects one row per `user_id` and `month`. The trained model requires `mrr`, `sessions`, `feature_usage_score`, `support_tickets`, `payment_failures`, `nps_score`, `product_incident`, `active_seats`, `tenure_month` and `plan_type`. The outcome column `churned_next_month` is not required for inference. Upload business data through the Data page; customer datasets are not bundled as a demo. The selected month is analyzed to keep historical panels efficient.

CRM retrieval uses the existing project JSONL reader when its source file is available. Records need a real `user_id` or `account_id` to be returned for that scope. Unscoped records are not linked to modeled users.

## Model files

The XGBoost model, preprocessor and metadata are committed under `models/xgboost_model/`. MiniLM is loaded only from `models/transformer/all-MiniLM-L6-v2-deploy/`; the Transformer loader uses local-only mode and never downloads a replacement. The original `models/transformer/all-MiniLM-L6-v2/` reference copy is excluded from Git. The requirements use a CPU-only PyTorch wheel; CUDA is not required.

## Streamlit Community Cloud deployment

1. In Streamlit Community Cloud, choose **Create app**.
2. Select repository `PVamsi2211/Garuda-X`, branch `main`, and entrypoint `app.py`.
3. In **Advanced settings**, select Python 3.12.
4. Deploy. Community Cloud starts from the repository root and installs `requirements.txt`.

No secrets or Linux system packages are currently required. The app requires the saved XGBoost artifacts and committed MiniLM deployment model in the checkout. No local caches, developer paths, external APIs or runtime model downloads are used. Community Cloud's filesystem and this project's decision/audit state are not durable storage; decisions and audit entries are lost when the session or app process restarts. Do not treat the in-memory audit log as a persistent compliance record.

## Tests and documentation

Run `python -m pytest` from the repository root. See [architecture](docs/architecture.md), [data schema](docs/data_schema.md), [decision logic](docs/decision_logic.md) and [evaluation](docs/evaluation.md) for backend details.
