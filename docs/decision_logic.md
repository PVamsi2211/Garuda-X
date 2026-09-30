# Decision logic

Risk thresholds are configurable in `config/thresholds.py`. Recommendations combine a risk band, evidence, and a playbook and remain pending until a named human approves them.

Any future execution integration must check approved status. Model output alone must never execute a recommended action.

## Stage 6 CRM evidence retrieval

MiniLM embeddings are generated locally from models/transformer/all-MiniLM-L6-v2 with sentence-transformers and local-files-only loading. Text is whitespace-normalized only for embedding; the original CRM text is preserved for returned evidence. The model is cached and reused. Embeddings are normalized to unit length, so cosine similarity is calculated as a dot product. Similarity is a relevance score, not a confidence, probability, truth score, or churn probability.

The index filters records by exact user_id or account_id before computing query similarities. Results are sorted by similarity descending, limited by top_k (default 3), and can optionally be filtered by min_similarity. An optional as_of timestamp excludes undated and later interactions. Interaction IDs, dates, channel/type, source row, and original text are preserved when available. Missing text is skipped; missing identifiers are never synthesized.

Retrieval queries are deterministic and use only approved Stage 3 feature values from Stage 5's top positive SHAP drivers, with fixed feature vocabulary. Churn labels, churn dates, future features, and outcomes are not used. Stage 5 structured feature evidence remains separate from Stage 6 CRM interaction evidence.

The project data inspection found no account-linked CRM table in data/synthetic-saas-churn: its schema contains only users and user_monthly. The only local interaction text is data/customer-churn-v2/full.jsonl, whose schema provides conversation_id, channel, and conversation text but no user_id, account_id, or interaction date. It therefore cannot be safely attached to a Stage 3 account or used for an as-of prediction example. The other support table is aggregated and has no interaction text; its A-… identifiers do not match Stage 3 user IDs. Such unlinked records are excluded from the searchable index and retrieval returns no_matching_crm_records rather than evidence from another account.

CRM retrieval provides supporting contextual evidence; it does not prove causality and does not alter the XGBoost prediction.

## Stage 5 explanation evidence

Stage 5 provides structured explanations from the saved XGBoost model's approved current-month features. Positive SHAP values pushed that prediction toward higher model churn risk; negative values pushed it toward lower model churn risk. Contributions are in raw margin (log-odds) space. They describe model behavior and do not establish causality, prove that churn will occur, or show that changing a feature will change the outcome.

Evidence items are deterministic renderings of feature values and their SHAP contributions. They are structured model evidence only. Stage 5 performs no CRM retrieval and does not imply that CRM evidence exists. It introduces no actionability threshold or recommendation.

## Stage 7 decision engine

`src.decision.playbooks` maps the existing Stage 3 risk bands and a configurable MRR threshold to exactly five versioned playbooks. Risk is LOW below `MEDIUM_RISK_THRESHOLD`, MEDIUM at or above that threshold and below `HIGH_RISK_THRESHOLD`, and HIGH at or above the high threshold. Revenue uses two bands: LOW_REVENUE when current MRR is below `HIGH_REVENUE_THRESHOLD`, and HIGH_REVENUE at or above it. The initial configurable revenue cutoff is 500 MRR units in `config/thresholds.py`; it should be set to the business-approved cutoff for the data's currency and units. LOW risk always selects `LOW_RISK_MONITOR`.

The baseline playbooks are `HIGH_RISK_HIGH_REVENUE`, `HIGH_RISK_LOW_REVENUE`, `MEDIUM_RISK_HIGH_REVENUE`, `MEDIUM_RISK_LOW_REVENUE`, and `LOW_RISK_MONITOR`. Each has a deterministic priority, internal review action, owner, required evidence, safeguards, version, and human-approval execution policy. Playbook selection is rule-based and does not use CRM similarity or SHAP magnitude.

`build_recommendation` accepts only current risk score, current MRR, supplied account/user identifiers, actual Stage 5 SHAP output, Stage 6 CRM retrieval output, and an optional as-of timestamp. It uses Stage 4 `calculate_revenue_at_risk` for `risk_score * current MRR`. The score is a model-predicted churn-risk score and is not calibrated. Scores must be finite and within [0, 1]; MRR and revenue at risk must be finite and non-negative. Each recommendation has a unique decision ID, UTC ISO-8601 creation time, risk and revenue bands, primary positive SHAP drivers from the Stage 3 feature allowlist, traceable CRM evidence, playbook details, rationale, and `PENDING` status with `requires_human_approval = true`. Missing identifiers are left as null; at least one of `account_id` or `user_id` is required.

CRM evidence is retained only when it has a recognized CRM source, a matching account or user scope, traceable source ID, text, similarity score, and a non-future date. Similarity remains a retrieval relevance score. If no qualifying evidence remains, `evidence_status` is `no_matching_crm_records`, and the rationale states CRM evidence was unavailable. No CRM narrative or customer intent is generated. SHAP drivers are reported as primary contributing signals, never as causal proof. Outcome, churn, future-revenue, future-MRR, and unrecognized fields are not accepted as recommendation inputs.

## Stage 8 human approval and audit

`DecisionService` keeps decision records in an encapsulated in-memory store and returns copies. Its Stage 9-facing getters include `list_decisions`, `get_decision`, `get_pending_decisions`, `get_accounts`, `get_decision_history`, and `get_audit_history`. The service does not depend on Streamlit or a database.

The only decision transitions are `PENDING → APPROVED`, `PENDING → MODIFIED`, and `PENDING → REJECTED`. Every transition requires a named human `approver_id`; AI, system, anonymous, and empty identities are rejected. Modification requires a non-empty final action and reason and retains the original recommended action. Rejection requires a reason and retains the original recommendation. Finalized decisions cannot transition again.

`authorize_execution` is a fail-closed authorization record, not an action executor. It requires an approved or modified decision, a matching human approval audit event, a final action, and no prior authorization. Pending and rejected decisions cannot pass. Duplicate authorization is rejected. The decision remains in its approved or modified state, while the audit lifecycle records `EXECUTION_AUTHORIZED`; no email, CRM, billing, account, or customer action is performed.

The append-only in-memory audit service records `RECOMMENDATION_CREATED`, `DECISION_APPROVED`, `DECISION_MODIFIED`, `DECISION_REJECTED`, and `EXECUTION_AUTHORIZED`. Events retain the same decision ID, account/user identifiers, UTC timestamp, prior and new status, approver, original recommendation, final action, reason, and relevant metadata. Event status order is checked, returned histories are copies, and audit history has no normal deletion or editing operation. A rejected decision can never have an execution authorization event.

All recommendation and transition validation occurs before state changes. Invalid inputs, invalid transitions, absent approval evidence, corrupt authorization details, and duplicate requests fail without opening the execution gate. The system remains decision support: AI recommends, a human decides, and the system records authorization only. Stage 9 UI and external action execution are not implemented.
