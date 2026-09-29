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
