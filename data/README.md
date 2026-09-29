# Data

The data files below are separate datasets and are not directly joined. Their identifiers have no verified cross-dataset mapping. In particular, do not join synthetic `user_id`, revenue-risk `account_id`, or CRM `conversation_id` values.

## Synthetic SaaS churn

Path: `synthetic-saas-churn/`

- Contains `users.parquet` (5,000 users) and `user_monthly.parquet` (79,817 user-month rows).
- `user_id` identifies a user and repeats across monthly observations. The monthly row key is `(user_id, month)`.
- This is the primary ML dataset. The intended supervised setup is features observed at month *t* predicting `churned_next_month`.
- The `users` table contains last-known and churn-summary fields. Do not merge it into historical monthly features without validating temporal availability and leakage.

## SaaS revenue-risk analysis

Path: `saas-customer-churn-revenue-risk-analysis-main/`

- Contains account, subscription/revenue, support, usage, and insights-summary CSV files.
- `account_id` is the common identifier across the relevant account intelligence files. It repeats in subscription- and usage-level tables; use the table's actual grain when checking uniqueness.
- Reserved for business/account intelligence. It is not the primary churn-model training source at this stage.

## Customer Churn Dataset V2

Path: `customer-churn-v2/`

- Contains `train.jsonl`, `eval.jsonl`, and `full.jsonl` conversation records.
- `conversation_id` identifies a conversation; it is not a verified account or user identifier.
- Reserved for later CRM evidence retrieval/NLP work. No NLP pipeline is implemented in this stage.

Do not commit additional raw or private business data. The repository ignore rules exclude raw dataset folders from version control.
