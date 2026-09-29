# Evaluation

## Stage 3 XGBoost baseline

The model uses only `data/synthetic-saas-churn/user_monthly.parquet`, with `churned_next_month` as its target. Rows are split by complete chronological months. Preprocessing is fit on the training partition; the validation partition drives early stopping; the test partition is evaluated after model selection. No CRM or account dataset is joined.

| Partition | Dates | Rows | Positive labels | Positive prevalence |
| --- | --- | ---: | ---: | ---: |
| Train | 2024-01-01 to 2025-06-01 | 55,208 | 716 | 1.297% |
| Validation | 2025-07-01 to 2025-09-01 | 12,610 | 209 | 1.657% |
| Test | 2025-10-01 to 2025-12-01 | 11,999 | 131 | 1.092% |

The training-only negative-to-positive ratio sets `scale_pos_weight` to 76.1061. Early stopping selected iteration 15 using validation average precision (`aucpr`). Classification metrics below use a probability threshold of 0.50 for initial comparison only; this is not an optimized business threshold.

| Metric | Validation | Final test |
| --- | ---: | ---: |
| ROC-AUC | 0.7940 | 0.7912 |
| PR-AUC (average precision) | 0.0578 | 0.0366 |
| Precision | 0.0401 | 0.0250 |
| Recall | 0.7943 | 0.7786 |
| F1 | 0.0763 | 0.0484 |
| Positive support | 209 | 131 |
| Confusion matrix `[[TN, FP], [FN, TP]]` | `[[8425, 3976], [43, 166]]` | `[[7889, 3979], [29, 102]]` |

The test PR-AUC exceeds the 1.092% positive prevalence baseline, but precision is weak at the 0.50 threshold and the classifier produces many false positives. Threshold selection needs business costs and validation-only analysis. The probabilities are **not calibrated**; no calibration method or Brier score was used.

The run metadata, parameters, feature order, split summaries, and metrics are stored in `models/xgboost_model/model_metadata.json`. These results are a baseline, not evidence of production readiness.

## Stage 5 structured explainability

The saved Stage 3 model is explained with XGBoost's exact TreeSHAP contributions through Booster.predict with pred_contribs=True and approx_contribs=False. Inputs use the Stage 3 predictor and saved preprocessor. The one-hot encoded plan_type contributions are summed back into the single Stage 3 feature, preserving additivity and the original ten-feature interface.

The contributions are in raw margin (log-odds) space, not probability space. The base value plus all feature contributions reconstructs the saved model's raw margin. For this early-stopped model, both contribution and raw-margin predictions use trees through best_iteration 15; their difference must be at most 1e-5. Applying the logistic function to that margin is checked against the Stage 3 predictor probability. Those probabilities remain uncalibrated.

Positive SHAP values indicate that a feature pushed this model prediction toward higher churn risk; negative values indicate a push toward lower churn risk. Features are ranked by absolute SHAP value, with feature name as a deterministic tie-breaker. These are descriptions of model behavior, not causality, proof of future churn, or evidence that changing a feature will change an outcome.

Stage 5 structured evidence is generated only from the approved current-month business features and their SHAP contributions. It does not retrieve CRM text or use MiniLM, Transformers, embeddings, or generated CRM summaries. Structured evidence does not imply CRM evidence exists.
