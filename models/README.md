# Models

Stage 3 writes the XGBoost churn model to `xgboost_model/xgboost_churn_model.json`, its fitted preprocessing to `xgboost_model/preprocessing_pipeline.joblib`, and evaluation metadata to `xgboost_model/model_metadata.json`.

Generated model artifacts are ignored by Git. The XGBoost probability output is not calibrated. The calibration and transformer directories are reserved for later stages; MiniLM is not used by the Stage 3 churn pipeline.
