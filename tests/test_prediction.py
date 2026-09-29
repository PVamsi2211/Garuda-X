import numpy as np
import pandas as pd
import pytest

from config.thresholds import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD
from src.data.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    prepare_churn_features,
    prepare_prediction_features,
)
from src.prediction.predictor import load_churn_predictor
from src.prediction.risk_scoring import classify_risk
from src.prediction.train import (
    TemporalSplit,
    baseline_model_parameters,
    build_model_metadata,
    build_preprocessor,
    calculate_scale_pos_weight,
    chronological_split,
    evaluate_probabilities,
    fit_xgboost_model,
    save_model_artifacts,
)


def monthly_fixture(rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [f"u-{index}" for index in range(rows)],
            "month": pd.to_datetime(["2025-01-01"] * rows),
            "mrr": np.arange(rows, dtype=float) + 10,
            "sessions": np.arange(rows) + 1,
            "feature_usage_score": np.arange(rows) * 5.0,
            "support_tickets": np.arange(rows) % 3,
            "payment_failures": np.arange(rows) % 2,
            "plan_type": [" Basic " if index % 2 else "Pro" for index in range(rows)],
            "nps_score": np.arange(rows) % 11,
            "product_incident": np.arange(rows) % 2,
            "active_seats": np.arange(rows) + 1,
            "tenure_month": np.arange(rows),
            "churned": np.arange(rows) % 2,
            "churn_date": [None] * rows,
            "future_month": ["2025-02-01"] * rows,
            "churned_next_month": np.arange(rows) % 2,
        }
    )


def small_model_and_artifacts(tmp_path):
    X_train = pd.DataFrame(
        {
            "mrr": np.arange(80, dtype=float),
            "plan_type": ["basic" if index % 2 else "pro" for index in range(80)],
        }
    )
    y_train = pd.Series(np.arange(80) % 4 == 0, dtype="int8").astype("int8")
    X_validation = pd.DataFrame(
        {
            "mrr": np.arange(80, 120, dtype=float),
            "plan_type": ["basic" if index % 2 else "pro" for index in range(40)],
        }
    )
    y_validation = pd.Series(np.arange(40) % 4 == 0, dtype="int8").astype("int8")
    X_test = pd.DataFrame(
        {
            "mrr": np.arange(120, 160, dtype=float),
            "plan_type": ["basic" if index % 2 else "unseen" for index in range(40)],
        }
    )
    y_test = pd.Series(np.arange(40) % 5 == 0, dtype="int8").astype("int8")

    preprocessor = build_preprocessor(["mrr"], ["plan_type"])
    train_values = preprocessor.fit_transform(X_train)
    validation_values = preprocessor.transform(X_validation)
    test_values = preprocessor.transform(X_test)
    parameters = baseline_model_parameters(
        calculate_scale_pos_weight(y_train),
        random_seed=11,
        n_estimators=8,
        early_stopping_rounds=2,
    )
    model = fit_xgboost_model(
        train_values,
        y_train,
        validation_values,
        y_validation,
        model_parameters=parameters,
    )
    validation_metrics = evaluate_probabilities(
        y_validation, model.predict_proba(validation_values)[:, 1]
    )
    test_metrics = evaluate_probabilities(y_test, model.predict_proba(test_values)[:, 1])
    split = TemporalSplit(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-06-01"),
        pd.Timestamp("2025-07-01"),
        pd.Timestamp("2025-08-01"),
        pd.Timestamp("2025-09-01"),
        pd.Timestamp("2025-10-01"),
    )
    metadata = build_model_metadata(
        dataset_path="data/synthetic-saas-churn/user_monthly.parquet",
        users_count=120,
        feature_list=["mrr", "plan_type"],
        numeric_features=["mrr"],
        categorical_features=["plan_type"],
        split=split,
        model_parameters=parameters,
        scale_pos_weight=calculate_scale_pos_weight(y_train),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        best_iteration=int(model.best_iteration),
        best_validation_metric=float(model.best_score),
        preprocessing_feature_names=list(preprocessor.get_feature_names_out()),
        random_seed=11,
    )
    artifacts = save_model_artifacts(model, preprocessor, metadata, tmp_path)
    return X_validation, X_test, model, preprocessor, metadata, artifacts


def test_chronological_split_keeps_complete_months_and_target_alignment() -> None:
    dates = pd.Series(pd.date_range("2025-01-01", periods=10, freq="MS").repeat(10))
    original_month = dates.dt.month.to_numpy()
    original_labels = np.tile(np.arange(10) % 2, 10).astype("int8")
    order = np.arange(len(dates))[::-1]
    shuffled_dates = dates.iloc[order].reset_index(drop=True)
    features = pd.DataFrame({"month_number": original_month[order]})
    target = pd.Series(original_labels[order])

    split = chronological_split(features, target, shuffled_dates)

    assert split.train_start <= split.train_end < split.validation_start
    assert split.validation_end < split.test_start <= split.test_end
    assert split.X_train["month_number"].max() < split.X_validation["month_number"].min()
    assert split.X_validation["month_number"].max() < split.X_test["month_number"].min()
    assert len(split.X_train) + len(split.X_validation) + len(split.X_test) == len(features)
    assert split.y_train.nunique() == 2
    assert split.y_validation.nunique() == 2
    assert split.y_test.nunique() == 2


def test_feature_allowlist_and_unlabeled_preparation_prevent_leakage() -> None:
    raw = monthly_fixture()
    features, target = prepare_churn_features(raw)
    inference_features = prepare_prediction_features(raw.drop(columns=TARGET_COLUMN))

    expected = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    assert features.columns.tolist() == expected
    assert inference_features.columns.tolist() == expected
    assert target.name == TARGET_COLUMN
    assert TARGET_COLUMN not in features
    assert not {"churned", "churn_date", "future_month", "user_id", "month"}.intersection(
        features.columns
    )
    assert features.equals(inference_features)


def test_target_validation_and_training_class_weight() -> None:
    raw = monthly_fixture()
    raw.loc[0, TARGET_COLUMN] = 2
    with pytest.raises(ValueError, match="only 0 or 1"):
        prepare_churn_features(raw)
    with pytest.raises(ValueError, match="binary"):
        calculate_scale_pos_weight(pd.Series([0, 2]))
    assert calculate_scale_pos_weight(pd.Series([0, 0, 0, 0, 1])) == 4.0


def test_risk_categories_use_configurable_business_thresholds() -> None:
    assert classify_risk(MEDIUM_RISK_THRESHOLD - 0.001) == "low"
    assert classify_risk(MEDIUM_RISK_THRESHOLD) == "medium"
    assert classify_risk(HIGH_RISK_THRESHOLD - 0.001) == "medium"
    assert classify_risk(HIGH_RISK_THRESHOLD) == "high"
    with pytest.raises(ValueError, match="thresholds"):
        classify_risk(0.5, medium_threshold=0.8, high_threshold=0.7)


def test_xgboost_training_serialization_prediction_metadata_and_feature_order(
    tmp_path,
) -> None:
    X_validation, X_test, model, preprocessor, metadata, artifacts = small_model_and_artifacts(
        tmp_path
    )
    restored = load_churn_predictor(tmp_path)
    restored_probabilities = restored.predict_proba(X_test)
    direct_probabilities = model.predict_proba(preprocessor.transform(X_test))[:, 1]

    assert model.best_iteration >= 0
    assert artifacts["model"].is_file()
    assert artifacts["preprocessor"].is_file()
    assert artifacts["metadata"].is_file()
    assert metadata["feature_list"] == ["mrr", "plan_type"]
    assert metadata["target_column"] == TARGET_COLUMN
    assert metadata["scale_pos_weight"] == 3.0
    assert metadata["calibration"]["calibrated"] is False
    assert np.allclose(restored_probabilities.to_numpy(), direct_probabilities)
    assert restored_probabilities.between(0, 1).all()
    assert restored.predict_proba(X_validation).between(0, 1).all()

    with pytest.raises(ValueError, match="Feature order mismatch"):
        restored.predict_proba(X_test[["plan_type", "mrr"]])
    with pytest.raises(ValueError, match="Feature columns mismatch"):
        restored.predict_proba(X_test.drop(columns="mrr"))


def test_baseline_configuration_records_weight_and_seed() -> None:
    parameters = baseline_model_parameters(7.5, random_seed=23)
    assert parameters["objective"] == "binary:logistic"
    assert parameters["scale_pos_weight"] == 7.5
    assert parameters["random_state"] == 23
    assert parameters["n_estimators"] == 400
