from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from config.datasets import get_dataset_path
from config.thresholds import (
    DEFAULT_PROBABILITY_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
)
from src.data.cleaner import clean_synthetic_monthly
from src.data.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    prepare_churn_features,
)
from src.data.loader import load_data
from src.data.validator import validate_synthetic_monthly

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "models" / "xgboost_model"
MODEL_FILENAME = "xgboost_churn_model.json"
PREPROCESSOR_FILENAME = "preprocessing_pipeline.joblib"
METADATA_FILENAME = "model_metadata.json"
RANDOM_SEED = 42
EARLY_STOPPING_ROUNDS = 35


@dataclass(frozen=True)
class TemporalSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def chronological_split(
    features: pd.DataFrame,
    target: pd.Series,
    months: pd.Series,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    """Split complete months chronologically near the requested row proportions."""
    if len(features) != len(target) or len(features) != len(months):
        raise ValueError("Features, target, and month values must have equal lengths")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("Target values must be non-null and binary")

    date_values = pd.Series(pd.to_datetime(pd.Series(months).reset_index(drop=True), errors="coerce"))
    if date_values.isna().any():
        raise ValueError("Month values must all be valid dates")
    if date_values.nunique() < 3:
        raise ValueError("At least three distinct months are required for temporal splits")

    order = np.argsort(date_values.to_numpy(), kind="stable")
    sorted_features = features.iloc[order].reset_index(drop=True)
    sorted_target = pd.Series(target).reset_index(drop=True).iloc[order].reset_index(drop=True)
    sorted_dates = date_values.iloc[order].reset_index(drop=True)
    month_counts = sorted_dates.value_counts(sort=False).sort_index()
    cumulative_rows = month_counts.cumsum().to_numpy()
    month_count = len(month_counts)
    total_rows = len(sorted_features)

    train_month_count = min(
        range(1, month_count - 1),
        key=lambda count: abs(cumulative_rows[count - 1] - train_fraction * total_rows),
    )
    validation_end_month_count = min(
        range(train_month_count + 1, month_count),
        key=lambda count: abs(
            cumulative_rows[count - 1]
            - (train_fraction + validation_fraction) * total_rows
        ),
    )
    train_end = pd.Timestamp(month_counts.index[train_month_count - 1])
    validation_end = pd.Timestamp(month_counts.index[validation_end_month_count - 1])

    train_mask = sorted_dates <= train_end
    validation_mask = (sorted_dates > train_end) & (sorted_dates <= validation_end)
    test_mask = sorted_dates > validation_end

    def select(mask: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        return sorted_features.loc[mask].reset_index(drop=True), sorted_target.loc[mask].reset_index(drop=True)

    X_train, y_train = select(train_mask)
    X_validation, y_validation = select(validation_mask)
    X_test, y_test = select(test_mask)
    if min(len(X_train), len(X_validation), len(X_test)) == 0:
        raise ValueError("Chronological split produced an empty partition")

    return TemporalSplit(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        train_start=pd.Timestamp(sorted_dates.loc[train_mask].min()),
        train_end=train_end,
        validation_start=pd.Timestamp(sorted_dates.loc[validation_mask].min()),
        validation_end=validation_end,
        test_start=pd.Timestamp(sorted_dates.loc[test_mask].min()),
        test_end=pd.Timestamp(sorted_dates.loc[test_mask].max()),
    )


def calculate_scale_pos_weight(target: pd.Series) -> float:
    """Return negative/positive class counts for the supplied training labels."""
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("Class-weight target values must be non-null and binary")
    positives = int((target == 1).sum())
    negatives = int((target == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Both positive and negative training examples are required")
    return negatives / positives


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Create deterministic numeric passthrough and categorical one-hot steps."""
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", "passthrough", numeric_features))
    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "missing_category",
                    SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                ),
                (
                    "one_hot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        dtype=np.float32,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    if not transformers:
        raise ValueError("At least one model feature is required")
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


def baseline_model_parameters(
    scale_pos_weight: float,
    *,
    random_seed: int = RANDOM_SEED,
    n_estimators: int = 400,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> dict[str, Any]:
    """Build the fixed, reproducible imbalanced-classification baseline."""
    if scale_pos_weight <= 0:
        raise ValueError("scale_pos_weight must be positive")
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_estimators": n_estimators,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_seed,
        "n_jobs": 1,
        "tree_method": "hist",
        "device": "cpu",
        "early_stopping_rounds": early_stopping_rounds,
        "verbosity": 0,
    }


def fit_xgboost_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_validation: np.ndarray,
    y_validation: pd.Series,
    *,
    model_parameters: dict[str, Any],
) -> XGBClassifier:
    """Fit XGBoost and select its iteration using only validation data."""
    if y_train.nunique() != 2:
        raise ValueError("Training labels must contain both binary classes")
    if y_validation.nunique() != 2:
        raise ValueError("Validation labels must contain both binary classes")
    model = XGBClassifier(**model_parameters)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )
    return model


def evaluate_probabilities(
    target: pd.Series,
    probabilities: np.ndarray | pd.Series,
    *,
    threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
) -> dict[str, Any]:
    """Calculate imbalance-aware metrics and confusion counts at a fixed threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("Classification threshold must be between 0 and 1")
    y_true = np.asarray(target, dtype=np.int8)
    y_probability = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(y_probability) or len(y_true) == 0:
        raise ValueError("Target and probabilities must have equal, non-zero lengths")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Evaluation target values must be binary")
    if not np.isfinite(y_probability).all() or ((y_probability < 0) | (y_probability > 1)).any():
        raise ValueError("Predicted probabilities must be finite and in [0, 1]")
    if np.unique(y_true).size != 2:
        raise ValueError("Evaluation requires positive and negative examples")

    y_predicted = (y_probability >= threshold).astype(np.int8)
    negatives = int((y_true == 0).sum())
    positives = int((y_true == 1).sum())
    matrix = confusion_matrix(y_true, y_predicted, labels=[0, 1])
    return {
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "pr_auc": float(average_precision_score(y_true, y_probability)),
        "precision": float(precision_score(y_true, y_predicted, zero_division=0)),
        "recall": float(recall_score(y_true, y_predicted, zero_division=0)),
        "f1": float(f1_score(y_true, y_predicted, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "positive_support": positives,
        "negative_support": negatives,
        "positive_prevalence": positives / len(y_true),
        "threshold": float(threshold),
        "pr_auc_definition": "average precision",
    }


def _split_summary(
    name: str,
    target: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    positive = int((target == 1).sum())
    negative = int((target == 0).sum())
    return {
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "rows": len(target),
        "positive": positive,
        "negative": negative,
        "positive_prevalence": positive / len(target),
    }


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def build_model_metadata(
    *,
    dataset_path: str,
    users_count: int,
    feature_list: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    split: TemporalSplit,
    model_parameters: dict[str, Any],
    scale_pos_weight: float,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    best_iteration: int | None,
    best_validation_metric: float | None,
    preprocessing_feature_names: list[str],
    random_seed: int,
) -> dict[str, Any]:
    """Create privacy-safe metadata for the trained model and its evaluation."""
    split_info = {
        "train": _split_summary(
            "train", split.y_train, split.train_start, split.train_end
        ),
        "validation": _split_summary(
            "validation",
            split.y_validation,
            split.validation_start,
            split.validation_end,
        ),
        "test": _split_summary("test", split.y_test, split.test_start, split.test_end),
    }
    total_positive = sum(item["positive"] for item in split_info.values())
    total_rows = sum(item["rows"] for item in split_info.values())
    return {
        "model_name": "GARUDA-X XGBoost Next-Month Churn Risk",
        "model_version": "1.0.0-stage3-baseline",
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "total_records": total_rows,
        "customer_count": users_count,
        "target_column": TARGET_COLUMN,
        "target_interpretation": "1 means churn in the next month; 0 means no next-month churn",
        "feature_list": feature_list,
        "feature_count": len(feature_list),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "preprocessing": {
            "numeric": "passthrough; no numeric imputation or scaling",
            "categorical": "constant missing-category token then OneHotEncoder(handle_unknown='ignore')",
            "fitted_on": "training partition only",
            "numeric_missing_values": "left as NaN for XGBoost native handling",
            "categorical_missing_values": "encoded as an explicit __MISSING__ category",
            "transformed_feature_order": preprocessing_feature_names,
        },
        "random_seed": random_seed,
        "model_parameters": model_parameters,
        "scale_pos_weight": scale_pos_weight,
        "split_method": "chronological complete-month partitions; no random row split",
        "splits": split_info,
        "overall_positive_count": total_positive,
        "overall_negative_count": total_rows - total_positive,
        "overall_positive_prevalence": total_positive / total_rows,
        "validation_metrics": validation_metrics,
        "final_test_metrics": test_metrics,
        "evaluation_probability_threshold": DEFAULT_PROBABILITY_THRESHOLD,
        "selected_business_probability_threshold": None,
        "threshold_note": "0.50 is used only for initial classification metrics; it is not selected as an optimal business threshold.",
        "risk_thresholds": {
            "medium_threshold": MEDIUM_RISK_THRESHOLD,
            "high_threshold": HIGH_RISK_THRESHOLD,
            "note": "Configurable initial business bands, not statistically optimized.",
        },
        "early_stopping": {
            "enabled": model_parameters.get("early_stopping_rounds") is not None,
            "rounds": model_parameters.get("early_stopping_rounds"),
            "best_iteration": best_iteration,
            "best_validation_metric": best_validation_metric,
            "validation_metric_name": "aucpr",
        },
        "calibration": {
            "calibrated": False,
            "brier_score": None,
            "note": "No probability calibration was performed; Brier score was not calculated.",
        },
        "software_versions": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "xgboost": xgboost.__version__,
            "scikit_learn": _package_version("scikit-learn"),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": _package_version("joblib"),
        },
        "artifacts": {
            "model": MODEL_FILENAME,
            "preprocessor": PREPROCESSOR_FILENAME,
            "metadata": METADATA_FILENAME,
        },
    }


def _print_split(name: str, details: dict[str, Any]) -> None:
    print(
        f"{name}: {details['start_date']} to {details['end_date']} | "
        f"rows={details['rows']} | positive={details['positive']} | "
        f"negative={details['negative']} | prevalence={details['positive_prevalence']:.3%}"
    )


def _print_metrics(name: str, metrics: dict[str, Any]) -> None:
    print(
        f"{name}: ROC-AUC={metrics['roc_auc']:.4f} | "
        f"PR-AUC/AP={metrics['pr_auc']:.4f} | "
        f"precision={metrics['precision']:.4f} | recall={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f} | support={metrics['positive_support']} | "
        f"confusion_matrix={metrics['confusion_matrix']}"
    )


def _artifact_path_for_metadata(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def save_model_artifacts(
    model: XGBClassifier,
    preprocessor: ColumnTransformer,
    metadata: dict[str, Any],
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> dict[str, Path]:
    """Save model, fitted preprocessing, and metadata to the artifact directory."""
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    preprocessor_path = output_dir / PREPROCESSOR_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    model.save_model(str(model_path))
    joblib.dump(preprocessor, preprocessor_path)
    metadata.setdefault("artifacts", {}).update(
        {
            "model": MODEL_FILENAME,
            "preprocessor": PREPROCESSOR_FILENAME,
            "metadata": METADATA_FILENAME,
            "model_path": _artifact_path_for_metadata(model_path),
            "preprocessor_path": _artifact_path_for_metadata(preprocessor_path),
            "metadata_path": _artifact_path_for_metadata(metadata_path),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "model": model_path,
        "preprocessor": preprocessor_path,
        "metadata": metadata_path,
    }


def train_churn_model(
    *,
    dataset_path: str | Path | None = None,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    random_seed: int = RANDOM_SEED,
    n_estimators: int = 400,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> dict[str, Any]:
    """Run the leakage-safe chronological XGBoost training and evaluation workflow."""
    source_path = Path(dataset_path) if dataset_path is not None else get_dataset_path(
        "synthetic_saas_churn", "user_monthly"
    )
    expected_path = get_dataset_path("synthetic_saas_churn", "user_monthly").resolve()
    if source_path.resolve() != expected_path:
        raise ValueError(
            "Stage 3 training is restricted to the registered synthetic monthly dataset: "
            f"{expected_path}"
        )

    raw = load_data(source_path)
    validate_synthetic_monthly(raw).raise_if_invalid()
    cleaned = clean_synthetic_monthly(raw)
    validate_synthetic_monthly(cleaned).raise_if_invalid()
    features, target = prepare_churn_features(cleaned)
    forbidden = {TARGET_COLUMN, "churned", "churn_date", "future_month", "future_months"}
    if forbidden.intersection(features.columns):
        raise ValueError("Feature engineering included forbidden future-outcome fields")
    feature_list = list(features.columns)
    if not feature_list or len(feature_list) != len(set(feature_list)):
        raise ValueError("Feature list must be non-empty and have unique columns")

    numeric_features = [
        column
        for column in (*NUMERIC_FEATURES, "seat_utilization")
        if column in feature_list
    ]
    categorical_features = [
        column for column in CATEGORICAL_FEATURES if column in feature_list
    ]
    if set(numeric_features + categorical_features) != set(feature_list):
        raise ValueError("Feature list contains columns without a preprocessing rule")

    split = chronological_split(features, target, cleaned["month"])
    split_values = {
        "train": _split_summary("train", split.y_train, split.train_start, split.train_end),
        "validation": _split_summary(
            "validation",
            split.y_validation,
            split.validation_start,
            split.validation_end,
        ),
        "test": _split_summary("test", split.y_test, split.test_start, split.test_end),
    }
    if any(item["positive"] == 0 or item["negative"] == 0 for item in split_values.values()):
        raise ValueError("Every temporal partition must contain both target classes")

    print(f"Dataset: {source_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Records: {len(cleaned)} | customers: {cleaned['user_id'].nunique()}")
    print(f"Features ({len(feature_list)}): {feature_list}")
    print(f"Target: {TARGET_COLUMN}")
    overall_positive = int((target == 1).sum())
    print(
        f"Overall labels: positive={overall_positive} | "
        f"negative={len(target) - overall_positive} | "
        f"prevalence={overall_positive / len(target):.3%}"
    )
    for name, details in split_values.items():
        _print_split(name, details)

    scale_pos_weight = calculate_scale_pos_weight(split.y_train)
    print(
        f"Training classes: positive={split_values['train']['positive']} | "
        f"negative={split_values['train']['negative']} | "
        f"scale_pos_weight={scale_pos_weight:.6f}"
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train = preprocessor.fit_transform(split.X_train)
    X_validation = preprocessor.transform(split.X_validation)
    parameters = baseline_model_parameters(
        scale_pos_weight,
        random_seed=random_seed,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )
    print(f"Model parameters: {json.dumps(parameters, sort_keys=True)}")
    model = fit_xgboost_model(
        X_train,
        split.y_train,
        X_validation,
        split.y_validation,
        model_parameters=parameters,
    )
    best_iteration = int(model.best_iteration) if hasattr(model, "best_iteration") else None
    best_validation_metric = float(model.best_score) if hasattr(model, "best_score") else None
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    validation_metrics = evaluate_probabilities(
        split.y_validation, validation_probabilities
    )
    _print_metrics("Validation at threshold 0.50", validation_metrics)
    if best_iteration is not None:
        print(
            f"Early stopping: best_iteration={best_iteration} | "
            f"best validation aucpr={best_validation_metric:.6f}"
        )

    X_test = preprocessor.transform(split.X_test)
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_probabilities(split.y_test, test_probabilities)
    _print_metrics("Final test at threshold 0.50", test_metrics)

    transformed_names = [
        str(name) for name in preprocessor.get_feature_names_out().tolist()
    ]
    dataset_relative_path = source_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    metadata = build_model_metadata(
        dataset_path=dataset_relative_path,
        users_count=int(cleaned["user_id"].nunique()),
        feature_list=feature_list,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        split=split,
        model_parameters=parameters,
        scale_pos_weight=scale_pos_weight,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        best_iteration=best_iteration,
        best_validation_metric=best_validation_metric,
        preprocessing_feature_names=transformed_names,
        random_seed=random_seed,
    )

    artifacts = save_model_artifacts(model, preprocessor, metadata, artifact_dir)
    model_path = artifacts["model"]
    preprocessor_path = artifacts["preprocessor"]
    metadata_path = artifacts["metadata"]

    from src.prediction.predictor import load_churn_predictor

    restored = load_churn_predictor(artifact_dir)
    restored_probabilities = restored.predict_proba(split.X_validation.head(8))
    if not restored_probabilities.between(0, 1).all():
        raise RuntimeError("Reloaded model returned probabilities outside [0, 1]")

    print(f"Model: {model_path}")
    print(f"Preprocessor: {preprocessor_path}")
    print(f"Metadata: {metadata_path}")
    print("Calibration: not implemented; probabilities are not calibrated.")
    return metadata


def main() -> None:
    train_churn_model()


if __name__ == "__main__":
    main()
