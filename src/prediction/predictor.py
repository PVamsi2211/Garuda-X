import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.data.cleaner import convert_numeric_columns, normalize_categorical_columns
from src.data.feature_engineering import prepare_prediction_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "models" / "xgboost_model"
MODEL_FILENAME = "xgboost_churn_model.json"
PREPROCESSOR_FILENAME = "preprocessing_pipeline.joblib"
METADATA_FILENAME = "model_metadata.json"


@dataclass
class ChurnPredictor:
    model: XGBClassifier
    preprocessor: Any
    metadata: dict[str, Any]

    def _prepare_input(self, features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        expected = self.metadata.get("feature_list")
        if not isinstance(expected, list) or not expected:
            raise ValueError("Model metadata has no valid feature_list")
        received = list(features.columns)
        if received != expected:
            missing = [column for column in expected if column not in received]
            extra = [column for column in received if column not in expected]
            if not missing and not extra:
                raise ValueError(
                    "Feature order mismatch: expected "
                    f"{expected}, received {received}"
                )
            raise ValueError(
                f"Feature columns mismatch: missing={missing}, extra={extra}, "
                f"expected_order={expected}, received={received}"
            )

        numeric = self.metadata.get("numeric_features", [])
        categorical = self.metadata.get("categorical_features", [])
        prepared = convert_numeric_columns(features, numeric) if numeric else features.copy()
        if categorical:
            prepared = normalize_categorical_columns(
                prepared, categorical, lowercase=True
            )
        return prepared

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """Return P(churned_next_month=1) in the training feature order."""
        prepared = self._prepare_input(features)
        transformed = self.preprocessor.transform(prepared)
        probabilities = np.asarray(self.model.predict_proba(transformed))[:, 1]
        if not np.isfinite(probabilities).all() or (
            (probabilities < 0) | (probabilities > 1)
        ).any():
            raise RuntimeError("The model returned invalid churn probabilities")
        return pd.Series(
            probabilities,
            index=features.index,
            name="churn_probability_next_month",
        )

    def predict_monthly(self, monthly_frame: pd.DataFrame) -> pd.Series:
        """Prepare unlabeled month-t rows with Stage 2 and predict their next-month risk."""
        features = prepare_prediction_features(monthly_frame)
        return self.predict_proba(features)


def load_churn_predictor(
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> ChurnPredictor:
    """Load the saved XGBoost model, preprocessing pipeline, and metadata."""
    directory = Path(artifact_dir)
    model_path = directory / MODEL_FILENAME
    preprocessor_path = directory / PREPROCESSOR_FILENAME
    metadata_path = directory / METADATA_FILENAME
    missing = [
        path.name for path in (model_path, preprocessor_path, metadata_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing churn model artifact(s) in {directory}: {', '.join(missing)}"
        )

    model = XGBClassifier()
    model.load_model(str(model_path))
    preprocessor = joblib.load(preprocessor_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_transformed = metadata.get("preprocessing", {}).get(
        "transformed_feature_order", []
    )
    if list(preprocessor.get_feature_names_out()) != expected_transformed:
        raise ValueError("Saved preprocessing feature order does not match metadata")
    if model.n_features_in_ != len(expected_transformed):
        raise ValueError("Saved model feature count does not match preprocessing metadata")
    return ChurnPredictor(model, preprocessor, metadata)


def predict_churn_probability(
    features: pd.DataFrame,
    *,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> pd.Series:
    """Load the saved predictor and return next-month churn probabilities."""
    return load_churn_predictor(artifact_dir).predict_proba(features)
