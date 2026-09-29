from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.prediction.predictor import (
    DEFAULT_ARTIFACT_DIR,
    ChurnPredictor,
    load_churn_predictor,
)
from src.explainability.evidence_engine import summarize_evidence

SHAP_OUTPUT_SPACE = "raw_margin_log_odds"
RECONSTRUCTION_TOLERANCE = 1e-5
APPROVED_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def _python_value(value: object) -> object:
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    return value


def _validated_shap_value(feature: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, np.number)):
        raise TypeError(f"SHAP value for {feature} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"SHAP value for {feature} must be finite")
    return result


def build_feature_explanations(
    feature_values: Mapping[str, object],
    shap_values: Mapping[str, object],
) -> list[dict[str, object]]:
    if set(feature_values) != set(shap_values):
        raise ValueError("Feature values and SHAP values must have identical feature names")
    rows: list[dict[str, object]] = []
    for feature in feature_values:
        shap_value = _validated_shap_value(feature, shap_values[feature])
        direction = (
            "increases_risk"
            if shap_value > 0
            else "decreases_risk"
            if shap_value < 0
            else "neutral"
        )
        rows.append(
            {
                "feature": str(feature),
                "feature_value": _python_value(feature_values[feature]),
                "shap_value": shap_value,
                "direction": direction,
                "absolute_importance": abs(shap_value),
            }
        )
    return sorted(rows, key=lambda item: (-float(item["absolute_importance"]), str(item["feature"])))


def _validated_top_n(top_n: int) -> int:
    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)):
        raise TypeError("top_n must be an integer")
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    return int(top_n)


def select_top_drivers(
    explanations: list[dict[str, object]],
    *,
    direction: str,
    top_n: int = 3,
) -> list[dict[str, object]]:
    limit = _validated_top_n(top_n)
    if direction not in {"increases_risk", "decreases_risk"}:
        raise ValueError("direction must be increases_risk or decreases_risk")
    selected = [
        dict(item)
        for item in explanations
        if item.get("direction") == direction
    ]
    selected.sort(
        key=lambda item: (
            -float(item["absolute_importance"]),
            str(item["feature"]),
        )
    )
    return selected[:limit]


class ShapEngine:
    def __init__(
        self,
        predictor: ChurnPredictor,
        *,
        reconstruction_tolerance: float = RECONSTRUCTION_TOLERANCE,
    ) -> None:
        if not isinstance(predictor, ChurnPredictor):
            raise TypeError("predictor must be a loaded Stage 3 ChurnPredictor")
        if (
            isinstance(reconstruction_tolerance, (bool, np.bool_))
            or not isinstance(reconstruction_tolerance, (Real, np.number))
            or not np.isfinite(float(reconstruction_tolerance))
            or reconstruction_tolerance <= 0
        ):
            raise ValueError("reconstruction_tolerance must be positive")
        self.predictor = predictor
        self.feature_order = list(predictor.metadata.get("feature_list", []))
        if self.feature_order != APPROVED_FEATURES:
            raise ValueError("Saved predictor feature order does not match Stage 3 allowlist")
        self.transformed_feature_order = [
            str(name) for name in predictor.preprocessor.get_feature_names_out()
        ]
        if len(self.transformed_feature_order) != predictor.model.n_features_in_:
            raise ValueError("Preprocessor output does not match saved model feature count")
        self.feature_groups = self._build_feature_groups()
        self.reconstruction_tolerance = float(reconstruction_tolerance)

    def _build_feature_groups(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {feature: [] for feature in self.feature_order}
        for index, transformed_name in enumerate(self.transformed_feature_order):
            matches = [
                feature
                for feature in self.feature_order
                if transformed_name == feature
                or transformed_name.startswith(f"{feature}_")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Transformed feature {transformed_name!r} cannot be mapped uniquely"
                )
            groups[matches[0]].append(index)
        empty_groups = [feature for feature, indexes in groups.items() if not indexes]
        if empty_groups:
            raise ValueError(f"Preprocessor has no transformed columns for {empty_groups}")
        return groups

    def explain_records(
        self,
        records: pd.DataFrame,
        *,
        top_n: int = 3,
    ) -> list[dict[str, object]]:
        limit = _validated_top_n(top_n)
        if not isinstance(records, pd.DataFrame):
            raise TypeError("records must be a pandas DataFrame")
        if records.empty:
            raise ValueError("records must contain at least one row")
        missing = [feature for feature in self.feature_order if feature not in records.columns]
        if missing:
            raise ValueError(f"Missing required model features: {missing}")

        model_features = records.loc[:, self.feature_order].copy(deep=True)
        prepared = self.predictor._prepare_input(model_features)
        probabilities = self.predictor.predict_proba(model_features)
        transformed = np.asarray(self.predictor.preprocessor.transform(prepared))
        if transformed.ndim != 2 or transformed.shape[1] != len(
            self.transformed_feature_order
        ):
            raise ValueError("Prepared feature matrix has an unexpected shape")
        if not np.isfinite(probabilities.to_numpy(dtype=float)).all():
            raise RuntimeError("Saved predictor returned a non-finite probability")

        booster = self.predictor.model.get_booster()
        best_iteration = getattr(self.predictor.model, "best_iteration", None)
        iteration_range = (
            (0, int(best_iteration) + 1) if best_iteration is not None else (0, 0)
        )
        matrix = xgb.DMatrix(transformed)
        contributions = np.asarray(
            booster.predict(
                matrix,
                pred_contribs=True,
                approx_contribs=False,
                iteration_range=iteration_range,
            ),
            dtype=float,
        )
        raw_margins = np.asarray(
            booster.predict(
                matrix,
                output_margin=True,
                iteration_range=iteration_range,
            ),
            dtype=float,
        ).reshape(-1)
        expected_shape = (len(records), len(self.transformed_feature_order) + 1)
        if contributions.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected TreeSHAP output shape {contributions.shape}; expected {expected_shape}"
            )
        if not np.isfinite(contributions).all() or not np.isfinite(raw_margins).all():
            raise RuntimeError("TreeSHAP returned non-finite values")

        outputs: list[dict[str, object]] = []
        for row_index in range(len(records)):
            expanded_values = contributions[row_index, :-1]
            grouped_values = {
                feature: float(expanded_values[indexes].sum())
                for feature, indexes in self.feature_groups.items()
            }
            feature_values = {
                feature: prepared.iloc[row_index][feature]
                for feature in self.feature_order
            }
            explanations = build_feature_explanations(feature_values, grouped_values)
            base_value = float(contributions[row_index, -1])
            reconstructed_margin = base_value + sum(grouped_values.values())
            raw_margin = float(raw_margins[row_index])
            reconstruction_error = reconstructed_margin - raw_margin
            if abs(reconstruction_error) > self.reconstruction_tolerance:
                raise RuntimeError(
                    "TreeSHAP contributions do not reconstruct the raw model margin "
                    f"within tolerance: error={reconstruction_error}"
                )

            probability = float(probabilities.iloc[row_index])
            probability_from_margin = (
                float(1.0 / (1.0 + np.exp(-raw_margin)))
                if raw_margin >= 0
                else float(np.exp(raw_margin) / (1.0 + np.exp(raw_margin)))
            )
            if not np.isclose(
                probability,
                probability_from_margin,
                rtol=1e-6,
                atol=self.reconstruction_tolerance,
            ):
                raise RuntimeError(
                    "Raw-margin sigmoid does not match the Stage 3 predictor probability"
                )
            risk_drivers = select_top_drivers(
                explanations, direction="increases_risk", top_n=limit
            )
            protective_factors = select_top_drivers(
                explanations, direction="decreases_risk", top_n=limit
            )
            record: dict[str, object] = {
                "churn_probability": probability,
                "probability_calibration": "uncalibrated",
                "shap_output_space": SHAP_OUTPUT_SPACE,
                "base_value": base_value,
                "raw_margin": raw_margin,
                "reconstructed_raw_margin": float(reconstructed_margin),
                "reconstruction_error": float(reconstruction_error),
                "reconstruction_tolerance": self.reconstruction_tolerance,
                "feature_explanations": explanations,
                "top_risk_drivers": risk_drivers,
                "protective_factors": protective_factors,
                "evidence": summarize_evidence(explanations),
                "evidence_scope": "structured_features_only",
            }
            if "user_id" in records.columns:
                record["user_id"] = _python_value(records.iloc[row_index]["user_id"])
            outputs.append(record)
        return outputs


def load_shap_engine(
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> ShapEngine:
    return ShapEngine(load_churn_predictor(artifact_dir))


def explain_prediction(
    records: pd.DataFrame,
    *,
    predictor: ChurnPredictor | None = None,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    top_n: int = 3,
) -> dict[str, object] | list[dict[str, object]]:
    engine = ShapEngine(predictor or load_churn_predictor(artifact_dir))
    results = engine.explain_records(records, top_n=top_n)
    return results[0] if len(results) == 1 else results
