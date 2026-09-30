from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np

from src.data.validator import ValidationReport, validate_prediction_monthly
from src.explainability.shap_engine import ShapEngine
from src.prediction.predictor import ChurnPredictor
from src.prediction.revenue_risk import aggregate_revenue_at_risk, calculate_account_revenue_risk
from src.prediction.risk_scoring import classify_risk


@dataclass(frozen=True)
class MonthlyAnalysis:
    records: pd.DataFrame
    explanations: list[dict[str, object]]
    validation: ValidationReport
    summary: pd.DataFrame


def analyze_monthly_records(
    records: pd.DataFrame,
    *,
    predictor: ChurnPredictor,
    shap_engine: ShapEngine,
) -> MonthlyAnalysis:
    if not isinstance(records, pd.DataFrame) or records.empty:
        raise ValueError("The selected month must contain at least one account row")
    validation = validate_prediction_monthly(records)
    validation.raise_if_invalid()
    probabilities = predictor.predict_monthly(records).to_numpy(dtype=float)
    explanations = shap_engine.explain_records(records, top_n=3)
    explanation_probabilities = np.asarray(
        [float(item["churn_probability"]) for item in explanations], dtype=float
    )
    if not np.allclose(probabilities, explanation_probabilities, rtol=1e-6, atol=1e-8):
        raise RuntimeError("Stage 3 prediction and TreeSHAP outputs disagree")
    risk_input = records.loc[:, ["user_id", "month", "mrr"]].copy()
    risk_input["churn_probability"] = probabilities
    scored = calculate_account_revenue_risk(risk_input)
    scored["risk_band"] = [classify_risk(value) for value in probabilities]
    scored["key_driver"] = [
        item["top_risk_drivers"][0]["feature"]
        if item["top_risk_drivers"]
        else "No positive risk driver"
        for item in explanations
    ]
    summary = aggregate_revenue_at_risk(scored, top_n=20).summary
    return MonthlyAnalysis(
        records=scored.reset_index(drop=True),
        explanations=explanations,
        validation=validation,
        summary=summary,
    )


def explanation_for_user(
    analysis: MonthlyAnalysis,
    user_id: object,
) -> dict[str, Any]:
    matches = [
        index
        for index, value in enumerate(analysis.records["user_id"].tolist())
        if str(value) == str(user_id)
    ]
    if not matches:
        raise KeyError(f"No analyzed record exists for user_id {user_id!r}")
    return analysis.explanations[matches[0]]
