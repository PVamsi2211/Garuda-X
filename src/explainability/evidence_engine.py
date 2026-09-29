from __future__ import annotations

from numbers import Real
from typing import Mapping

import numpy as np
import pandas as pd

FEATURE_LABELS = {
    "mrr": "MRR",
    "sessions": "Sessions",
    "feature_usage_score": "Feature usage score",
    "support_tickets": "Support tickets",
    "payment_failures": "Payment failures",
    "nps_score": "NPS score",
    "product_incident": "Product incident",
    "active_seats": "Active seats",
    "tenure_month": "Tenure month",
    "plan_type": "Plan type",
}


def _format_value(value: object) -> str:
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return "missing"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (Real, np.number)) and not isinstance(value, (bool, np.bool_)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("feature_value must be finite or missing")
        return format(number, ".12g")
    if isinstance(value, str):
        return value
    raise TypeError("feature_value must be a string, number, or missing")


def format_feature_evidence(record: Mapping[str, object]) -> dict[str, object]:
    required = {"feature", "feature_value", "shap_value", "direction"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"explanation record is missing fields: {missing}")

    feature = str(record["feature"])
    shap_value = record["shap_value"]
    if isinstance(shap_value, (bool, np.bool_)) or not isinstance(
        shap_value, (Real, np.number)
    ):
        raise TypeError("shap_value must be numeric")
    shap_value = float(shap_value)
    if not np.isfinite(shap_value):
        raise ValueError("shap_value must be finite")
    expected_direction = (
        "increases_risk"
        if shap_value > 0
        else "decreases_risk"
        if shap_value < 0
        else "neutral"
    )
    if record["direction"] != expected_direction:
        raise ValueError("direction does not match the sign of shap_value")

    label = FEATURE_LABELS.get(feature, feature.replace("_", " ").strip().capitalize())
    feature_value = record["feature_value"]
    value = _format_value(feature_value)
    if isinstance(feature_value, np.generic):
        feature_value = feature_value.item()
    if value == "missing":
        feature_value = None
    if expected_direction == "increases_risk":
        explanation = (
            f"{label} of {value} pushed the model's prediction toward higher churn risk."
        )
    elif expected_direction == "decreases_risk":
        explanation = (
            f"{label} of {value} pushed the model's prediction toward lower churn risk."
        )
    else:
        explanation = (
            f"{label} of {value} did not move the model's prediction toward higher "
            "or lower churn risk."
        )

    return {
        "feature": feature,
        "feature_value": feature_value,
        "shap_value": shap_value,
        "direction": expected_direction,
        "explanation": explanation,
        "source": "structured_model_feature",
    }


def summarize_evidence(records: list[Mapping[str, object]]) -> list[dict[str, object]]:
    if not isinstance(records, list):
        raise TypeError("records must be a list of structured feature explanations")
    return [format_feature_evidence(record) for record in records]
