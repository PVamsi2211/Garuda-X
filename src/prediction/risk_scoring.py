import math

from config.thresholds import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD


def classify_risk(
    probability: float,
    *,
    medium_threshold: float = MEDIUM_RISK_THRESHOLD,
    high_threshold: float = HIGH_RISK_THRESHOLD,
) -> str:
    """Map a churn probability to configurable initial business risk bands."""
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if not 0 <= medium_threshold < high_threshold <= 1:
        raise ValueError("Risk thresholds must satisfy 0 <= medium < high <= 1")
    if probability < medium_threshold:
        return "low"
    if probability < high_threshold:
        return "medium"
    return "high"
