import pandas as pd

from src.data.cleaner import convert_numeric_columns, normalize_categorical_columns
from src.data.validator import SYNTHETIC_MONTHLY_REQUIRED

NUMERIC_FEATURES = (
    "mrr",
    "sessions",
    "feature_usage_score",
    "support_tickets",
    "payment_failures",
    "nps_score",
    "product_incident",
    "active_seats",
    "tenure_month",
)
CATEGORICAL_FEATURES = ("plan_type",)
TARGET_COLUMN = "churned_next_month"
LEAKAGE_COLUMNS = frozenset(
    {"churned", "churn_date", "future_month", "future_months", "churn_flag"}
)


def prepare_churn_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare current-month allowlisted features and the next-month churn target.

    Identifiers, dates, current/future churn outcomes, and all unrecognized fields
    are excluded from the feature matrix. ``seat_utilization`` is derived only when
    both active and purchased seat counts are supplied in this same input; this
    function does not join the per-user summary table to monthly history.
    """
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Cannot prepare churn labels; missing {TARGET_COLUMN}")
    features = prepare_prediction_features(frame)
    target = convert_numeric_columns(frame[[TARGET_COLUMN]], [TARGET_COLUMN])[
        TARGET_COLUMN
    ]
    if target.isna().any():
        raise ValueError("The churn target contains missing values")
    invalid_values = sorted(set(target.unique()) - {0, 1})
    if invalid_values:
        raise ValueError(f"The churn target must contain only 0 or 1: {invalid_values}")
    target = target.astype("int8").rename(TARGET_COLUMN)
    return features, target


def prepare_prediction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare the same allowlisted features for unlabeled inference rows."""
    required_for_prediction = set(SYNTHETIC_MONTHLY_REQUIRED) - {TARGET_COLUMN}
    missing = sorted(required_for_prediction - set(frame.columns))
    if missing:
        raise ValueError(
            "Cannot prepare prediction features; missing required columns: "
            + ", ".join(missing)
        )

    feature_columns = [
        column for column in (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
        if column in frame.columns
    ]
    features = frame.loc[:, feature_columns].copy()
    numeric_columns = [column for column in NUMERIC_FEATURES if column in features]
    if numeric_columns:
        features = convert_numeric_columns(features, numeric_columns)
    categorical_columns = [
        column for column in CATEGORICAL_FEATURES if column in features
    ]
    if categorical_columns:
        features = normalize_categorical_columns(
            features, categorical_columns, lowercase=True
        )

    if "active_seats" in frame.columns and "seats_purchased" in frame.columns:
        seat_counts = convert_numeric_columns(
            frame[["active_seats", "seats_purchased"]],
            ["active_seats", "seats_purchased"],
        )
        active_seats = seat_counts["active_seats"]
        seats_purchased = seat_counts["seats_purchased"]
        denominator = seats_purchased.where(seats_purchased > 0)
        features["seat_utilization"] = active_seats / denominator

    if any(column in features.columns for column in LEAKAGE_COLUMNS):
        features = features.drop(columns=list(LEAKAGE_COLUMNS & set(features.columns)))
    return features


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the allowlisted feature matrix for labeled or unlabeled observations."""
    return prepare_prediction_features(frame)
