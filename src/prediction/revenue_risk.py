from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from typing import Protocol

import numpy as np
import pandas as pd


class MonthlyPredictor(Protocol):
    def predict_monthly(self, monthly_frame: pd.DataFrame) -> pd.Series:
        ...


@dataclass(frozen=True)
class RevenueRiskAggregation:
    summary: pd.DataFrame
    top_customers: pd.DataFrame


def _numeric_series(
    values: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[pd.Series, bool]:
    scalar = pd.api.types.is_scalar(values)
    if scalar:
        raw_values = [values]
        index = pd.RangeIndex(1)
    elif isinstance(values, pd.Series):
        raw_values = values.tolist()
        index = values.index
    else:
        array = np.asarray(values, dtype=object)
        if array.ndim != 1:
            raise TypeError(f"{name} must be a scalar or one-dimensional numeric values")
        raw_values = array.tolist()
        index = pd.RangeIndex(len(raw_values))

    converted: list[float] = []
    invalid: list[object] = []
    missing: list[object] = []
    out_of_range: list[tuple[object, float]] = []
    for position, value in zip(index, raw_values):
        missing_value = pd.isna(value)
        if isinstance(missing_value, (bool, np.bool_)) and missing_value:
            missing.append(position)
            converted.append(np.nan)
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Real, Decimal, np.number)
        ) or np.iscomplexobj(value):
            invalid.append(position)
            converted.append(np.nan)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            invalid.append(position)
            converted.append(np.nan)
            continue
        if not np.isfinite(number):
            missing.append(position)
            converted.append(np.nan)
            continue
        if (minimum is not None and number < minimum) or (
            maximum is not None and number > maximum
        ):
            out_of_range.append((position, number))
        converted.append(number)

    if missing:
        raise ValueError(f"{name} contains missing or non-finite values at {missing}")
    if invalid:
        raise TypeError(f"{name} must contain numeric values; invalid positions: {invalid}")
    if out_of_range:
        if name == "churn_probability":
            raise ValueError(f"{name} must be within [0, 1]; invalid values: {out_of_range}")
        raise ValueError(f"{name} must be at least {minimum}; invalid values: {out_of_range}")
    return pd.Series(converted, index=index, name=name, dtype="float64"), scalar


def calculate_revenue_at_risk(
    mrr: object,
    churn_probability: object,
) -> float | pd.Series:
    mrr_values, mrr_scalar = _numeric_series(mrr, "mrr", minimum=0)
    probability_values, probability_scalar = _numeric_series(
        churn_probability, "churn_probability", minimum=0, maximum=1
    )

    if not mrr_scalar and not probability_scalar:
        if len(mrr_values) != len(probability_values):
            raise ValueError("mrr and churn_probability must have equal lengths")
        if isinstance(mrr, pd.Series) and isinstance(churn_probability, pd.Series):
            if not mrr.index.equals(churn_probability.index):
                raise ValueError("mrr and churn_probability Series must have matching indexes")
    size = len(probability_values) if mrr_scalar else len(mrr_values)
    if not mrr_scalar and probability_scalar:
        probability_values = pd.Series(
            np.repeat(probability_values.iloc[0], size), index=mrr_values.index
        )
    elif mrr_scalar and not probability_scalar:
        mrr_values = pd.Series(np.repeat(mrr_values.iloc[0], size), index=probability_values.index)
    elif mrr_scalar and probability_scalar:
        result = float(mrr_values.iloc[0] * probability_values.iloc[0])
        return result

    result = mrr_values.to_numpy() * probability_values.to_numpy()
    index = mrr_values.index
    return pd.Series(result, index=index, name="revenue_at_risk", dtype="float64")


def _validated_record_base(
    records: pd.DataFrame,
    *,
    require_month: bool = False,
) -> pd.DataFrame:
    if not isinstance(records, pd.DataFrame):
        raise TypeError("records must be a pandas DataFrame")
    required = {"user_id", "mrr"}
    if require_month:
        required.add("month")
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(f"records are missing required columns: {missing}")
    if records["user_id"].isna().any():
        raise ValueError("user_id contains missing values")

    has_month = "month" in records.columns
    if has_month and records["month"].isna().any():
        raise ValueError("month contains missing values")
    keys = ["month", "user_id"] if has_month else ["user_id"]
    duplicate_mask = records.duplicated(keys, keep=False)
    if duplicate_mask.any():
        raise ValueError(f"records contain duplicate customer keys: {keys}")

    mrr_values, _ = _numeric_series(records["mrr"], "mrr", minimum=0)
    columns = ["user_id"]
    if has_month:
        columns.append("month")
    columns.append("mrr")
    result = records.loc[:, columns].copy(deep=True)
    result["mrr"] = mrr_values
    return result


def calculate_account_revenue_risk(records: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "mrr", "churn_probability"}
    if not isinstance(records, pd.DataFrame):
        raise TypeError("records must be a pandas DataFrame")
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(f"records are missing required columns: {missing}")

    result = _validated_record_base(records)
    probability_values, _ = _numeric_series(
        records["churn_probability"], "churn_probability", minimum=0, maximum=1
    )
    result["churn_probability"] = probability_values
    result["revenue_at_risk"] = calculate_revenue_at_risk(
        result["mrr"], result["churn_probability"]
    )
    ordered = ["user_id"]
    if "month" in result.columns:
        ordered.append("month")
    ordered.extend(["mrr", "churn_probability", "revenue_at_risk"])
    return result.loc[:, ordered]


def predict_monthly_revenue_at_risk(
    monthly_records: pd.DataFrame,
    predictor: MonthlyPredictor,
) -> pd.DataFrame:
    monthly = _validated_record_base(monthly_records, require_month=True)
    predict_monthly = getattr(predictor, "predict_monthly", None)
    if not callable(predict_monthly):
        raise TypeError("predictor must provide a predict_monthly method")
    probabilities = predict_monthly(monthly_records)
    if isinstance(probabilities, pd.Series):
        if not probabilities.index.equals(monthly_records.index):
            raise ValueError("predictor probabilities must retain the monthly input index")
        probability_values = probabilities.to_numpy()
    else:
        probability_values = np.asarray(probabilities)
    if probability_values.ndim != 1 or len(probability_values) != len(monthly):
        raise ValueError("predictor must return one churn probability per monthly record")
    monthly["churn_probability"] = probability_values
    return calculate_account_revenue_risk(monthly)


def _validated_result(records: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "mrr", "churn_probability", "revenue_at_risk"}
    if not isinstance(records, pd.DataFrame):
        raise TypeError("records must be a pandas DataFrame")
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(f"records are missing required columns: {missing}")
    result = calculate_account_revenue_risk(records)
    supplied, _ = _numeric_series(records["revenue_at_risk"], "revenue_at_risk", minimum=0)
    if not np.allclose(
        supplied.to_numpy(), result["revenue_at_risk"].to_numpy(), rtol=1e-9, atol=1e-9
    ):
        raise ValueError("revenue_at_risk values must equal churn_probability multiplied by mrr")
    return result


def rank_customers_by_revenue_at_risk(records: pd.DataFrame) -> pd.DataFrame:
    result = _validated_result(records)
    return result.sort_values(
        "revenue_at_risk", ascending=False, kind="mergesort"
    ).copy()


def rank_customers_by_churn_probability(records: pd.DataFrame) -> pd.DataFrame:
    result = _validated_result(records)
    return result.sort_values(
        "churn_probability", ascending=False, kind="mergesort"
    ).copy()


def _validate_top_n(top_n: int) -> int:
    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)):
        raise TypeError("top_n must be an integer")
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    return int(top_n)


def aggregate_revenue_at_risk(
    records: pd.DataFrame,
    *,
    top_n: int = 10,
) -> RevenueRiskAggregation:
    limit = _validate_top_n(top_n)
    result = _validated_result(records)
    if result.empty:
        raise ValueError("records must contain at least one customer")

    aggregations = {
        "total_revenue_at_risk": ("revenue_at_risk", "sum"),
        "average_revenue_at_risk": ("revenue_at_risk", "mean"),
        "median_revenue_at_risk": ("revenue_at_risk", "median"),
        "customer_count": ("user_id", "size"),
    }
    if "month" not in result.columns:
        summary = pd.DataFrame(
            [
                {
                    "total_revenue_at_risk": float(result["revenue_at_risk"].sum()),
                    "average_revenue_at_risk": float(result["revenue_at_risk"].mean()),
                    "median_revenue_at_risk": float(result["revenue_at_risk"].median()),
                    "customer_count": len(result),
                }
            ]
        )
        top_customers = rank_customers_by_revenue_at_risk(result).head(limit).copy()
    else:
        summary = (
            result.groupby("month", sort=True, as_index=False)
            .agg(**aggregations)
            .reset_index(drop=True)
        )
        top_groups = [
            rank_customers_by_revenue_at_risk(group).head(limit)
            for _, group in result.groupby("month", sort=True)
        ]
        top_customers = (
            pd.concat(top_groups, axis=0).copy()
            if top_groups
            else result.head(0).copy()
        )
    return RevenueRiskAggregation(summary=summary, top_customers=top_customers)
