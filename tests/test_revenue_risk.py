import numpy as np
import pandas as pd
import pytest

from src.prediction.revenue_risk import (
    aggregate_revenue_at_risk,
    calculate_account_revenue_risk,
    calculate_revenue_at_risk,
    predict_monthly_revenue_at_risk,
    rank_customers_by_churn_probability,
    rank_customers_by_revenue_at_risk,
)


def account_records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u-low", "u-mid", "u-high"],
            "mrr": [100.0, 50.0, 150.0],
            "churn_probability": [0.2, 0.8, 0.6],
        }
    )


def test_basic_revenue_at_risk_calculation() -> None:
    assert calculate_revenue_at_risk(80.31, 0.72) == pytest.approx(57.8232)


def test_zero_probability_produces_zero_risk() -> None:
    assert calculate_revenue_at_risk(125.0, 0.0) == 0.0


def test_probability_one_produces_full_mrr_exposure() -> None:
    assert calculate_revenue_at_risk(125.0, 1.0) == 125.0


def test_typical_probability_calculation() -> None:
    assert calculate_revenue_at_risk(80.0, 0.25) == 20.0


@pytest.mark.parametrize("probability", [-0.01, -1.0])
def test_probability_below_zero_is_rejected(probability: float) -> None:
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        calculate_revenue_at_risk(100.0, probability)


@pytest.mark.parametrize("probability", [1.01, 2.0])
def test_probability_above_one_is_rejected(probability: float) -> None:
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        calculate_revenue_at_risk(100.0, probability)


def test_missing_probability_is_reported() -> None:
    with pytest.raises(ValueError, match="churn_probability contains missing"):
        calculate_revenue_at_risk(100.0, None)


def test_missing_mrr_is_reported() -> None:
    with pytest.raises(ValueError, match="mrr contains missing"):
        calculate_revenue_at_risk(float("nan"), 0.5)


def test_invalid_or_negative_mrr_is_rejected_without_string_coercion() -> None:
    with pytest.raises(TypeError, match="mrr must contain numeric"):
        calculate_revenue_at_risk("85.25", 0.5)
    with pytest.raises(ValueError, match="mrr must be at least 0"):
        calculate_revenue_at_risk(-1.0, 0.5)


def test_customer_output_preserves_ids_and_supports_multiple_customers() -> None:
    result = calculate_account_revenue_risk(account_records())
    assert result.columns.tolist() == [
        "user_id",
        "mrr",
        "churn_probability",
        "revenue_at_risk",
    ]
    assert result["user_id"].tolist() == ["u-low", "u-mid", "u-high"]
    assert result["revenue_at_risk"].tolist() == pytest.approx([20.0, 40.0, 90.0])


def test_ranking_supports_revenue_risk_and_probability_independently() -> None:
    result = calculate_account_revenue_risk(account_records())
    ranked_risk = rank_customers_by_revenue_at_risk(result)
    ranked_probability = rank_customers_by_churn_probability(result)
    assert ranked_risk["user_id"].tolist() == ["u-high", "u-mid", "u-low"]
    assert ranked_probability["user_id"].tolist() == ["u-mid", "u-high", "u-low"]


def test_aggregation_reports_summary_and_top_n() -> None:
    result = calculate_account_revenue_risk(account_records())
    aggregation = aggregate_revenue_at_risk(result, top_n=2)
    summary = aggregation.summary.iloc[0]
    assert summary["total_revenue_at_risk"] == pytest.approx(150.0)
    assert summary["average_revenue_at_risk"] == pytest.approx(50.0)
    assert summary["median_revenue_at_risk"] == pytest.approx(40.0)
    assert summary["customer_count"] == 3
    assert aggregation.top_customers["user_id"].tolist() == ["u-high", "u-mid"]


def test_calculation_does_not_mutate_input_dataframe() -> None:
    source = account_records()
    before = source.copy(deep=True)
    calculate_account_revenue_risk(source)
    pd.testing.assert_frame_equal(source, before)


def test_monthly_records_keep_month_dimension_and_aggregate_per_month() -> None:
    monthly = pd.DataFrame(
        {
            "user_id": ["u-1", "u-2", "u-1", "u-3"],
            "month": pd.to_datetime(
                ["2025-01-01", "2025-01-01", "2025-02-01", "2025-02-01"]
            ),
            "mrr": [100.0, 80.0, 80.0, 50.0],
            "churn_probability": [0.5, 0.25, 0.4, 0.8],
        }
    )
    result = calculate_account_revenue_risk(monthly)
    aggregation = aggregate_revenue_at_risk(result, top_n=1)

    assert result.columns.tolist() == [
        "user_id",
        "month",
        "mrr",
        "churn_probability",
        "revenue_at_risk",
    ]
    assert aggregation.summary["month"].tolist() == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-02-01"),
    ]
    assert aggregation.summary["total_revenue_at_risk"].tolist() == pytest.approx(
        [70.0, 72.0]
    )
    assert aggregation.top_customers["user_id"].tolist() == ["u-1", "u-3"]
    assert aggregation.top_customers["month"].tolist() == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-02-01"),
    ]


def test_forbidden_future_and_churn_fields_do_not_affect_output() -> None:
    source = pd.DataFrame(
        {
            "user_id": ["u-1"],
            "month": [pd.Timestamp("2025-03-01")],
            "mrr": [100.0],
            "churn_probability": [0.5],
            "future_mrr": [900.0],
            "future_month": [pd.Timestamp("2025-04-01")],
            "churned": [1],
            "churn_date": [pd.Timestamp("2025-04-15")],
            "churned_next_month": [1],
        }
    )
    changed = source.copy(deep=True)
    changed["future_mrr"] = 1.0
    changed["churned"] = 0
    changed["churn_date"] = pd.NaT
    changed["churned_next_month"] = 0

    first = calculate_account_revenue_risk(source)
    second = calculate_account_revenue_risk(changed)
    pd.testing.assert_frame_equal(first, second)
    assert first["revenue_at_risk"].iloc[0] == 50.0
    assert not {
        "future_mrr",
        "future_month",
        "churned",
        "churn_date",
        "churned_next_month",
    }.intersection(first.columns)


def test_monthly_predictor_bridge_uses_current_month_mrr() -> None:
    class StubPredictor:
        def predict_monthly(self, monthly_frame: pd.DataFrame) -> pd.Series:
            return pd.Series([0.5, 0.25], index=monthly_frame.index)

    monthly = pd.DataFrame(
        {
            "user_id": ["u-1", "u-2"],
            "month": pd.to_datetime(["2025-03-01", "2025-03-01"]),
            "mrr": [100.0, 80.0],
            "future_mrr": [1000.0, 800.0],
            "churned": [1, 0],
        }
    )
    before = monthly.copy(deep=True)
    result = predict_monthly_revenue_at_risk(monthly, StubPredictor())

    assert result["revenue_at_risk"].tolist() == pytest.approx([50.0, 20.0])
    assert result["mrr"].tolist() == [100.0, 80.0]
    assert result["month"].tolist() == monthly["month"].tolist()
    pd.testing.assert_frame_equal(monthly, before)


def test_duplicate_customer_periods_are_rejected_to_avoid_double_counting() -> None:
    duplicate_accounts = pd.DataFrame(
        {
            "user_id": ["u-1", "u-1"],
            "mrr": [100.0, 100.0],
            "churn_probability": [0.5, 0.5],
        }
    )
    with pytest.raises(ValueError, match="duplicate customer keys"):
        calculate_account_revenue_risk(duplicate_accounts)

    duplicate_month = pd.DataFrame(
        {
            "user_id": ["u-1", "u-1"],
            "month": pd.to_datetime(["2025-03-01", "2025-03-01"]),
            "mrr": [100.0, 100.0],
            "churn_probability": [0.5, 0.5],
        }
    )
    with pytest.raises(ValueError, match="duplicate customer keys"):
        calculate_account_revenue_risk(duplicate_month)


@pytest.mark.parametrize("top_n", [-1, 1.5, True])
def test_top_n_must_be_a_non_negative_integer(top_n: object) -> None:
    result = calculate_account_revenue_risk(account_records())
    with pytest.raises((TypeError, ValueError)):
        aggregate_revenue_at_risk(result, top_n=top_n)
