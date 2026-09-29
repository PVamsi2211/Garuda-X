import pandas as pd
import pytest

from config.datasets import DATASET_REGISTRY, get_dataset_path
from src.data.cleaner import (
    clean_synthetic_monthly,
    convert_numeric_columns,
    drop_duplicate_rows,
    handle_missing_values,
    normalize_categorical_columns,
    parse_date_columns,
)
from src.data.feature_engineering import prepare_churn_features
from src.data.loader import load_csv, load_data, load_jsonl, load_parquet
from src.data.validator import (
    validate_account_table,
    validate_crm_dataset,
    validate_required_columns,
    validate_synthetic_monthly,
)


def synthetic_monthly() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u-1", "u-1"],
            "month": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "mrr": [100.0, 90.0],
            "sessions": [12, 8],
            "feature_usage_score": [80.0, 60.0],
            "support_tickets": [1, 2],
            "payment_failures": [0, 1],
            "plan_type": [" Pro ", "BASIC"],
            "nps_score": [8, 5],
            "product_incident": [0, 1],
            "active_seats": [5, 4],
            "seats_purchased": [10, 8],
            "tenure_month": [2, 3],
            "churned": [0, 0],
            "churn_date": [None, None],
            "future_month": ["2025-02-01", "2025-03-01"],
            "churned_next_month": [0, 1],
        }
    )


def test_csv_loading(tmp_path) -> None:
    path = tmp_path / "accounts.csv"
    pd.DataFrame({"account_id": ["a-1"]}).to_csv(path, index=False)
    assert load_csv(path).loc[0, "account_id"] == "a-1"
    assert load_data(path).equals(load_csv(path))


def test_jsonl_loading_preserves_nested_values(tmp_path) -> None:
    path = tmp_path / "conversations.jsonl"
    path.write_text(
        '{"conversation_id":"c-1","churn_signals":["pricing"]}\n',
        encoding="utf-8",
    )
    frame = load_jsonl(path)
    assert frame.loc[0, "churn_signals"] == ["pricing"]
    assert load_data(path).equals(frame)


def test_parquet_loading(tmp_path) -> None:
    path = tmp_path / "monthly.parquet"
    original = synthetic_monthly()
    original.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(load_parquet(path), original)
    pd.testing.assert_frame_equal(load_data(path), original)


def test_loaders_report_missing_and_unsupported_paths(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_csv(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        load_data(tmp_path / "dataset.xlsx")


def test_required_column_validation() -> None:
    validate_required_columns(pd.DataFrame(columns=["account_id"]), ["account_id"])
    with pytest.raises(ValueError, match="account_id"):
        validate_required_columns(pd.DataFrame(), ["account_id"])


def test_synthetic_schema_reports_month_grain_duplicates_and_nulls() -> None:
    frame = synthetic_monthly()
    valid = validate_synthetic_monthly(frame)
    assert valid.is_valid
    assert valid.null_counts["mrr"] == 0

    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    report = validate_synthetic_monthly(duplicated)
    assert report.duplicate_identifiers["user_id + month"] == 1


def test_synthetic_schema_reports_missing_columns_and_target_values() -> None:
    frame = synthetic_monthly().drop(columns=["payment_failures"])
    frame.loc[0, "churned_next_month"] = 2
    report = validate_synthetic_monthly(frame)
    assert report.missing_required_columns == ["payment_failures"]
    assert report.invalid_target_values["churned_next_month"] == [2]


def test_account_and_crm_schema_validation() -> None:
    subscriptions = pd.DataFrame(
        {"subscription_id": ["s-1", "s-2"], "account_id": ["a-1", "a-1"]}
    )
    assert validate_account_table(
        subscriptions, "subscription_revenue_analysis"
    ).is_valid
    assert not validate_account_table(
        subscriptions.drop(columns="account_id"), "subscription_revenue_analysis"
    ).is_valid

    crm = pd.DataFrame(
        {
            "conversation_id": ["c-1"],
            "churn_risk_level": ["high"],
            "churn_signals": [["pricing_complaint"]],
            "conversation": [[{"role": "customer", "text": "Concerned"}]],
            "summary": ["Pricing concern"],
        }
    )
    assert validate_crm_dataset(crm).is_valid
    crm.loc[0, "churn_risk_level"] = "unknown"
    assert validate_crm_dataset(crm).invalid_target_values["churn_risk_level"] == [
        "unknown"
    ]


def test_numeric_conversion_and_invalid_values() -> None:
    frame = convert_numeric_columns(pd.DataFrame({"mrr": ["10.5", None]}), ["mrr"])
    assert frame["mrr"].iloc[0] == 10.5
    assert pd.isna(frame["mrr"].iloc[1])
    blank = convert_numeric_columns(pd.DataFrame({"mrr": ["  "]}), ["mrr"])
    assert pd.isna(blank["mrr"].iloc[0])
    with pytest.raises(ValueError, match="Invalid numeric values"):
        convert_numeric_columns(pd.DataFrame({"mrr": ["unknown"]}), ["mrr"])


def test_date_conversion_and_invalid_values() -> None:
    converted = parse_date_columns(
        pd.DataFrame({"month": ["2025-01-01", None]}), ["month"]
    )
    assert pd.api.types.is_datetime64_any_dtype(converted["month"])
    assert pd.isna(converted["month"].iloc[1])
    blank = parse_date_columns(pd.DataFrame({"month": [" "]}), ["month"])
    assert pd.isna(blank["month"].iloc[0])
    with pytest.raises(ValueError, match="Invalid date values"):
        parse_date_columns(pd.DataFrame({"month": ["not-a-date"]}), ["month"])


def test_categorical_duplicate_and_missing_value_cleaning() -> None:
    normalized = normalize_categorical_columns(
        pd.DataFrame({"plan_type": [" Pro ", ""]}), ["plan_type"], lowercase=True
    )
    assert normalized.loc[0, "plan_type"] == "pro"
    assert pd.isna(normalized.loc[1, "plan_type"])

    rows = pd.DataFrame({"id": [1, 1, 2], "value": ["x", "x", None]})
    assert len(drop_duplicate_rows(rows)) == 2
    nested = pd.DataFrame(
        {"conversation": [[{"text": "hello"}], [{"text": "hello"}]]}
    )
    assert len(drop_duplicate_rows(nested)) == 1
    assert len(handle_missing_values(rows, required_columns=["id"])) == 3
    assert len(handle_missing_values(rows, required_columns=["value"])) == 2


def test_clean_synthetic_monthly_is_deterministic_and_preserves_nulls() -> None:
    frame = synthetic_monthly()
    frame["sessions"] = frame["sessions"].astype("object")
    frame.loc[0, "sessions"] = "12"
    frame.loc[0, "feature_usage_score"] = None
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    cleaned = clean_synthetic_monthly(duplicated)
    assert len(cleaned) == 2
    assert cleaned.loc[0, "plan_type"] == "pro"
    assert pd.isna(cleaned.loc[0, "feature_usage_score"])


def test_feature_engineering_separates_target_and_omits_future_fields() -> None:
    frame = synthetic_monthly()
    features, target = prepare_churn_features(frame)

    assert target.tolist() == [0, 1]
    assert "churned_next_month" not in features
    assert "churned" not in features
    assert "churn_date" not in features
    assert "future_month" not in features
    assert "user_id" not in features
    assert "month" not in features
    assert features["plan_type"].tolist() == ["pro", "basic"]
    assert features["seat_utilization"].tolist() == [0.5, 0.5]


def test_feature_engineering_checks_target_and_optional_derived_sources() -> None:
    frame = synthetic_monthly().drop(columns="seats_purchased")
    features, _ = prepare_churn_features(frame)
    assert "seat_utilization" not in features

    frame.loc[0, "churned_next_month"] = 3
    with pytest.raises(ValueError, match="only 0 or 1"):
        prepare_churn_features(frame)


def test_dataset_registry_points_to_discovered_files() -> None:
    assert set(DATASET_REGISTRY) == {
        "synthetic_saas_churn",
        "saas_revenue_risk",
        "customer_churn_v2",
    }
    assert get_dataset_path("synthetic_saas_churn", "user_monthly").name == "user_monthly.parquet"
    assert get_dataset_path("saas_revenue_risk", "account_analysis").name == "account_analysis.csv"
    assert get_dataset_path("customer_churn_v2", "full").name == "full.jsonl"


def test_validate_required_columns_accepts_complete_schema() -> None:
    validate_required_columns(pd.DataFrame(columns=["account_id"]), ["account_id"])


def test_validate_required_columns_reports_missing_fields() -> None:
    with pytest.raises(ValueError, match="account_id"):
        validate_required_columns(pd.DataFrame(), ["account_id"])
