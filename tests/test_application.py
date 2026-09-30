import pandas as pd
import pytest

from src.data.validator import validate_prediction_monthly


def prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u-1"],
            "month": pd.to_datetime(["2025-10-01"]),
            "mrr": [120.0],
            "sessions": [13],
            "feature_usage_score": [72.0],
            "support_tickets": [1],
            "payment_failures": [0],
            "nps_score": [8],
            "product_incident": [0],
            "active_seats": [4],
            "tenure_month": [11],
            "plan_type": ["pro"],
        }
    )


def test_prediction_schema_accepts_unlabeled_model_input() -> None:
    report = validate_prediction_monthly(prediction_rows())
    assert report.is_valid
    assert "churned_next_month" not in report.missing_required_columns


def test_prediction_schema_rejects_duplicate_user_month_keys() -> None:
    rows = pd.concat([prediction_rows(), prediction_rows()], ignore_index=True)
    report = validate_prediction_monthly(rows)
    assert not report.is_valid
    assert report.duplicate_identifiers == {"user_id + month": 1}


def test_prediction_schema_allows_missing_model_signals_but_requires_mrr() -> None:
    rows = prediction_rows()
    rows.loc[0, "sessions"] = None
    assert validate_prediction_monthly(rows).is_valid
    rows.loc[0, "mrr"] = None
    assert not validate_prediction_monthly(rows).is_valid


def test_streamlit_entrypoint_import_is_safe() -> None:
    import app

    assert callable(app.main)
