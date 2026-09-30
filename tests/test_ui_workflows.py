from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ui.components.crm_upload import validate_crm_upload
from ui.components.tables import prioritize_records
from ui.runtime import normalize_crm_dates


def test_crm_upload_requires_account_scope_and_text() -> None:
    no_scope = validate_crm_upload(pd.DataFrame([{"text": "A renewal concern"}]))
    no_text = validate_crm_upload(pd.DataFrame([{"user_id": "acct-1"}]))

    assert "CRM evidence requires user_id or account_id for safe account-scoped retrieval." in no_scope
    assert "CRM evidence requires a text field." in no_text


def test_crm_upload_validates_each_row_and_accepts_either_identifier() -> None:
    frame = pd.DataFrame(
        [
            {"account_id": "acct-1", "text": "Asked about renewal"},
            {"user_id": "acct-2", "text": "Requested a support follow-up"},
        ]
    )
    invalid = pd.DataFrame(
        [
            {"user_id": "acct-1", "text": "Useful interaction"},
            {"account_id": None, "text": "Unscoped interaction"},
        ]
    )

    assert validate_crm_upload(frame) == []
    assert "Each CRM interaction must include a non-empty user_id or account_id." in validate_crm_upload(invalid)


def test_crm_dates_are_normalized_to_utc_without_mutating_upload() -> None:
    uploaded = pd.DataFrame([{"date": pd.Timestamp("2025-12-25"), "text": "Interaction"}])

    normalized = normalize_crm_dates(uploaded)

    assert str(normalized.loc[0, "date"].tz) == "UTC"
    assert uploaded.loc[0, "date"].tz is None


def test_priority_order_uses_risk_then_revenue_then_probability() -> None:
    records = pd.DataFrame(
        [
            {"user_id": "low", "risk_band": "low", "revenue_at_risk": 1000, "churn_probability": 0.9},
            {"user_id": "medium", "risk_band": "medium", "revenue_at_risk": 500, "churn_probability": 0.5},
            {"user_id": "high-low", "risk_band": "high", "revenue_at_risk": 300, "churn_probability": 0.8},
            {"user_id": "high-top", "risk_band": "high", "revenue_at_risk": 300, "churn_probability": 0.9},
            {"user_id": "high-revenue", "risk_band": "high", "revenue_at_risk": 400, "churn_probability": 0.7},
        ]
    )

    assert prioritize_records(records)["user_id"].tolist() == [
        "high-revenue",
        "high-top",
        "high-low",
        "medium",
        "low",
    ]
    assert "_risk_order" not in records.columns


@pytest.mark.parametrize("page", ["Command Center", "Accounts", "Decisions", "Data", "Audit Log"])
def test_streamlit_pages_render_without_analysis(page: str) -> None:
    app_path = str(Path(__file__).parents[1] / "app.py")
    app = AppTest.from_file(app_path).run(timeout=30)

    app.radio[0].set_value(page).run(timeout=30)

    assert not app.exception

