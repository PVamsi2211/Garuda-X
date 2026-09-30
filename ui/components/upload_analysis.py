from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable

import pandas as pd
import streamlit as st

from src.application import analyze_monthly_records
from src.data.validator import validate_prediction_monthly
from ui.runtime import get_analysis_resources, load_uploaded_dataset

logger = logging.getLogger(__name__)


def render_analysis_uploader(*, compact: bool = False, after_upload: Callable[[], None] | None = None) -> None:
    upload = st.file_uploader("Upload structured business data", type=["csv", "parquet"], key="business_upload")
    if upload is not None:
        payload = upload.getvalue()
        signature = hashlib.sha256(payload).hexdigest()
        if st.session_state.get("business_data_signature") != signature:
            _invalidate_business_session()
            st.session_state["business_dataset_changed_notice"] = True
            try:
                st.session_state["business_data"] = load_uploaded_dataset(payload, upload.name)
                st.session_state["business_data_name"] = upload.name
                st.session_state["business_data_signature"] = signature
                st.session_state.pop("upload_error", None)
            except Exception as error:
                logger.exception("Uploaded dataset could not be loaded")
                st.session_state.pop("business_data", None)
                st.session_state["upload_error"] = f"File could not be loaded ({type(error).__name__}). Check the file format and app logs."
        else:
            st.session_state["business_data_name"] = upload.name
    elif st.session_state.get("business_data_signature"):
        _invalidate_business_session()
        st.session_state["business_dataset_changed_notice"] = True
    if st.session_state.pop("business_dataset_changed_notice", False):
        st.session_state["business_dataset_changed_message"] = True
        st.rerun()
    if after_upload is not None:
        after_upload()
    if st.session_state.pop("business_dataset_changed_message", False):
        st.info("Dataset changed. Run analysis again.")
    if st.session_state.get("upload_error"):
        st.error(st.session_state["upload_error"])
    frame = st.session_state.get("business_data")
    if frame is None:
        if not compact:
            st.info("No dataset is loaded. Data and model artifacts are not bundled as a demo dataset; upload your structured business data to begin.")
        return
    if frame.empty:
        st.error("The uploaded dataset contains no rows.")
        return
    report = validate_prediction_monthly(frame)
    if compact:
        st.caption(f"Detected file: **{st.session_state.get('business_data_name', 'Uploaded dataset')}** · {len(frame):,} rows · {len(frame.columns)} columns")
    else:
        st.subheader("Detected file")
        st.write(f"**{st.session_state.get('business_data_name', 'Uploaded dataset')}** · {len(frame):,} rows · {len(frame.columns)} columns")
    if report.is_valid:
        st.success("Prediction schema check passed for the uploaded dataset.")
    else:
        st.error("The uploaded file does not match the model input schema.")
        for issue in report.errors:
            st.write(f"- {issue}")
    if report.null_counts and not compact:
        with st.expander("Null-count report"):
            st.caption("Numeric and categorical signals may be null when supported by the saved preprocessor. User ID, month, and MRR must be present.")
            st.dataframe(pd.DataFrame([{"Column": name, "Null rows": count} for name, count in report.null_counts.items()]), hide_index=True, use_container_width=True)
    if not compact:
        st.dataframe(frame.head(12), hide_index=True, use_container_width=True)
    if not report.is_valid:
        return
    try:
        months = pd.to_datetime(frame["month"], errors="coerce")
        if months.isna().any():
            st.error("Month contains values that cannot be interpreted as dates.")
            return
        month_values = sorted(months.dt.to_period("M").astype(str).unique().tolist())
    except Exception as error:
        logger.exception("Could not inspect dataset month values")
        st.error(f"The month column could not be interpreted: {error}")
        return
    if st.session_state.get("analysis_reporting_month") not in month_values:
        st.session_state.pop("analysis_reporting_month", None)
    month_label = st.selectbox("Reporting month", month_values, index=len(month_values) - 1, key="analysis_reporting_month")
    st.caption("Only the selected month is scored, avoiding repeat inference over the full historical panel.")
    if st.button("Run analysis", type="primary", key="run_monthly_analysis"):
        analysis_succeeded = False
        try:
            predictor, shap_engine, _embedding_model = get_analysis_resources()
            selected_month = months.dt.to_period("M").astype(str).eq(month_label)
            selected_records = frame.loc[selected_month].copy()
            analysis = analyze_monthly_records(selected_records, predictor=predictor, shap_engine=shap_engine)
            st.session_state["analysis"] = analysis
            st.session_state["analysis_month"] = month_label
            st.session_state["models_loaded"] = True
            st.session_state.pop("model_error", None)
            st.session_state["analysis_completed_message"] = f"Analysis completed — {len(analysis.records):,} accounts scored for {pd.Timestamp(month_label).strftime('%b %Y')}."
            st.session_state["navigate_to_command_center"] = True
            analysis_succeeded = True
        except (ValueError, TypeError) as error:
            logger.exception("GARUDA-X rejected uploaded analysis input")
            st.session_state["model_error"] = f"Analysis input failure ({type(error).__name__})."
            st.error(f"Analysis failed: {error}")
        except Exception as error:
            logger.exception("GARUDA-X analysis failed")
            st.session_state["model_error"] = f"Analysis resource failure ({type(error).__name__}); check app logs."
            st.error(f"Analysis failed ({type(error).__name__}). Check the input schema, required model files, and app logs.")
        if analysis_succeeded:
            st.rerun()


def _invalidate_business_session() -> None:
    for key in (
        "business_data",
        "business_data_name",
        "business_data_signature",
        "analysis",
        "analysis_month",
        "analysis_reporting_month",
        "analysis_completed_message",
        "focused_decision_id",
        "selected_account",
        "models_loaded",
        "decision_service",
        "decision_reporting_periods",
        "model_error",
    ):
        st.session_state.pop(key, None)

