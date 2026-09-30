from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.application import analyze_monthly_records
from src.data.validator import validate_prediction_monthly
from ui.components.layout import page_header
from ui.runtime import get_analysis_resources, load_uploaded_dataset

logger = logging.getLogger(__name__)


def render_data() -> None:
    page_header("Data", "Load a user-month business dataset, inspect its schema, and run the GARUDA-X inference pipeline.")
    st.markdown("Supported formats: **CSV** and **Parquet**. Prediction requires one row per user and month, with the trained model feature columns.")
    upload = st.file_uploader("Upload structured business data", type=["csv", "parquet"], key="business_upload")
    if upload is not None:
        try:
            if st.session_state.get("business_data_name") != upload.name:
                st.session_state.pop("analysis", None)
                st.session_state.pop("analysis_month", None)
            st.session_state["business_data"] = load_uploaded_dataset(upload.getvalue(), upload.name)
            st.session_state["business_data_name"] = upload.name
        except Exception as error:
            logger.exception("Uploaded dataset could not be loaded")
            st.error(f"File could not be loaded ({type(error).__name__}). Check the file format and app logs.")
    frame = st.session_state.get("business_data")
    if frame is None:
        st.info("No dataset is loaded. Data and model artifacts are not bundled as a demo dataset; upload your structured business data to begin.")
        _model_status()
        return
    if frame.empty:
        st.error("The uploaded dataset contains no rows.")
        return
    st.subheader("Detected file")
    st.write(f"**{st.session_state.get('business_data_name', 'Uploaded dataset')}** · {len(frame):,} rows · {len(frame.columns)} columns")
    report = validate_prediction_monthly(frame)
    if report.is_valid:
        st.success("Stage 2 prediction schema check passed for the uploaded dataset.")
    else:
        st.error("The uploaded file does not match the model input schema.")
        for issue in report.errors:
            st.write(f"- {issue}")
    if report.null_counts:
        with st.expander("Null-count report"):
            st.caption("Numeric and categorical signals may be null when supported by the saved preprocessor. User ID, month, and MRR must be present.")
            st.dataframe(pd.DataFrame([{"Column": name, "Null rows": count} for name, count in report.null_counts.items()]), hide_index=True, use_container_width=True)
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
    month_label = st.selectbox("Reporting month", month_values, index=len(month_values) - 1)
    st.caption("Only the selected month is scored, avoiding repeat inference over the full historical panel.")
    if st.button("Run analysis", type="primary", disabled=not report.is_valid):
        try:
            predictor, shap_engine, _embedding_model = get_analysis_resources()
            selected_month = months.dt.to_period("M").astype(str).eq(month_label)
            selected_records = frame.loc[selected_month].copy()
            analysis = analyze_monthly_records(selected_records, predictor=predictor, shap_engine=shap_engine)
            st.session_state["analysis"] = analysis
            st.session_state["analysis_month"] = month_label
            st.session_state["models_loaded"] = True
            st.session_state.pop("model_error", None)
            st.success(f"Scored {len(analysis.records):,} account rows for {month_label}.")
        except Exception as error:
            logger.exception("GARUDA-X analysis failed")
            st.session_state["model_error"] = f"Analysis resource failure ({type(error).__name__}); check app logs."
            st.error(f"Analysis failed ({type(error).__name__}). Check the input schema, required model files, and app logs.")
    _model_status()


def _model_status() -> None:
    from src.nlp.transformer import DEFAULT_MODEL_PATH
    from src.prediction.predictor import DEFAULT_ARTIFACT_DIR

    st.subheader("Runtime assets")
    st.write({"XGBoost artifacts": "present" if DEFAULT_ARTIFACT_DIR.is_dir() else "missing", "MiniLM deployment model": "present" if DEFAULT_MODEL_PATH.is_dir() else "missing", "Loaded for inference": st.session_state.get("models_loaded", False)})
