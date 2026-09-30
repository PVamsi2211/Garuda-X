from __future__ import annotations

import streamlit as st

from ui.components.layout import page_header
from ui.components.upload_analysis import render_analysis_uploader


def render_data() -> None:
    page_header("Data", "Load a user-month business dataset, inspect its schema, and run the GARUDA-X inference pipeline.")
    st.markdown("Supported formats: **CSV** and **Parquet**. Prediction requires one row per user and month, with the trained model feature columns.")
    render_analysis_uploader()
    _model_status()


def _model_status() -> None:
    from src.nlp.transformer import DEFAULT_MODEL_PATH
    from src.prediction.predictor import DEFAULT_ARTIFACT_DIR

    st.subheader("Runtime assets")
    st.write({"XGBoost artifacts": "present" if DEFAULT_ARTIFACT_DIR.is_dir() else "missing", "MiniLM deployment model": "present" if DEFAULT_MODEL_PATH.is_dir() else "missing", "Loaded for inference": st.session_state.get("models_loaded", False)})
