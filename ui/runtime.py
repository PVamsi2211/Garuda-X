from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.audit.audit_logger import AuditLogger
from src.data.loader import load_data
from src.decision.approval import DecisionService
from src.explainability.shap_engine import ShapEngine
from src.nlp.crm_retrieval import CRMEmbeddingIndex, DEFAULT_CRM_PATH, load_crm_records
from src.nlp.transformer import DEFAULT_MODEL_PATH, load_embedding_model
from src.prediction.predictor import DEFAULT_ARTIFACT_DIR, ChurnPredictor, load_churn_predictor


@st.cache_resource(show_spinner="Loading the saved GARUDA-X models...")
def get_analysis_resources() -> tuple[ChurnPredictor, ShapEngine, object]:
    predictor = load_churn_predictor(DEFAULT_ARTIFACT_DIR)
    shap_engine = ShapEngine(predictor)
    embedding_model = load_embedding_model(DEFAULT_MODEL_PATH, device="cpu")
    return predictor, shap_engine, embedding_model


@st.cache_resource(show_spinner="Preparing account-scoped CRM retrieval...")
def _get_default_crm_index() -> CRMEmbeddingIndex:
    _, _, embedding_model = get_analysis_resources()
    if DEFAULT_CRM_PATH.is_file():
        records = load_crm_records(DEFAULT_CRM_PATH)
    else:
        records = pd.DataFrame(columns=["text", "user_id"])
    return CRMEmbeddingIndex(records, model=embedding_model)


def get_crm_index() -> CRMEmbeddingIndex:
    signature = st.session_state.get("crm_data_signature")
    if signature:
        if st.session_state.get("crm_index_signature") == signature:
            cached = st.session_state.get("crm_index")
            if isinstance(cached, CRMEmbeddingIndex):
                return cached
        _, _, embedding_model = get_analysis_resources()
        uploaded = st.session_state.get("crm_data")
        if st.session_state.get("crm_data_valid") and isinstance(uploaded, pd.DataFrame):
            index = CRMEmbeddingIndex(normalize_crm_dates(uploaded), model=embedding_model)
        else:
            index = CRMEmbeddingIndex(pd.DataFrame(columns=["user_id", "text"]), model=embedding_model)
        st.session_state["crm_index"] = index
        st.session_state["crm_index_signature"] = signature
        return index
    return _get_default_crm_index()


@st.cache_data(show_spinner=False)
def load_uploaded_dataset(payload: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".parquet", ".jsonl"}:
        raise ValueError("Only CSV, Parquet, and JSONL uploads are supported")
    with tempfile.TemporaryDirectory(prefix="garuda_x_upload_") as directory:
        path = Path(directory) / f"upload{suffix}"
        path.write_bytes(payload)
        return load_data(path)


def normalize_crm_dates(records: pd.DataFrame) -> pd.DataFrame:
    normalized = records.copy()
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce", utc=True)
    return normalized


def get_decision_service() -> DecisionService:
    if "decision_service" not in st.session_state:
        st.session_state["decision_service"] = DecisionService(audit_logger=AuditLogger())
    return st.session_state["decision_service"]

