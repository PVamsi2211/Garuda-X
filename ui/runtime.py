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
def get_crm_index() -> CRMEmbeddingIndex:
    _, _, embedding_model = get_analysis_resources()
    if DEFAULT_CRM_PATH.is_file():
        records = load_crm_records(DEFAULT_CRM_PATH)
    else:
        records = pd.DataFrame(columns=["text", "user_id"])
    return CRMEmbeddingIndex(records, model=embedding_model)


@st.cache_data(show_spinner=False)
def load_uploaded_dataset(payload: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise ValueError("Only CSV and Parquet uploads are supported")
    with tempfile.TemporaryDirectory(prefix="garuda_x_upload_") as directory:
        path = Path(directory) / f"upload{suffix}"
        path.write_bytes(payload)
        return load_data(path)


def get_decision_service() -> DecisionService:
    if "decision_service" not in st.session_state:
        st.session_state["decision_service"] = DecisionService(audit_logger=AuditLogger())
    return st.session_state["decision_service"]
