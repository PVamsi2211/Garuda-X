from __future__ import annotations

import hashlib
import logging

import pandas as pd
import streamlit as st

from ui.runtime import load_uploaded_dataset

logger = logging.getLogger(__name__)


def validate_crm_upload(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    identifiers = [column for column in ("user_id", "account_id") if column in frame.columns]
    if not identifiers:
        errors.append("CRM evidence requires user_id or account_id for safe account-scoped retrieval.")
    if "text" not in frame.columns:
        errors.append("CRM evidence requires a text field.")
    if frame.empty:
        errors.append("The CRM file contains no interaction rows.")
    if identifiers and not frame.empty:
        scoped = pd.Series(False, index=frame.index)
        for column in identifiers:
            values = frame[column]
            scoped |= values.notna() & values.astype(str).str.strip().ne("")
        if not scoped.all():
            errors.append("Each CRM interaction must include a non-empty user_id or account_id.")
    if "text" in frame.columns and not frame.empty:
        text = frame["text"]
        usable = text.map(lambda value: isinstance(value, str) and bool(value.strip()))
        if not usable.all():
            errors.append("Each CRM interaction must include non-empty text.")
    return errors


def render_crm_uploader() -> None:
    st.markdown("#### CRM EVIDENCE")
    st.write("Upload account-scoped CRM interactions. CRM evidence is optional and remains in this session.")
    upload = st.file_uploader("Upload account-scoped CRM interactions", type=["jsonl"], key="crm_upload")
    if upload is not None:
        payload = upload.getvalue()
        signature = hashlib.sha256(payload).hexdigest()
        if st.session_state.get("crm_data_signature") != signature:
            _invalidate_crm_session()
            st.session_state["crm_dataset_changed_notice"] = True
            st.session_state["crm_data_signature"] = signature
            try:
                frame = load_uploaded_dataset(payload, upload.name)
                errors = validate_crm_upload(frame)
                st.session_state["crm_data"] = frame
                st.session_state["crm_data_name"] = upload.name
                st.session_state["crm_validation_errors"] = errors
                st.session_state["crm_data_valid"] = not errors
                if errors:
                    st.session_state["crm_upload_error"] = None
                else:
                    st.session_state.pop("crm_upload_error", None)
            except Exception as error:
                logger.exception("Uploaded CRM dataset could not be loaded")
                st.session_state["crm_data"] = None
                st.session_state["crm_data_name"] = upload.name
                st.session_state["crm_data_valid"] = False
                st.session_state["crm_validation_errors"] = []
                st.session_state["crm_upload_error"] = f"CRM file could not be loaded ({type(error).__name__}). Confirm it is valid JSONL."
        else:
            st.session_state["crm_data_name"] = upload.name
    if st.session_state.pop("crm_dataset_changed_notice", False):
        st.session_state["crm_dataset_changed_message"] = True
        st.rerun()
    if st.session_state.pop("crm_dataset_changed_message", False):
        st.info("CRM dataset changed. Account evidence will use the new upload.")
    if st.session_state.get("crm_upload_error"):
        st.error(st.session_state["crm_upload_error"])
    frame = st.session_state.get("crm_data")
    if frame is None:
        st.info("No uploaded CRM evidence. Business-only analysis is available.")
        return
    if st.session_state.get("crm_validation_errors"):
        st.error("The uploaded CRM file cannot be used for evidence retrieval.")
        for issue in st.session_state["crm_validation_errors"]:
            st.write(f"- {issue}")
        return
    st.success(f"CRM schema check passed · {len(frame):,} interactions · account-scoped")
    st.caption(f"CRM dataset: {st.session_state.get('crm_data_name', 'Uploaded file')} · session-only source")
    synthetic = st.checkbox("This CRM file is synthetic demonstration data", key="crm_is_synthetic")
    if synthetic:
        st.caption("Provenance: uploaded synthetic demonstration data.")
    else:
        st.caption("Provenance: uploaded CRM records as supplied.")
    if "date" not in frame.columns:
        st.warning("This CRM file has no date field, so month-based evidence cutoff will exclude its records.")


def _invalidate_crm_session() -> None:
    for key in (
        "crm_data",
        "crm_data_name",
        "crm_data_signature",
        "crm_data_valid",
        "crm_validation_errors",
        "crm_upload_error",
        "crm_index",
        "crm_index_signature",
        "crm_is_synthetic",
    ):
        st.session_state.pop(key, None)

