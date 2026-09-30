from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.application import MonthlyAnalysis, explanation_for_user
from src.decision.errors import DecisionError
from src.nlp.crm_retrieval import retrieve_account_evidence
from ui.components.layout import page_header
from ui.runtime import get_crm_index

logger = logging.getLogger(__name__)


def render_accounts(analysis: MonthlyAnalysis | None, service: object) -> None:
    page_header("Accounts", "Account intelligence combines the saved churn model, TreeSHAP evidence, and scoped CRM retrieval.")
    if analysis is None:
        st.info("Run an analysis from the Data page to view account intelligence.")
        return
    account_ids = analysis.records["user_id"].astype(str).tolist()
    selected = st.selectbox("Select account", account_ids, key="selected_account")
    row = analysis.records.loc[analysis.records["user_id"].astype(str) == selected].iloc[0]
    explanation = explanation_for_user(analysis, row["user_id"])
    crm_result = None
    try:
        crm_result = retrieve_account_evidence(get_crm_index(), structured_explanation=explanation, user_id=row["user_id"], as_of=row["month"])
    except Exception:
        logger.exception("CRM evidence retrieval failed for selected account")
        st.session_state["model_error"] = "A required inference resource could not be loaded. Check the app logs and model files."
    st.markdown(f"### {selected}")
    metrics = st.columns(3)
    metrics[0].metric("Risk level", str(row["risk_band"]).title())
    metrics[1].metric("Churn probability", f"{row['churn_probability']:.1%}")
    metrics[2].metric("Revenue at risk", f"${row['revenue_at_risk']:,.2f}")
    st.caption("Predicts customer churn risk from structured business data. Probabilities are uncalibrated; SHAP contributions are raw-margin log-odds and are not probability contributions or causal effects.")
    left, right = st.columns(2)
    with left:
        st.subheader("Top risk drivers")
        if explanation["top_risk_drivers"]:
            st.dataframe(_driver_frame(explanation["top_risk_drivers"]), hide_index=True, use_container_width=True)
        else:
            st.write("No positive risk drivers for this account.")
    with right:
        st.subheader("Protective factors")
        if explanation["protective_factors"]:
            st.dataframe(_driver_frame(explanation["protective_factors"]), hide_index=True, use_container_width=True)
        else:
            st.write("No protective factors identified.")
    st.subheader("Structured business signals")
    st.dataframe(_driver_frame(explanation["feature_explanations"]), hide_index=True, use_container_width=True)
    st.caption(f"SHAP output space: {explanation['shap_output_space']}. These are model explanations, not causal findings.")
    st.subheader("CRM evidence")
    if crm_result is None:
        st.warning("CRM retrieval failed. Recommendation creation is disabled until retrieval is available; check app logs and model/data assets.")
    elif crm_result.get("evidence_status") == "retrieved" and crm_result.get("crm_evidence"):
        for evidence in crm_result["crm_evidence"]:
            with st.container(border=True):
                st.caption(f"{evidence.get('date') or 'Date unavailable'} · {evidence.get('interaction_type') or 'Interaction'} · similarity {evidence.get('similarity_score', 0):.2f}")
                st.write(evidence["text"])
    else:
        st.info("No matching CRM evidence found. The available corpus may be absent or lack a matching account/user identifier; this does not indicate that the customer has no issues.")
    st.subheader("Decision recommendation")
    if st.button("Create playbook recommendation", type="primary", key=f"recommend-{selected}", disabled=crm_result is None):
        try:
            as_of = pd.Timestamp(row["month"])
            if as_of.tzinfo is None:
                as_of = as_of.tz_localize("UTC")
            result = service.create_recommendation(user_id=row["user_id"], risk_score=float(row["churn_probability"]), mrr=float(row["mrr"]), shap_result=explanation, crm_result=crm_result, as_of=as_of)
            st.session_state["focused_decision_id"] = result["decision_id"]
            st.success(f"Recommendation {result['decision_id']} created and queued for human review.")
        except (DecisionError, ValueError, TypeError) as error:
            logger.exception("Could not create a decision recommendation")
            st.error(f"Recommendation could not be created: {error}")
    st.caption("Recommendations are advisory. A human reviewer must approve, modify, or reject each decision before execution authorization is possible.")


def _driver_frame(items: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame([{"Signal": str(item["feature"]).replace("_", " ").title(), "Observed value": item.get("feature_value"), "SHAP contribution (raw margin)": item.get("shap_value"), "Direction": str(item.get("direction", "")).replace("_", " ").title()} for item in items])
