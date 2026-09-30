from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.application import MonthlyAnalysis, explanation_for_user
from src.decision.errors import DecisionError
from src.nlp.crm_retrieval import retrieve_account_evidence
from ui.components.layout import page_header
from ui.components.tables import prioritize_records
from ui.runtime import get_crm_index

logger = logging.getLogger(__name__)


def render_accounts(analysis: MonthlyAnalysis | None, service: object) -> None:
    page_header("Accounts", "Account intelligence combines the saved churn model, TreeSHAP evidence, and scoped CRM retrieval.")
    if analysis is None:
        st.info("Run an analysis from the Data page to view account intelligence.")
        return
    prioritized = prioritize_records(analysis.records)
    account_ids = prioritized["user_id"].astype(str).tolist()
    selected = st.selectbox("Select account", account_ids, key="selected_account")
    row = analysis.records.loc[analysis.records["user_id"].astype(str) == selected].iloc[0]
    explanation = explanation_for_user(analysis, row["user_id"])
    crm_result = None
    as_of = pd.Timestamp(row["month"]).to_period("M").end_time.tz_localize("UTC")
    crm_index = None
    try:
        crm_index = get_crm_index()
        crm_result = retrieve_account_evidence(crm_index, structured_explanation=explanation, user_id=row["user_id"], as_of=as_of)
        if not crm_result.get("crm_evidence") and "account_id" in crm_index.records.columns:
            crm_result = retrieve_account_evidence(crm_index, structured_explanation=explanation, account_id=row["user_id"], as_of=as_of)
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
    with st.expander("All structured business signals"):
        st.dataframe(_driver_frame(explanation["feature_explanations"]), hide_index=True, use_container_width=True)
        st.caption(f"SHAP output space: {explanation['shap_output_space']}. These are model explanations, not causal findings.")
    st.subheader("CRM evidence")
    if crm_result is None:
        st.warning("CRM retrieval failed. Recommendation creation is disabled until retrieval is available; check app logs and model/data assets.")
    elif crm_result.get("evidence_status") == "retrieved" and crm_result.get("crm_evidence"):
        st.success("Evidence status: STRUCTURED + CRM")
        st.caption(f"{len(crm_result['crm_evidence'])} supporting CRM interaction(s) · eligible through {as_of.strftime('%d %b %Y')}.")
        for evidence in crm_result["crm_evidence"]:
            with st.container(border=True):
                source_id = evidence.get("interaction_id")
                source_row = evidence.get("source_row")
                source_label = f"Interaction {source_id}" if source_id else f"Source row {source_row}" if source_row is not None else "Source reference unavailable"
                st.caption(f"{evidence.get('date') or 'Date unavailable'} · {evidence.get('interaction_type') or 'Interaction'} · similarity {evidence.get('similarity_score', 0):.2f} · {source_label}")
                st.write(evidence["text"])
    else:
        st.info("Evidence status: STRUCTURED ONLY")
        st.write("No matching CRM evidence found. This does not indicate that the customer has no issues.")
    st.subheader("Decision recommendation")
    period = str(pd.Timestamp(row["month"]).to_period("M"))
    period_key = (str(row["user_id"]), period)
    created_periods = st.session_state.setdefault("decision_reporting_periods", {})
    existing_id = created_periods.get(period_key)
    if existing_id is not None:
        try:
            already_recommended = service.get_decision(existing_id).get("status") == "PENDING"
        except DecisionError:
            created_periods.pop(period_key, None)
            already_recommended = False
    else:
        already_recommended = False
    if already_recommended:
        st.info("An active decision already exists for this account and reporting period.")
    if st.button("Create playbook recommendation", type="primary", key=f"recommend-{selected}", disabled=crm_result is None or already_recommended):
        try:
            result = service.create_recommendation(user_id=row["user_id"], account_id=crm_result.get("account_id"), risk_score=float(row["churn_probability"]), mrr=float(row["mrr"]), shap_result=explanation, crm_result=crm_result, as_of=as_of)
            st.session_state["focused_decision_id"] = result["decision_id"]
            created_periods[period_key] = result["decision_id"]
            st.success(f"Recommendation {result['decision_id']} created and queued for human review.")
        except (DecisionError, ValueError, TypeError) as error:
            logger.exception("Could not create a decision recommendation")
            st.error(f"Recommendation could not be created: {error}")
    st.caption("Recommendations are advisory. A human reviewer must approve, modify, or reject each decision before execution authorization is possible.")


def _driver_frame(items: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame([{"Signal": str(item["feature"]).replace("_", " ").title(), "Observed value": "—" if item.get("feature_value") is None else str(item["feature_value"]), "SHAP contribution (raw margin)": item.get("shap_value"), "Direction": str(item.get("direction", "")).replace("_", " ").title()} for item in items])

