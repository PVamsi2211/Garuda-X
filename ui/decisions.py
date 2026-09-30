from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.decision.errors import DecisionError
from ui.components.layout import page_header

logger = logging.getLogger(__name__)


def render_decisions(service: object) -> None:
    page_header("Decisions", "Review deterministic playbook recommendations and record a human decision.")
    decisions = service.list_decisions()
    if not decisions:
        st.info("No decisions have been created. Open Accounts and create a recommendation from a scored account.")
        return
    frame = pd.DataFrame([{"Decision": _short_id(item["decision_id"]), "Account": item.get("account_id") or item.get("user_id"), "Risk": str(item["risk_band"]).title(), "Revenue at risk": item["revenue_at_risk"], "Recommendation": item["recommended_action"], "Review status": item["status"], "Reviewer": item.get("approver_id") or "—"} for item in decisions])
    frame["Recommendation"] = frame["Recommendation"].astype(str).map(lambda value: value if len(value) <= 76 else value[:73] + "…")
    st.dataframe(frame, hide_index=True, use_container_width=True, column_config={"Revenue at risk": st.column_config.NumberColumn(format="$%.2f")})
    decision_ids = [item["decision_id"] for item in decisions]
    focus_id = st.session_state.get("focused_decision_id")
    index = decision_ids.index(focus_id) if focus_id in decision_ids else 0
    selected_id = st.selectbox("Decision workspace", decision_ids, index=index, format_func=_short_id)
    decision = service.get_decision(selected_id)
    st.markdown("### Decision workspace")
    summary = st.columns(4)
    summary[0].metric("Account", str(decision.get("account_id") or decision.get("user_id") or "—"))
    summary[1].metric("Risk", str(decision["risk_band"]).title())
    summary[2].metric("Revenue at risk", f"${float(decision['revenue_at_risk']):,.2f}")
    summary[3].metric("Review status", decision["status"])
    left, right = st.columns(2)
    with left:
        st.subheader("Recommendation")
        st.write(f"**Recommended action:** {decision['recommended_action']}")
        st.write(decision["rationale"])
        st.write(f"**Playbook:** {decision['playbook_id']} · **Owner:** {decision['owner']} · **Priority:** {decision['priority']}")
        crm_count = len(decision.get("crm_evidence", []))
        driver_count = len(decision.get("primary_drivers", []))
        st.write(f"**Evidence summary:** {driver_count} structured risk driver(s) · {crm_count} CRM interaction(s) · {decision.get('evidence_status', 'structured_only').replace('_', ' ').title()}")
        st.write(f"**Original action:** {decision['original_recommended_action']}")
    with right:
        st.subheader("Review status")
        st.write(f"**Decision ID:** `{_short_id(selected_id)}`")
        st.caption(f"Full identifier: `{selected_id}`")
        st.write(f"**Status:** {decision['status']}")
        st.write(f"**Human approval required:** {'Yes' if decision['requires_human_approval'] else 'No'}")
        st.write(f"**Execution authorized:** {'Yes' if service.is_execution_authorized(selected_id) else 'No'}")
        if decision.get("status_updated_at"):
            st.write(f"**Updated:** {decision['status_updated_at']}")
    st.subheader("Human review")
    if decision["status"] == "PENDING":
        approver = st.text_input("Reviewer identifier", key=f"approver-{selected_id}").strip()
        approve, modify, reject = st.tabs(["Approve", "Modify", "Reject"])
        with approve:
            if st.button("Approve recommendation", type="primary", key=f"approve-{selected_id}", disabled=not approver):
                _transition(lambda: service.approve_decision(selected_id, approver), "approved")
        with modify:
            action = st.text_input("Final reviewed action", value=str(decision["recommended_action"]), key=f"action-{selected_id}")
            reason = st.text_area("Reason for modification", key=f"modify-reason-{selected_id}")
            if st.button("Save modified decision", key=f"modify-{selected_id}", disabled=not approver or not action.strip() or not reason.strip()):
                _transition(lambda: service.modify_decision(selected_id, approver, action, reason), "modified")
        with reject:
            reason = st.text_area("Reason for rejection", key=f"reject-reason-{selected_id}")
            if st.button("Reject recommendation", key=f"reject-{selected_id}", disabled=not approver or not reason.strip()):
                _transition(lambda: service.reject_decision(selected_id, approver, reason), "rejected")
    elif decision["status"] in {"APPROVED", "MODIFIED"}:
        st.success(f"Human-reviewed by {decision.get('approver_id')} · {decision['status']}")
        if not decision.get("execution_authorized") and st.button("Record execution authorization", key=f"authorize-{selected_id}"):
            try:
                service.authorize_execution(selected_id)
                st.success("Authorization was recorded in the audit log. GARUDA-X does not execute the business action.")
                st.rerun()
            except DecisionError as error:
                logger.exception("Execution authorization was rejected")
                st.error(f"Authorization failed: {error}")
    else:
        st.info(f"This decision is finalized as {decision['status']}.")
    with st.expander("Decision audit history"):
        st.dataframe(pd.DataFrame(service.get_decision_history(selected_id)), hide_index=True, use_container_width=True)


def _transition(operation: object, label: str) -> None:
    try:
        operation()
        st.success(f"Decision {label} and audit event recorded.")
        st.rerun()
    except DecisionError as error:
        logger.exception("Decision review action failed")
        st.error(f"Decision could not be {label}: {error}")


def _short_id(value: object) -> str:
    return f"DEC-{str(value)[:8].upper()}"

