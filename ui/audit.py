from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components.layout import page_header


def render_audit(service: object) -> None:
    page_header("Audit Log", "Append-only events from the active application session.")
    st.warning("Audit events are held in memory and will not persist after this Streamlit session or process restarts.")
    events = service.get_audit_history()
    if not events:
        st.info("No decision or approval events have been recorded in this session.")
        return
    frame = pd.DataFrame([{"Timestamp": event["timestamp"], "Decision": f"DEC-{str(event['decision_id'])[:8].upper()}", "Account": event.get("account_id") or event.get("user_id"), "Event": event["event_type"], "Review status": event["new_status"], "Reviewer": event.get("approver_id") or "—"} for event in events])
    st.dataframe(frame, hide_index=True, use_container_width=True)
    with st.expander("Event details"):
        details = pd.DataFrame([{"Timestamp": event["timestamp"], "Full decision ID": event["decision_id"], "Event ID": event.get("event_id"), "Previous status": event.get("previous_status") or "—", "New status": event["new_status"], "Reviewer": event.get("approver_id") or "—", "Action": event.get("final_action") or event.get("original_recommendation") or "—", "Reason": event.get("reason") or "—", "Metadata": str(event.get("metadata") or {})} for event in events])
        st.dataframe(details, hide_index=True, use_container_width=True)

