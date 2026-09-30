import streamlit as st

from src.application import MonthlyAnalysis
from ui.accounts import render_accounts
from ui.audit import render_audit
from ui.dashboard import render_dashboard
from ui.data import render_data
from ui.decisions import render_decisions
from ui.runtime import get_decision_service


def main() -> None:
    st.set_page_config(page_title="GARUDA-X | Decision Engine", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
    st.markdown("<style>.stApp{background:#111315}.block-container{max-width:1440px;padding-top:2rem;padding-bottom:3rem}.gx-eyebrow{color:#d6ad58;font-size:.72rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase}.gx-title{font-size:2.25rem;font-weight:650;letter-spacing:-.035em;margin:.35rem 0}.gx-subtitle{color:#aab0b7;margin:0 0 1.6rem}.stMetric{background:#1a1d20;border:1px solid #303337;border-radius:12px;padding:1rem 1.1rem}.stMetric label{color:#adb3ba}.stMetric [data-testid=stMetricValue]{color:#f1f2f3}.stButton>button[kind=primary]{background:#bd9442;border-color:#bd9442;color:#111315}.stButton>button[kind=primary]:hover{background:#d2ab5e;border-color:#d2ab5e;color:#111315}[data-testid=stSidebar]{background:#17191b;border-right:1px solid #303337}</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("# GARUDA-X")
        st.caption("AI-Powered Customer Risk & Revenue Decision Engine")
        st.divider()
        page = st.radio("Navigation", ["Command Center", "Accounts", "Decisions", "Data", "Audit Log"], label_visibility="collapsed")
        if st.session_state.get("model_error"):
            st.error("System Degraded")
            st.caption(st.session_state["model_error"])
        elif st.session_state.get("models_loaded"):
            st.success("System Ready")
            st.caption("XGBoost, TreeSHAP, and local MiniLM loaded")
        else:
            st.info("Awaiting analysis")
            st.caption("Models load on demand when a dataset is analyzed.")
        st.divider()
        st.caption("Human approval is required for every recommended action.")

    service = get_decision_service()
    analysis: MonthlyAnalysis | None = st.session_state.get("analysis")
    if page == "Command Center":
        render_dashboard(analysis, service)
    elif page == "Accounts":
        render_accounts(analysis, service)
    elif page == "Decisions":
        render_decisions(service)
    elif page == "Data":
        render_data()
    else:
        render_audit(service)


if __name__ == "__main__":
    main()
