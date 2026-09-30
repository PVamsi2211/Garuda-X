from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.application import MonthlyAnalysis
from ui.components.layout import page_header
from ui.components.tables import render_priority_table


def render_dashboard(analysis: MonthlyAnalysis | None, service: object) -> None:
    page_header("Command Center", "Customer risk and revenue exposure for the selected reporting month.")
    if analysis is None:
        st.info("Load a model-compatible CSV or Parquet file in Data, then run an analysis to populate the command center.")
        return
    records = analysis.records
    decisions = service.list_decisions()
    pending = sum(item.get("status") == "PENDING" for item in decisions)
    latest_month = pd.to_datetime(records["month"]).max().strftime("%b %Y")
    columns = st.columns(4)
    values = (
        ("Total accounts", f"{records['user_id'].nunique():,}"),
        ("High risk", f"{int((records['risk_band'] == 'high').sum()):,}"),
        ("Revenue at risk", f"${records['revenue_at_risk'].sum():,.0f}"),
        ("Pending review", f"{pending:,}"),
    )
    for column, (label, value) in zip(columns, values):
        with column:
            st.metric(label, value)
    st.caption(f"Scored from actual uploaded records for {latest_month}. Model probabilities are uncalibrated churn-risk estimates.")
    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("Risk vs. revenue exposure")
        chart = px.scatter(records, x="churn_probability", y="revenue_at_risk", color="risk_band", hover_name="user_id", hover_data={"mrr": ":$,.2f", "churn_probability": ":.1%", "revenue_at_risk": ":$,.2f"}, color_discrete_map={"low": "#8da39b", "medium": "#d6a84b", "high": "#e47767"}, labels={"churn_probability": "Churn probability", "revenue_at_risk": "Revenue at risk", "risk_band": "Risk"})
        chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8, r=8, t=8, b=8), legend_title_text="Risk band")
        st.plotly_chart(chart, use_container_width=True)
    with right:
        st.subheader("Risk distribution")
        counts = records["risk_band"].value_counts().reindex(["high", "medium", "low"], fill_value=0).rename_axis("risk_band").reset_index(name="accounts")
        chart = px.bar(counts, x="risk_band", y="accounts", color="risk_band", color_discrete_map={"low": "#8da39b", "medium": "#d6a84b", "high": "#e47767"}, labels={"risk_band": "Risk band", "accounts": "Accounts"})
        chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(l=8, r=8, t=8, b=8))
        st.plotly_chart(chart, use_container_width=True)
    st.subheader("Priority accounts")
    render_priority_table(records, decisions)
