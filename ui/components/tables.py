from __future__ import annotations

import pandas as pd
import streamlit as st


def render_priority_table(records: pd.DataFrame, decisions: list[dict[str, object]]) -> None:
    decision_by_user = {str(item.get("user_id")): item for item in decisions}
    table = records.sort_values("revenue_at_risk", ascending=False, kind="mergesort").copy()
    decision_records = table["user_id"].astype(str).map(decision_by_user)
    table["status"] = decision_records.map(lambda item: item.get("status") if isinstance(item, dict) else "Not reviewed")
    table["action"] = decision_records.map(lambda item: item.get("recommended_action") if isinstance(item, dict) else "—")
    table = table.rename(columns={"user_id": "Account", "risk_band": "Risk", "churn_probability": "Churn probability", "revenue_at_risk": "Revenue at risk", "key_driver": "Key driver", "status": "Status", "action": "Action"})
    visible = ["Account", "Risk", "Churn probability", "Revenue at risk", "Key driver", "Status", "Action"]
    st.dataframe(table.loc[:, visible].head(20), hide_index=True, use_container_width=True, column_config={"Churn probability": st.column_config.NumberColumn(format="%.1%%"), "Revenue at risk": st.column_config.NumberColumn(format="$%.2f")})
