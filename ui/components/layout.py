from __future__ import annotations

import streamlit as st


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='gx-eyebrow'>GARUDA-X · DECISION ENGINE</div><h1 class='gx-title'>{title}</h1><p class='gx-subtitle'>{subtitle}</p>", unsafe_allow_html=True)
