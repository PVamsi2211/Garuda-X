from __future__ import annotations

import streamlit as st


def page_header(title: str, subtitle: str) -> None:
    st.markdown("<p class='gx-eyebrow'>GARUDA-X · DECISION ENGINE</p>", unsafe_allow_html=True)
    st.title(title)
    st.markdown(f"<p class='gx-subtitle'>{subtitle}</p>", unsafe_allow_html=True)
