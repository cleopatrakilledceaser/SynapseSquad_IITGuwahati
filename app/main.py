"""GlucoTwin clinician dashboard (Part 9). Run: streamlit run app/main.py   (or: make app).

Offline: reads only the bundle ``app/demo`` built by ``make app-data``. Research prototype for
decision support, not a medical device.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.dash import pages, style  # noqa: E402

st.set_page_config(page_title="GlucoTwin", layout="wide", initial_sidebar_state="expanded")
style.inject()

NAV = [
    st.Page(pages.overview, title="Overview", url_path="overview", default=True),
    st.Page(pages.live_twin, title="Live twin", url_path="live"),
    st.Page(pages.what_if_lab, title="What-if lab", url_path="whatif"),
    st.Page(pages.twin_state, title="Twin state", url_path="twin"),
    st.Page(pages.personalization, title="Personalization", url_path="personalization"),
    st.Page(pages.input_tiers, title="Input tiers", url_path="tiers"),
    st.Page(pages.data_quality, title="Data quality", url_path="quality"),
    st.Page(pages.evidence, title="Evidence", url_path="evidence"),
    st.Page(pages.export, title="Export", url_path="export"),
]
st.navigation(NAV, position="sidebar").run()
