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

from src.dash import pages, style, units  # noqa: E402

st.set_page_config(page_title="GlucoTwin", layout="wide", initial_sidebar_state="expanded")
style.inject()

PAGES = {
    "overview": st.Page(pages.overview, title="Overview", icon=":material/dashboard:", default=True),
    "live": st.Page(pages.live_twin, title="Live twin", icon=":material/monitor_heart:", url_path="live"),
    "report": st.Page(pages.glucose_report, title="Glucose report", icon=":material/bar_chart:", url_path="report"),
    "whatif": st.Page(pages.what_if_lab, title="What-if lab", icon=":material/science:", url_path="whatif"),
    "twin": st.Page(pages.twin_state, title="Twin state", icon=":material/hub:", url_path="twin"),
    "personalization": st.Page(pages.personalization, title="Personalization", icon=":material/person_search:", url_path="personalization"),
    "tiers": st.Page(pages.input_tiers, title="Input tiers", icon=":material/stacked_bar_chart:", url_path="tiers"),
    "quality": st.Page(pages.data_quality, title="Data quality", icon=":material/verified:", url_path="quality"),
    "evidence": st.Page(pages.evidence, title="Evidence", icon=":material/fact_check:", url_path="evidence"),
    "export": st.Page(pages.export, title="Export", icon=":material/picture_as_pdf:", url_path="export"),
}
pages.PAGES.update(PAGES)
unit_choice = st.sidebar.radio(
    "Glucose units", units.UNITS, horizontal=True, key="unit", help="Evidence tables stay in mg/dL."
)
units.set_unit(unit_choice)
st.navigation(list(PAGES.values()), position="sidebar").run()
