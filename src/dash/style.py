"""CSS of the dashboard (docs/DESIGN.md section 2). Fonts are served locally from app/static."""

from __future__ import annotations

FONT_BASE = "/app/static/fonts"

CSS = f"""
@font-face {{ font-family: 'Source Serif 4'; font-weight: 400; font-style: normal; font-display: swap;
  src: url('{FONT_BASE}/source-serif-4-latin-400-normal.woff2') format('woff2'); }}
@font-face {{ font-family: 'Source Serif 4'; font-weight: 600; font-style: normal; font-display: swap;
  src: url('{FONT_BASE}/source-serif-4-latin-600-normal.woff2') format('woff2'); }}

:root {{ --paper:#F4F2F7; --surface:#FFFFFF; --ink:#1B2230; --slate:#5A6478; --rule:#E6E2EC; --damson:#43285E;
  --deep:#2B1B3F; --mint:#9AD3B8; --amber:#E09A12; --blue:#2B6CB0; --wine:#8C1D2B;
  --sans:'Segoe UI Variable Text','Segoe UI',system-ui,-apple-system,Roboto,'Helvetica Neue',Arial,sans-serif;
  --serif:'Source Serif 4',Georgia,serif; --shadow:0 1px 2px rgba(43,27,63,.06),0 8px 24px rgba(43,27,63,.07); }}

html, body, .stApp, [data-testid="stAppViewContainer"] {{ background: var(--paper); color: var(--ink);
  font-family: var(--sans); font-size: 15px; line-height: 22px; }}
/* font on text elements only: never on the Material icon spans (that printed "_arrow_right") */
.stApp :where(p, div, label, button, input, textarea, li, td, th, a, h1, h2, h3, h4, span):not([data-testid="stIconMaterial"]):not(.material-icons):not([class*="material-symbols"]) {{ font-family: var(--sans); }}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDeployButton"], [data-testid="stDecoration"], #MainMenu, footer {{ display: none !important; }}
.block-container {{ max-width: 1180px; padding: 28px 36px 28px 36px; }}

h1, h2, h3 {{ color: var(--ink); letter-spacing: -0.01em; text-transform: none; font-weight: 700 !important; }}
h1 {{ font-size: 30px; line-height: 36px; margin: 0; padding: 0; }}
h2 {{ font-size: 18px; line-height: 26px; margin: 28px 0 6px 0; }}
h3 {{ font-size: 16px; line-height: 24px; }}
.num {{ font-variant-numeric: tabular-nums; }}
.eyebrow {{ font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--damson); font-weight: 700; margin-bottom: 4px; }}
.pagehead {{ margin: 0 0 18px 0; }}
.pagehead .sub {{ color: var(--slate); font-size: 15px; margin-top: 4px; max-width: 760px; }}

/* sidebar: deep damson, pill navigation, brand on top */
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, var(--deep) 0%, #3A2354 100%); border-right: 0; min-width: 232px; max-width: 244px; }}
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {{ display: none; }}
[data-testid="stSidebarNav"]::before {{ content: "GlucoTwin"; display: block; color: #fff; font-weight: 800; font-size: 22px;
  letter-spacing: -0.02em; padding: 26px 20px 2px 20px; }}
[data-testid="stSidebarNav"]::after {{ content: "Glucose digital twin for clinicians"; display: block; color: #CDBFE0; font-size: 12px;
  padding: 0 20px 18px 20px; }}
[data-testid="stSidebarNav"] {{ max-height: none; }}
[data-testid="stSidebarNavLink"] {{ border-radius: 10px; margin: 2px 12px; padding: 9px 12px; min-height: 40px; color: #E9E1F3; gap: 10px; }}
[data-testid="stSidebarNavLink"] span {{ color: #E9E1F3; font-size: 15px; }}
[data-testid="stSidebarNavLink"]:hover {{ background: rgba(255,255,255,.08); }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: rgba(255,255,255,.16); box-shadow: inset 3px 0 0 var(--mint); }}
[data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: #fff; font-weight: 700; }}

/* cards: every bordered container is a white card */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{ border-radius: 16px; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--surface); border: 1px solid var(--rule) !important; border-radius: 16px;
  box-shadow: var(--shadow); padding: 4px 6px; }}
[data-testid="stExpander"] {{ background: var(--surface); border: 1px solid var(--rule) !important; border-radius: 14px; box-shadow: var(--shadow); }}
[data-testid="stExpander"] summary {{ font-weight: 600; }}
[data-testid="stPlotlyChart"] {{ border: 0; padding: 0; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--rule); border-radius: 12px; overflow: hidden; }}

/* hero decision */
.decision {{ font-family: var(--serif); font-weight: 600; font-size: 26px; line-height: 33px; margin: 2px 0 10px 0; color: var(--ink); letter-spacing: -0.01em; }}
.note {{ background: #F1ECF7; border-radius: 12px; border-left: 4px solid var(--damson); padding: 10px 14px; margin: 14px 0; color: var(--ink); }}
.banner {{ background: #FFF4D6; border: 2px solid var(--ink); border-radius: 12px; padding: 12px 16px; margin: 12px 0; font-weight: 700; }}
.disclaimer {{ border-top: 1px solid var(--rule); margin-top: 40px; padding-top: 12px; font-size: 13px; color: var(--slate); }}
.topline {{ font-size: 13px; color: var(--slate); margin: 0 0 6px 0; }}
.synthetic-tag {{ display: inline-block; border: 1px solid var(--ink); border-radius: 999px; padding: 0 9px; margin-left: 8px;
  font-size: 11px; letter-spacing: .06em; color: var(--ink); font-weight: 700; }}

/* chips: always a word, never colour alone */
.chip {{ display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 3px 12px; font-size: 13px; font-weight: 600;
  border: 1px solid var(--rule); background: #fff; color: var(--ink); white-space: nowrap; }}
.chip .dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--ink); display: inline-block; }}
.chip.low .dot {{ background: #2F8F6B; }} .chip.moderate .dot {{ background: var(--amber); }}
.chip.high .dot, .chip.very .dot {{ background: var(--wine); }} .chip.na .dot {{ background: var(--slate); }}
.bars span {{ display: inline-block; width: 14px; height: 8px; margin-right: 3px; border-radius: 2px; border: 1px solid var(--ink); }}
.bars span.on {{ background: var(--ink); }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 8px 0; }}

/* KPI tiles */
.kpis {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 6px 0 14px 0; }}
.kpi {{ background: var(--paper); border-radius: 14px; padding: 12px 16px; }}
.kpi .k {{ font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--slate); font-weight: 700; }}
.kpi .v {{ font-size: 32px; line-height: 38px; font-weight: 800; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }}
.kpi .v small {{ font-size: 14px; font-weight: 600; color: var(--slate); margin-left: 4px; letter-spacing: 0; }}
.kpi .s {{ font-size: 13px; color: var(--slate); }}

/* patient cards */
.pcard {{ display: grid; grid-template-columns: 1fr auto; gap: 4px 12px; align-items: center; }}
.pcard .name {{ font-weight: 700; font-size: 16px; }}
.pcard .st {{ color: var(--slate); font-size: 13px; grid-column: 1 / 2; }}
.pcard .big {{ font-size: 24px; font-weight: 800; font-variant-numeric: tabular-nums; text-align: right; }}
.pcard .big small {{ font-size: 12px; font-weight: 600; color: var(--slate); }}
.selected-tag {{ font-size: 11px; font-weight: 700; letter-spacing: .06em; color: var(--damson); text-transform: uppercase; }}

/* probability bars */
.pbar {{ display: grid; grid-template-columns: 170px 1fr 52px; align-items: center; gap: 14px; margin: 10px 0; }}
.pbar .track {{ position: relative; height: 12px; background: #ECE8F1; border-radius: 999px; }}
.pbar .fill {{ position: absolute; left: 0; top: 0; bottom: 0; background: linear-gradient(90deg, #6C4A8F, var(--damson)); border-radius: 999px; min-width: 4px; }}
.pbar .tick {{ position: absolute; top: -5px; bottom: -5px; width: 3px; background: var(--ink); border-radius: 2px; }}
.pbar .val {{ font-weight: 800; text-align: right; font-variant-numeric: tabular-nums; font-size: 17px; }}
.pbar .na {{ color: var(--slate); font-size: 14px; }}

/* alert row */
.alertrow {{ display: flex; gap: 12px; align-items: flex-start; padding: 10px 0; border-top: 1px solid var(--rule); }}
.alertrow:first-of-type {{ border-top: 0; }}
.alertrow .t {{ font-weight: 700; min-width: 170px; }}
.alertrow .d {{ color: var(--slate); font-size: 14px; }}
.chip.on {{ background: #FFF4D6; border-color: var(--ink); }}

/* controls */
.stButton > button, .stDownloadButton > button {{ border-radius: 10px; min-height: 42px; font-weight: 600; border: 1px solid var(--rule); background: #fff; }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{ background: var(--damson); border-color: var(--damson); color: #fff; }}
.stButton > button[kind="primary"]:hover {{ background: #583480; border-color: #583480; color: #fff; }}
[data-baseweb="select"] > div, [data-baseweb="input"] {{ border-radius: 10px; }}
[data-testid="stSlider"] [role="slider"] {{ background: var(--damson); }}
:focus-visible {{ outline: 2px solid var(--damson) !important; outline-offset: 2px; }}

@media (max-width: 900px) {{
  .block-container {{ padding: 16px; }}
  h1 {{ font-size: 24px; line-height: 29px; }} .decision {{ font-size: 21px; line-height: 27px; }}
  .kpis {{ grid-template-columns: 1fr; }} .pbar {{ grid-template-columns: 120px 1fr 48px; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
"""


def inject() -> None:
    """Add the stylesheet to the page (call once per script run)."""
    import streamlit as st

    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
