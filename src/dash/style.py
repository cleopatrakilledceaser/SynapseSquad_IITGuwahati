"""CSS of the dashboard (docs/DESIGN.md section 2). Fonts are served locally from app/static."""

from __future__ import annotations

FONT_BASE = "/app/static/fonts"

CSS = f"""
@font-face {{ font-family: 'Atkinson Hyperlegible'; font-weight: 400; font-style: normal; font-display: swap;
  src: url('{FONT_BASE}/atkinson-hyperlegible-latin-400-normal.woff2') format('woff2'); }}
@font-face {{ font-family: 'Atkinson Hyperlegible'; font-weight: 700; font-style: normal; font-display: swap;
  src: url('{FONT_BASE}/atkinson-hyperlegible-latin-700-normal.woff2') format('woff2'); }}
@font-face {{ font-family: 'Source Serif 4'; font-weight: 400; font-style: normal; font-display: swap;
  src: url('{FONT_BASE}/source-serif-4-latin-400-normal.woff2') format('woff2'); }}
@font-face {{ font-family: 'Source Serif 4'; font-weight: 600; font-style: normal; font-display: swap;
  src: url('{FONT_BASE}/source-serif-4-latin-600-normal.woff2') format('woff2'); }}

:root {{ --paper:#F5F6F8; --surface:#FFFFFF; --ink:#1B2230; --slate:#566174; --rule:#D8DDE5; --damson:#43285E;
  --sans:'Atkinson Hyperlegible','Segoe UI',system-ui,sans-serif; --serif:'Source Serif 4',Georgia,serif; }}

html, body, .stApp, [data-testid="stAppViewContainer"] {{ background: var(--paper); color: var(--ink);
  font-family: var(--sans); font-size: 16px; line-height: 24px; }}
.stApp * {{ font-family: var(--sans); }}
[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ max-width: 1200px; padding: 32px 32px 24px 32px; }}

h1, h2, h3, .serif {{ font-family: var(--serif) !important; font-weight: 600 !important; color: var(--ink);
  letter-spacing: 0; text-transform: none; }}
h1 {{ font-size: 28px; line-height: 34px; margin: 8px 0 16px 0; }}
h2 {{ font-size: 20px; line-height: 28px; margin: 32px 0 8px 0; }}
h3 {{ font-size: 20px; line-height: 28px; }}
p, li, td, th, label {{ font-size: 16px; line-height: 24px; }}
.caption, small {{ font-size: 12px; line-height: 16px; color: var(--slate); }}
.small {{ font-size: 14px; line-height: 20px; }}
.num {{ font-variant-numeric: tabular-nums; }}

/* navigation: text only, selected = 3 px damson bar + bold */
[data-testid="stSidebar"] {{ background: var(--paper); border-right: 1px solid var(--rule); min-width: 200px; max-width: 220px; }}
[data-testid="stSidebarNavLink"] {{ border-left: 3px solid transparent; border-radius: 0; padding: 8px 12px; min-height: 40px; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ border-left-color: var(--damson); font-weight: 700; background: transparent; }}
[data-testid="stSidebarNavLink"] span {{ font-size: 16px; }}

/* decision sentence and fixed texts */
.decision {{ font-family: var(--serif); font-weight: 600; font-size: 28px; line-height: 34px; margin: 8px 0 8px 0; color: var(--ink); }}
.note {{ border-left: 3px solid var(--damson); padding: 4px 0 4px 12px; margin: 16px 0; font-size: 16px; color: var(--ink); }}
.banner {{ border: 2px solid var(--ink); padding: 12px 16px; margin: 16px 0; font-weight: 700; }}
.disclaimer {{ border-top: 1px solid var(--rule); margin-top: 48px; padding-top: 12px; font-size: 14px; color: var(--ink); }}
.topline {{ font-size: 14px; color: var(--slate); margin-bottom: 4px; }}
.synthetic-tag {{ display: inline-block; border: 1px solid var(--ink); border-radius: 3px; padding: 0 8px; margin-left: 8px;
  font-size: 14px; color: var(--ink); font-weight: 700; }}

/* confidence badge: word plus bar count, plain ink outline */
.badge {{ display: inline-block; border: 1px solid var(--ink); border-radius: 3px; padding: 2px 10px; font-size: 14px; }}
.bars span {{ display: inline-block; width: 14px; height: 8px; margin-right: 3px; border: 1px solid var(--ink); }}
.bars span.on {{ background: var(--ink); }}

/* probability bars: 0 to 100 %, number at the bar end, optional alert tick */
.pbar {{ display: grid; grid-template-columns: 190px 1fr 56px; align-items: center; gap: 12px; margin: 8px 0; }}
.pbar .track {{ position: relative; height: 12px; background: var(--rule); }}
.pbar .fill {{ position: absolute; left: 0; top: 0; bottom: 0; background: var(--damson); }}
.pbar .tick {{ position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--ink); }}
.pbar .val {{ font-weight: 700; text-align: right; font-variant-numeric: tabular-nums; min-width: 56px; }}
.pbar .na {{ color: var(--slate); font-size: 14px; }}

/* the one framed object: the chart */
[data-testid="stPlotlyChart"] {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 4px; }}

/* controls */
button[kind="primary"], .stButton > button[kind="primary"] {{ background: var(--damson); border-color: var(--damson); color: #fff;
  border-radius: 3px; min-height: 40px; font-weight: 700; }}
.stButton > button {{ border-radius: 3px; min-height: 40px; }}
:focus-visible {{ outline: 2px solid var(--damson) !important; outline-offset: 2px; }}

@media (max-width: 900px) {{
  .block-container {{ padding: 20px; }}
  h1, .decision {{ font-size: 24px; line-height: 29px; }}
  .pbar {{ grid-template-columns: 130px 1fr 52px; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
"""


def inject() -> None:
    """Add the stylesheet to the page (call once per script run)."""
    import streamlit as st

    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
