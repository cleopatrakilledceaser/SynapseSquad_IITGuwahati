"""Streamlit pages of the GlucoTwin clinician dashboard (Part 9, docs/DESIGN.md, D-058, D-059).

Each public function is one page. Pages only read the offline bundle ``app/demo``; nothing is
computed from raw data. Language is decision support, never diagnosis. A fixed footer with the
disclaimer closes every page. Missing numbers are written "not available".
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from src.dash import bundle as B
from src.dash import agp, charts, ref, units
from src.dash import text as T
from src.dash import whatif as W
from src.dash.bundle import HORIZONS, Bundle, PatientData

PLOT_CFG = {"displayModeBar": False, "staticPlot": False}
LIMITS_TEXT = [
    "It cannot tell you what the patient will actually do or eat, only what the models predict "
    "for the change you enter.",
    "It does not know about insulin or other medication doses, illness, stress or sleep.",
    "Carbohydrate effects are understated by the learned models (about half of the observed rise "
    "in matched real breakfasts); treat the size of any change as a lower bound.",
    "A walk can only be added 15 min after a meal for 30 min, the one setting that was tested.",
    "The twin is a simplified physiological model; its walk effect is learned from the data or "
    "assumed (D-052), and the two can differ.",
]


# --------------------------------------------------------------------------- helpers


def _wide(fn: Any, *args: Any, **kw: Any) -> None:
    """Call a Streamlit element at full width on both old and new Streamlit versions."""
    try:
        fn(*args, width="stretch", **kw)
    except Exception:  # noqa: BLE001
        fn(*args, use_container_width=True, **kw)


def plot(fig: Any) -> None:
    """Show a Plotly figure without its toolbar."""
    _wide(st.plotly_chart, fig, config=PLOT_CFG)


def table(df: pd.DataFrame) -> None:
    """Static-looking table, no index column (glucose numbers follow the unit switch)."""
    if units.unit() != units.MGDL:
        df = df.apply(lambda col: col.map(units.localize) if col.dtype == object else col)
    _wide(st.dataframe, df, hide_index=True)


def html_block(s: str) -> None:
    """Raw HTML block (all dynamic text is escaped by the caller)."""
    st.markdown(units.localize(s), unsafe_allow_html=True)


def esc(s: Any) -> str:
    """HTML-escape any text that comes from data."""
    return html.escape(str(s))


@st.cache_resource(show_spinner=False)
def get_bundle() -> Bundle:
    """The app bundle, read once per server process."""
    return B.load_bundle()


@st.cache_resource(show_spinner="Loading models")
def get_forms() -> dict[str, Any]:
    """Shipped fold models (no LightGBM needed)."""
    from src.appdata import rt

    return rt.load_forms(B.DEMO / "models")


@st.cache_resource(show_spinner=False)
def get_state(pid: str) -> Any:
    """Live what-if state of a patient."""
    b = get_bundle()
    return W.patient_state(b.patient(pid), b.context, get_forms()["mono"].cols)


def sync_units() -> None:
    """Apply the unit chosen in the sidebar (also inside fragments, which rerun on their own)."""
    units.set_unit(st.session_state.get("unit", units.MGDL))


def bundle_or_stop() -> Bundle:
    """The bundle, or a plain instruction and a stop if it has not been built."""
    sync_units()
    try:
        return get_bundle()
    except B.BundleMissing as e:
        st.markdown("# GlucoTwin")
        st.error(str(e))
        footer()
        st.stop()


def display_cfg(b: Bundle) -> dict[str, Any]:
    """Display settings stored with the bundle (configs/app.yaml)."""
    return b.manifest["display"]


def name_of(meta: dict[str, Any]) -> str:
    """'Patient 004' / 'Synthetic patient'."""
    return str(meta["label"]).replace("cgmacros_", "")


PAGES: dict[str, Any] = {}  # filled by app/main.py: url_path -> st.Page (for switch_page)


def header(title: str, sub: str = "", eyebrow: str = "") -> None:
    """Page title block."""
    eb = f'<div class="eyebrow">{esc(eyebrow)}</div>' if eyebrow else ""
    sb = f'<div class="sub">{esc(sub)}</div>' if sub else ""
    html_block(f'<div class="pagehead">{eb}<h1>{esc(title)}</h1>{sb}</div>')


def stamp(ts: Any) -> str:
    """'Mon 09:55'."""
    return pd.Timestamp(ts).strftime("%a %H:%M")


def risk_chip(word: str) -> str:
    """Risk word as a chip (a dot colour plus the word, never colour alone)."""
    cls = {"low": "low", "moderate": "moderate", "high": "high", "very high": "very"}.get(
        word, "na"
    )
    text = f"{word.capitalize()} risk" if word != T.NA else "Risk not available"
    return f'<span class="chip {cls}"><span class="dot"></span>{esc(text)}</span>'


def kpi_html(row: dict[str, Any], cfg: dict[str, Any]) -> str:
    """Three tiles: glucose now, forecast at +60, chance of the headline event."""
    d = T.decision(row, cfg)
    now, p60 = row.get("cgm_now"), row.get("pred|60")
    trend = T.trend_of(row)
    ev = d["event"]
    r = next(x for x in T.readouts(row, 60) if x["event"] == ev)
    if r["p"] is not None:
        third = f'{100 * r["p"]:.0f}<small>%</small>'
        sub = risk_chip(d["risk_word"])
    else:
        third = '<span style="font-size:20px">' + esc("now" if r["note"] else T.NA) + "</span>"
        sub = esc(r["note"] or T.NA)
    interval = f'80% interval {T.mgdl(row.get("lo80|60"))} to {T.mgdl(row.get("hi80|60"))}'
    return (
        '<div class="kpis">'
        f'<div class="kpi"><div class="k">Glucose now</div><div class="v">{esc(T.mgdl_num(now))}<small>{units.unit()}</small></div>'
        f'<div class="s">{esc(trend.capitalize() if trend != T.NA else "Trend not available")}</div></div>'
        f'<div class="kpi"><div class="k">In 60 minutes</div><div class="v">{esc(T.mgdl_num(p60))}<small>{units.unit()}</small></div>'
        f'<div class="s">{esc(interval)}</div></div>'
        f'<div class="kpi"><div class="k">Chance {esc(T.EVENT_TEXT[ev][2])}</div><div class="v">{third}</div>'
        f'<div class="s">{sub}</div></div></div>'
    )


def footer() -> None:
    """Disclaimer on every page."""
    html_block(f'<p class="disclaimer">{esc(T.DISCLAIMER)}</p>')


def topline(b: Bundle, key: str, only_real: bool = False) -> str:
    """Patient selector with the synthetic label; the choice follows the user across pages."""
    ids = [p for p in b.ids() if not (only_real and b.patients[p].get("synthetic"))]
    cur = st.session_state.get("sel_pid")
    if cur not in ids:
        cur = ids[0]
    pid = st.selectbox("Patient", ids, index=ids.index(cur), key=f"pid_{key}",
                       format_func=lambda p: name_of(b.patients[p]) + (" (synthetic)" if b.patients[p].get("synthetic") else ""))  # fmt: skip
    st.session_state["sel_pid"] = pid
    if b.patients[pid].get("synthetic"):
        html_block(
            '<p class="topline">Synthetic patient<span class="synthetic-tag">SYNTHETIC</span> '
            "A simulated record with an assumed profile. Not a real person.</p>"
        )
    return pid


def anchor_index(pid: str, n: int) -> int:
    """Current replay anchor of a patient (shared by Overview, Live twin and Export)."""
    i = int(st.session_state.get(f"idx_{pid}", 0))
    return max(0, min(i, n - 1))


def history(p: PatientData) -> pd.DataFrame:
    """Real CGM readings of the patient's period."""
    g = p.grid
    if "glucose_is_real" in g.columns:
        g = g[g["glucose_is_real"].astype(bool)]
    return g[["t", "glucose"]]


def forecast_dict(row: pd.Series, clamp: list[float]) -> dict[int, dict[str, float | None]]:
    """Forecast of an anchor as horizon -> mean, lo, hi (clipped to the physiological range)."""
    lo_c, hi_c = clamp
    out = {}
    for h in HORIZONS:
        out[h] = {
            k: float(min(max(row[c], lo_c), hi_c))
            for k, c in (("mean", f"pred|{h}"), ("lo", f"lo80|{h}"), ("hi", f"hi80|{h}"))
        }
    return out


def prob_bars(row: dict[str, Any], tick: dict[str, float | None]) -> str:
    """Three horizontal probability bars (0 to 100 %) with the alert threshold as a tick."""
    out = []
    for r in T.readouts(row, 60):
        label = f"Glucose {esc(r['label'])}"
        if r["p"] is None:
            out.append(
                f'<div class="pbar"><span>{label}</span><span class="na">{esc(r["note"])}</span><span></span></div>'
            )
            continue
        t = tick.get(r["event"])
        tk = f'<div class="tick" style="left:{100 * t:.1f}%"></div>' if t is not None else ""
        out.append(f'<div class="pbar"><span>{label}</span><div class="track"><div class="fill" style="width:{100 * r["p"]:.1f}%"></div>{tk}</div>'
                   f'<span class="val">{100 * r["p"]:.0f}%</span></div>')  # fmt: skip
    return "".join(out)


def alert_ticks(b: Bundle) -> dict[str, float | None]:
    """Alert thresholds mapped to the calibrated-probability scale (Platt map of the same event)."""
    import math

    out: dict[str, float | None] = {}
    theta = b.calib.get("theta", {})
    for e, th in theta.items():
        mp = b.calib["platt"].get(f"{e}|60")
        if th is None or mp is None:
            out[e] = None
            continue
        s = min(max(float(th), 1e-4), 1 - 1e-4)
        z = mp["coef"] * math.log(s / (1 - s)) + mp["intercept"]
        out[e] = 1 / (1 + math.exp(-z))
    return out


def conf_html(conf: dict[str, Any]) -> str:
    """Confidence chip: the word plus a bar count."""
    bars = "".join(f'<span class="{"on" if i < conf["bars"] else ""}"></span>' for i in range(3))
    return f'<span class="chip">Confidence: {esc(conf["label"])}&nbsp;<span class="bars">{bars}</span></span>'


def why_drawer(p: PatientData, row: pd.Series) -> None:
    """Closed drawer with the template sentences that explain the forecast."""
    with st.expander("Why this forecast"):
        hz = st.radio(
            "Horizon",
            HORIZONS,
            index=1,
            horizontal=True,
            format_func=lambda h: f"+{h} min",
            key=f"why_{p.pid}",
        )
        for s in T.explain_sentences(p.explain, row["t"], int(hz)):
            st.markdown(units.localize(f"- {s}"))
        st.caption(
            "What moved the model's forecast, not a physiological cause. Time of day and HbA1c can appear when nothing else stands out."
        )


def decision_block(b: Bundle, p: PatientData, row: pd.Series) -> None:
    """Decision sentence, forecast line, confidence, uncertain banner and the 'why' drawer."""
    decision_block_light(b, p, row)
    st.markdown(units.localize(T.forecast_line(row.to_dict(), 60)))
    why_drawer(p, row)


# --------------------------------------------------------------------------- overview


def _select(pid: str) -> None:
    st.session_state["sel_pid"] = pid


def overview() -> None:
    """Patient cards on the left, the selected patient's decision, tiles and profile on the right."""
    b = bundle_or_stop()
    cfg = display_cfg(b)
    header(
        "Patients",
        "Replayed days of recorded data. Select a patient to see the latest forecast.",
        "Overview",
    )
    ids = b.ids()
    cur = st.session_state.get("sel_pid", ids[0])
    pid = cur if cur in ids else ids[0]
    left, right = st.columns([5, 7], gap="large")
    with left:
        for q in ids:
            qp = b.patient(q)
            qrow = qp.replay.iloc[anchor_index(q, len(qp.replay))].to_dict()
            tag = '<span class="synthetic-tag">SYNTHETIC</span>' if qp.synthetic else ""
            sel = '<span class="selected-tag">Selected</span>' if q == pid else ""
            with st.container(border=True):
                html_block(
                    f'<div class="pcard"><div><span class="name">{esc(name_of(qp.meta))}</span>{tag} {sel}</div>'
                    f'<div class="big">{esc(T.mgdl_num(qrow.get("cgm_now")))} <small>{units.unit()}</small></div>'
                    f'<div class="st">{esc(T.short_status(qrow, cfg))}</div></div>'
                )
                st.button("Selected" if q == pid else "View", key=f"sel_{q}", on_click=_select, args=(q,),
                          disabled=q == pid, type="secondary")  # fmt: skip
    p = b.patient(pid)
    row = p.replay.iloc[anchor_index(pid, len(p.replay))]
    with right:
        with st.container(border=True):
            tag = '<span class="synthetic-tag">SYNTHETIC</span>' if p.synthetic else ""
            html_block(
                f'<p class="topline">{esc(name_of(p.meta))}{tag} &middot; forecast made at {esc(stamp(row["t"]))}'
                + (" &middot; a simulated record with an assumed profile" if p.synthetic else "")
                + "</p>"
            )
            decision_block_light(b, p, row)
            why_drawer(p, row)
            if "live" in PAGES and st.button("Open the live twin", type="primary", key="open_live"):
                st.switch_page(PAGES["live"])
        with st.container(border=True):
            st.markdown("#### Profile")
            chips = "".join(
                f'<span class="chip">{esc(k)}: {esc(v)}</span>' for k, v in T.profile_lines(p.meta)
            )
            html_block(f'<div class="chips">{chips}</div>')
            st.markdown("#### Data quality")
            st.markdown(T.quality_banner(row.to_dict()))
    footer()


# --------------------------------------------------------------------------- live twin


def live_twin() -> None:
    """The hero forecast chart with replay, event readouts, alerts and confidence."""
    b = bundle_or_stop()
    cfg = display_cfg(b)
    header(
        "Live twin",
        "Replay of recorded data: move through the day or press Play. Not a live feed.",
        "Forecast",
    )
    speeds = list(cfg["replay_speeds"])
    with st.container(border=True):
        c0, c1, c2, c3 = st.columns([4, 1.2, 1.2, 2.2], vertical_alignment="bottom")
        with c0:
            pid = topline(b, "live")
        if c1.button("Play", key="play_btn", type="primary"):
            st.session_state["playing"] = True
        if c2.button("Pause", key="pause_btn"):
            st.session_state["playing"] = False
        speed = c3.selectbox("Speed", speeds, format_func=lambda s: f"{s}x", key="speed")
    p = b.patient(pid)
    n = len(p.replay)
    playing = bool(st.session_state.get("playing", False))
    run_every = (1.0 / float(speed) * float(cfg.get("replay_tick_s", 1.0))) if playing else None
    ticks, hist = alert_ticks(b), history(p)
    ev_stats = b.stats
    rate = int(b.calib.get("fa_per_day", 2))

    def view() -> None:
        sync_units()
        key = f"idx_{pid}"
        i = anchor_index(pid, n)
        if st.session_state.get("playing", False) and i < n - 1:
            i += 1
        elif i >= n - 1:
            st.session_state["playing"] = False
        st.session_state[key] = i
        i = st.select_slider("Time of forecast", options=list(range(n)), key=key,
                             format_func=lambda k: stamp(p.replay.iloc[k]["t"]),
                             help="Each step is 15 minutes of the replayed record.")  # fmt: skip
        row = p.replay.iloc[i]
        r = row.to_dict()
        with st.container(border=True):
            html_block(f'<p class="topline">Forecast made at {esc(stamp(row["t"]))}</p>')
            decision_block_light(b, p, row)
            fig = charts.hero_figure(
                hist,
                pd.Timestamp(row["t"]),
                float(row["cgm_now"]),
                forecast_dict(row, cfg["clamp"]),
                p.meals,
                cfg,
            )
            plot(fig)
            st.caption(
                "Lines connect the three forecast horizons; the model does not forecast between them. Shaded band: 80% interval."
            )
        a, c = st.columns([6, 6], gap="medium")
        with a, st.container(border=True):
            st.markdown("#### Forecast")
            rows = []
            for h in HORIZONS:
                rows.append({"Horizon": f"+{h} min", "Median": T.mgdl(r.get(f"pred|{h}")),
                             "80% interval": f"{T.mgdl(r.get(f'lo80|{h}'))} to {T.mgdl(r.get(f'hi80|{h}'))}",
                             "Width": T.mgdl(r.get(f"hi80|{h}") - r.get(f"lo80|{h}"))})  # fmt: skip
            table(pd.DataFrame(rows))
        with c, st.container(border=True):
            st.markdown("#### Risk within 60 minutes")
            html_block(prob_bars(r, ticks))
            st.caption("Calibrated probabilities. The tick marks where the alert switches on.")
        with st.container(border=True):
            _alert_lines(r, ev_stats, rate)
        why_drawer(p, row)

    if hasattr(st, "fragment"):
        st.fragment(run_every=run_every)(view)()
    else:
        view()
    footer()


def decision_block_light(b: Bundle, p: PatientData, row: pd.Series, kpis: bool = True) -> None:
    """Decision sentence, chips, number tiles and the uncertain banner."""
    cfg, r = display_cfg(b), row.to_dict()
    d = T.decision(r, cfg)
    conf = T.confidence(r, cfg)
    html_block(f'<div class="decision">{esc(d["sentence"])}</div>')
    html_block(f'<div class="chips">{risk_chip(d["risk_word"])}{conf_html(conf)}</div>')
    banner = T.uncertain_banner(conf)
    if banner:
        html_block(f'<div class="banner">{esc(banner)}</div>')
    if kpis:
        html_block(kpi_html(r, cfg))


def _alert_lines(r: dict[str, Any], stats: dict[str, Any], rate: int) -> None:
    on = T.alert_state(r)["on"]
    st.markdown("#### Alerts")
    for e, name in (("hyper180", "High-glucose alert"), ("hypo70", "Low-glucose alert")):
        ev = T.alert_evidence(stats, e, rate)
        chip = (
            '<span class="chip on"><span class="dot"></span>ON</span>'
            if e in on
            else '<span class="chip"><span class="dot"></span>Off</span>'
        )
        det = ""
        if ev and ev["sensitivity"] is not None:
            lim = " Limited evidence: few episodes." if (ev["episodes"] or 0) < 20 else ""
            det = (f"In testing at this setting the alert caught {100 * ev['sensitivity']:.0f} % of {ev['episodes']} episodes "
                   f"with a median warning of {ev['lead_median_min']:.0f} min and about {rate} false alarms a day.{lim}")  # fmt: skip
        html_block(
            f'<div class="alertrow"><div class="t">{esc(name)}&nbsp;&nbsp;{chip}</div><div class="d">{esc(det)}</div></div>'
        )
    st.caption(
        "Alerts here use only readings up to the forecast time (a display rule that differs slightly from the Part 6 test)."
    )


# --------------------------------------------------------------------------- glucose report


def _report_tile(k: str, v: str, small: str, sub: str) -> str:
    return (
        f'<div class="kpi"><div class="k">{esc(k)}</div><div class="v">{esc(v)}<small>{esc(small)}</small></div>'
        f'<div class="s">{esc(sub)}</div></div>'
    )


def glucose_report() -> None:
    """Standard CGM report: time in ranges against consensus targets, GMI, variability, AGP."""
    b = bundle_or_stop()
    cfg = display_cfg(b)
    header(
        "Glucose report",
        "The summary clinicians already read from a CGM record: time in range, GMI and the daily profile.",
        "Report",
    )
    with st.container(border=True):
        pid = topline(b, "report")
    p = b.patient(pid)
    df = history(p)
    r = agp.report(df, cfg["zones"])
    sh, cov = r["shares"], r["coverage"]
    html_block(
        '<div class="kpis" style="grid-template-columns:repeat(4,minmax(0,1fr))">'
        + _report_tile("Mean glucose", units.num(r["mean"]), units.unit(), f"{r['n']} readings")
        + _report_tile("GMI", f"{r['gmi']:.1f}", "%", "Estimated from mean glucose; not an HbA1c result")
        + _report_tile("Variability (CV)", f"{r['cv']:.0f}", "%", f"Consensus target at most {agp.CV_TARGET:.0f}%")
        + _report_tile("Time in range", f"{sh['range']:.0f}", "%", "Target above 70%")
        + "</div>"
    )
    with st.container(border=True):
        st.markdown("#### Time in ranges")
        plot(charts.tir_figure(sh))
        rows = []
        for key, label, _, limit in agp.TARGETS:
            direction = "above" if key == "range" else "below"
            rows.append({"Range": label, "Your readings": f"{sh[key]:.1f}%",
                         "Consensus target": f"{direction} {limit:g}%",
                         "Meets target": "Yes" if agp.meets(key, sh[key]) else "No"})  # fmt: skip
        table(pd.DataFrame(rows))
        st.caption(
            "Consensus targets for people with diabetes in general (Battelino et al., Diabetes Care 2019). "
            "They are population values; a clinician sets the goal for each patient."
        )
    with st.container(border=True):
        st.markdown("#### Daily profile")
        plot(charts.agp_figure(agp.agp_profile(df), cfg))
        st.caption(
            "All days laid over one 24-hour clock. Dark line: median. Darker band: middle half of readings. "
            "Lighter band: 90% of readings."
        )
    short = cov["days"] < agp.MIN_DAYS or cov["fraction"] < 0.7
    st.info(
        f"This record covers {cov['days']:.1f} days with {100 * cov['fraction']:.0f}% of expected readings. "
        f"The consensus asks for {agp.MIN_DAYS} days and at least 70% of readings"
        + (", so treat these numbers as a rough picture." if short else ".")
    )
    footer()


# --------------------------------------------------------------------------- what-if lab


def what_if_lab() -> None:
    """Controls, the chart with a ghost trajectory, the comparison, the fixed note and the evidence."""
    b = bundle_or_stop()
    cfg = display_cfg(b)
    header(
        "What-if lab",
        "Replace a logged meal with another one, optionally add a walk, and see how the forecast changes.",
        "Scenarios",
    )
    pid = topline(b, "whatif")
    p = b.patient(pid)
    if not p.windows:
        st.info(
            "This patient has no meal windows for what-ifs (the synthetic patient has no logged meals with "
            "macros and no fitted twin). Choose a CGMacros patient above."
        )
        html_block(f'<div class="note">{esc(T.WHATIF_NOTE)}</div>')
        footer()
        return
    ctl_col, out = st.columns([5, 7], gap="large")
    with ctl_col, st.container(border=True):
        wi = st.selectbox(
            "Window",
            range(len(p.windows)),
            format_func=lambda i: W.window_label(p, p.windows[i]),
            key="wi_window",
        )
        win = p.windows[wi]
        base = W.logged_scenario(p, win)
        src = st.radio("Meal", ["Logged meal", "Indian meal preset", "Custom grams"], key="wi_src")
        lim = {"carbs_g": 150, "protein_g": 80, "fat_g": 80, "fiber_g": 40}
        if src == "Indian meal preset":
            pre = W.presets()
            key = st.selectbox(
                "Preset",
                list(pre),
                format_func=lambda k: f"{pre[k]['label']} ({pre[k]['portion']})",
                key="wi_preset",
            )
            m = pre[key]["macros"]
            st.caption(f"{m['carbs_g']:.0f} g carbohydrate, {m['protein_g']:.0f} g protein, {m['fat_g']:.0f} g fat, {m['fiber_g']:.0f} g fibre per portion. "
                       "Our own assumed recipe with IFCT 2017 values, not clinical or dietary advice.")  # fmt: skip
            chosen = (
                float(m["carbs_g"]),
                float(m["protein_g"]),
                float(m["fat_g"]),
                float(m["fiber_g"]),
            )
        elif src == "Custom grams":
            chosen = tuple(float(st.number_input(lbl, 0.0, float(lim[c]), float(round(getattr(base, c))), 5.0, key=f"wi_{c}"))
                           for c, lbl in (("carbs_g", "Carbohydrate (g)"), ("protein_g", "Protein (g)"), ("fat_g", "Fat (g)"), ("fiber_g", "Fibre (g)")))  # fmt: skip
        else:
            chosen = (base.carbs_g, base.protein_g, base.fat_g, base.fiber_g)
        walk = st.toggle("Add a 30-minute walk, starting 15 min after the meal", key="wi_walk")
        model = st.selectbox(
            "Model", list(W.MODEL_LABELS), format_func=lambda k: W.MODEL_LABELS[k], key="wi_model"
        )
        scen = W.Scenario(*chosen, walk=bool(walk))
        sig = (pid, wi, scen, model)
        try:
            run = st.button("Run what-if", type="primary", key="wi_run", width="stretch")
        except TypeError:  # older Streamlit
            run = st.button("Run what-if", type="primary", key="wi_run", use_container_width=True)
    with out:
        if run:
            with st.spinner("Running"):
                st.session_state["wi_res"] = (sig, _run_whatif(b, p, win, base, scen, model, cfg))
        got = st.session_state.get("wi_res")
        if not got or got[0] != sig:
            with st.container(border=True):
                st.markdown("#### Your scenario will appear here")
                st.markdown(
                    "Choose a meal on the left and press **Run what-if**."
                    if not got
                    else "The inputs changed. Press **Run what-if** to update."
                )
        else:
            res = got[1]
            with st.container(border=True):
                _show_whatif(b, p, win, res, cfg, model)
    st.markdown("## Plausibility checks")
    st.markdown(
        "Pass rates of physiological checks on patients the models never saw (Part 7). A check that a model passes by construction says nothing about the data."
    )
    with st.expander("Show the checks", expanded=False):
        table(ref.plausibility_table(b.stats))
        st.markdown(
            "Median predicted peak rise per gram of carbohydrate, against the range reported in the literature:"
        )
        table(ref.dose_table(b.stats))
    with st.expander("What this tool cannot tell you"):
        for s in LIMITS_TEXT:
            st.markdown(units.localize(f"- {s}"))
    footer()


def _run_whatif(
    b: Bundle,
    p: PatientData,
    win: dict[str, Any],
    base: W.Scenario,
    scen: W.Scenario,
    model: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    clamp = tuple(cfg["clamp"])
    if model in W.ML_FORMS:
        return W.ml_run(get_forms()[model], get_state(p.pid), win, scen, clamp)
    res = W.twin_run(p, win, model, scen, base, b.context, clamp)
    return res if res is not None else {"kind": "none"}


def _show_whatif(
    b: Bundle,
    p: PatientData,
    win: dict[str, Any],
    res: dict[str, Any],
    cfg: dict[str, Any],
    model: str,
) -> None:
    if res.get("kind") == "none":
        st.warning("The twin is not available for this patient: no twin was fitted.")
        return
    row = p.replay[p.replay["t"] == pd.Timestamp(win["t"])].iloc[0]
    ghost: dict[str, Any] = {"fc": res["new"], "traj": None}
    if res["kind"] == "twin":
        t0 = pd.Timestamp(win["t"])
        ghost["traj"] = (
            [t0 + pd.Timedelta(minutes=res["dt"] * k) for k in range(len(res["new_traj"]))],
            list(res["new_traj"]),
        )
    fig = charts.hero_figure(
        history(p),
        pd.Timestamp(win["t"]),
        float(row["cgm_now"]),
        res["base"],
        p.meals,
        cfg,
        ghost=ghost,
    )
    plot(fig)
    st.caption(
        "Solid: baseline (the logged meal, no walk). Dashed: what-if. Both are model estimates."
    )
    table(W.comparison_table(res))
    html_block(f'<div class="note">{esc(T.WHATIF_NOTE)}</div>')
    if res["kind"] == "twin":
        st.markdown(
            "Twin values are read from a grid computed in advance and interpolated between grid points"
            + (
                " (this result is interpolated)."
                if res["interpolated"]
                else " (this result is on a grid point)."
            )
        )
        if res.get("clamped_inputs"):
            st.markdown(
                "Inputs beyond the grid (150 g carbohydrate or the highest slowing ratio) were set to the grid edge."
            )
        if "assumed" in model:
            st.markdown(
                "The walk effect of this twin is ASSUMED (a 20 % smaller area after a meal), not learned from the data."
            )
    if model == "mono":
        st.caption(
            "The monotone model cannot predict a higher peak after less carbohydrate or after a walk, by construction."
        )


# --------------------------------------------------------------------------- reference pages


def twin_state() -> None:
    """Only the basal glucose of the twin is shown: it is the only identifiable parameter."""
    b = bundle_or_stop()
    header("Twin state", "What the digital twin can and cannot tell you.", "Model")
    pid = topline(b, "twin")
    p = b.patient(pid)
    st.markdown(
        "The twin is a small physiological model fitted to a patient's first days of data. "
        "Only one of its parameters can be read from the data reliably: the basal glucose Gb."
    )
    gb = p.meta.get("twin_gb_mgdl")
    sim = p.meta.get("gb_simulated_true_mgdl")
    if gb is not None:
        html_block(
            f'<div class="serif num" style="font-size:40px;line-height:44px;font-weight:700">{gb:.0f} mg/dL</div>'
        )
        st.markdown(
            "Basal glucose Gb of this patient's twin: the level the twin settles at without meals."
        )
    elif sim is not None:
        html_block(
            f'<div class="serif num" style="font-size:40px;line-height:44px;font-weight:700">{sim:.0f} mg/dL</div>'
        )
        st.markdown(
            "Basal glucose Gb that the simulator used for this SYNTHETIC patient (a simulated value, not a fit). "
            "No twin was fitted to this record."
        )
    else:
        st.markdown(f"Basal glucose Gb: {T.NA}. No twin was fitted for this patient.")
    st.markdown("## What is not shown")
    st.markdown(
        "The twin's other parameters (insulin sensitivity, glucose effectiveness, absorption rate, "
        "meal fraction and secretion) are not shown because they cannot be identified from this data: "
        "in a repeat-fit test Gb agreed between two halves of a patient's record (correlation 0.62), "
        "the others did not (0.14 to 0.24). The twin also adds little to the machine-learning forecast "
        "(about 0.5 mg/dL after three days of data)."
    )
    footer()


def personalization() -> None:
    """Cold-start curve and the honest paragraph."""
    b = bundle_or_stop()
    header(
        "Personalization",
        "How much does the forecast improve when it learns from a patient's own first days of data?",
        "Evidence",
    )
    hz = st.radio(
        "Horizon",
        ["30", "60", "120"],
        index=1,
        horizontal=True,
        format_func=lambda h: f"+{h} min",
        key="pers_h",
    )
    left, right = st.columns([7, 5], gap="large")
    cur = ref.personalization_curve(b.stats, hz)
    with left:
        if cur.empty:
            st.info(f"The curve is {T.NA}.")
        else:
            plot(charts.curve_figure(cur, f"Mean absolute error at +{hz} min (mg/dL)"))
            st.caption(
                "Mean absolute error with 95% intervals over patients; 44 patients with enough data."
            )
    gain = ref.personalization_gain(b.stats, hz)
    hs = ref.head_start(b.stats)
    with right:
        if gain:
            st.markdown(
                "Fine-tuning on the patient's own first days changed the error at +"
                + hz
                + " min by "
                + ", ".join(
                    f"{v:+.1f} mg/dL after {int(k)} day{'s' if k > 1 else ''}"
                    for k, v in gain.items()
                )
                + " (negative means a lower error)."
            )
        st.markdown(
            "About 1 mg/dL after 3 to 5 days is the realistic size of the gain. Heart rate and activity data add nothing measurable."
        )
        if hs["mgdl"] is not None:
            ci = hs["ci"] or [None, None]
            chg = ref.ci_text(
                -hs["mgdl"], [-ci[1], -ci[0]] if ci[0] is not None else None, 2, " mg/dL"
            )
            st.markdown("The clinical profile did not give the models a head start. With no patient data, adding the profile changed the "
                        f"error by {chg} (positive means worse; the interval includes zero). The profile only helps the twin, which is not competitive.")  # fmt: skip
    footer()


def input_tiers() -> None:
    """What happens to accuracy when inputs are missing."""
    b = bundle_or_stop()
    header(
        "Input tiers",
        "Accuracy when some inputs are missing, from the Part 3 tests on 45 CGMacros patients (mean absolute error, mg/dL, with 95% intervals).",
        "Evidence",
    )
    table(ref.tier_table(b.stats))
    st.markdown(
        "Fingerstick values in these tests are real sensor readings taken at fixed clock times (4 or 2 a day), "
        "so the fingerstick tiers are optimistic. Without a CGM the models are worse than simply carrying the last CGM value "
        "forward at +30 and +60 min, and the clinical profile alone does not rescue them."
    )
    footer()


def data_quality() -> None:
    """The data-quality gate, an example and its honest limits."""
    b = bundle_or_stop()
    cfg = display_cfg(b)
    c = cfg["confidence"]
    header(
        "Data quality",
        "When a forecast is labelled uncertain, and what the quality score cannot see.",
        "Safety",
    )
    st.markdown(
        "A forecast is labelled uncertain, and the alert is replaced by a warning, when any of these holds "
        f"(a design rule, not a validated threshold): the interval is more than {c['wide_ratio']:g} times the usual width, "
        f"the data-quality score is below {c['dq_uncertain_below']:.2f}, or the last sensor reading is older than {c['stale_min']} minutes."
    )
    worst: tuple[float, str, pd.Series] | None = None
    for pid in b.ids():
        p = b.patient(pid)
        i = int(p.replay["dq"].astype(float).idxmin())
        v = float(p.replay.loc[i, "dq"])
        if worst is None or v < worst[0]:
            worst = (v, pid, p.replay.loc[i])
    if worst is not None:
        v, pid, r = worst
        st.markdown(f"Lowest score in the demo data: {v:.2f} for {name_of(b.patients[pid])} at {T.clock(r['t'])} "
                    f"(completeness {r['dq_complete']:.2f}, recency {r['dq_recency']:.2f}, plausibility {r['dq_plaus']:.2f}, "
                    f"rate of change {r['dq_roc']:.2f}).")  # fmt: skip
    st.markdown("## What the score cannot see")
    st.markdown(
        T.LIMIT
        + " In the Part 6 fault tests, a constant offset of 20 mg/dL or a drift of 20 mg/dL per day left the quality "
        "score near 0.99 while the 80% intervals covered far fewer than 80 % of the true values:"
    )
    table(ref.robustness_table(b.stats))
    st.caption(
        "Split-conformal intervals, +60 min, CGMacros test patients. Coverage under faults is the main risk of the system."
    )
    footer()


FIGURES = [
    (
        "fusion_ablation",
        "Which inputs help",
        "Removing the CGM history hurts most. Adding wearable data or the clinical profile did not improve the forecasts.",
    ),
    (
        "personalize_cold_start_lgbm",
        "Learning from a patient's first days",
        "Fine-tuning on the patient's own data lowers the error by about 1 mg/dL after 3 to 5 days.",
    ),
    (
        "unc_coverage",
        "Do 80% intervals contain the truth?",
        "Pooled coverage is close to 80 %, but it is not equal for all groups, and one method undercovered on the second cohort.",
    ),
    (
        "unc_alarm_froc",
        "Alert burden",
        "How many episodes each alert catches for a given number of false alarms a day.",
    ),
    (
        "whatif_plausibility",
        "What-if plausibility checks",
        "Pass rates of physiological checks. The monotone model passes the carbohydrate and walk checks by construction.",
    ),
    (
        "whatif_matched_meals",
        "Matched real breakfasts",
        "Predicted against observed peak difference in 87 matched pairs. The models understate the rise.",
    ),
]


def evidence() -> None:
    """Six figures with plain captions, the stored tables and the limitations."""
    b = bundle_or_stop()
    header(
        "Evidence",
        "Results on patients the models never saw. Every interval is a 95% interval over patients, not over rows.",
        "Evidence",
    )
    cols = st.columns(2, gap="large")
    for i, (name, title, cap) in enumerate(FIGURES):
        with cols[i % 2]:
            st.markdown(f"### {title}")
            pth = b.figure_path(name)
            if pth.exists():
                _wide(st.image, str(pth))
            st.markdown(cap)
    st.markdown("## Alerts at about two false alarms a day")
    table(ref.alarm_table(b.stats, str(int(b.calib.get("fa_per_day", 2)))))
    for s in ref.auroc_text(b.stats):
        st.markdown(f"- {s}")
    st.markdown("## Intervals")
    st.markdown(ref.coverage_text(b.stats))
    st.markdown("## Matched real breakfasts")
    table(ref.matched_summary(b.stats))
    st.markdown("## Limitations")
    for s in ref.LIMITATIONS:
        st.markdown(f"- {s}")
    footer()


def export() -> None:
    """One-page visit summary as a PDF."""
    b = bundle_or_stop()
    cfg = display_cfg(b)
    header(
        "Export",
        "A one-page summary of the forecast at the time currently selected on the Live twin page.",
        "Report",
    )
    pid = topline(b, "export")
    p = b.patient(pid)
    i = anchor_index(pid, len(p.replay))
    row = p.replay.iloc[i]
    st.markdown(f"Forecast time: {T.clock(row['t'])}. Change it on the Live twin page.")
    decision_block_light(b, p, row)
    if st.button("Prepare visit summary", type="primary", key="exp_btn"):
        from src.dash import pdfexport

        st.session_state["pdf"] = (pid, i, pdfexport.visit_summary_pdf(
            p.meta | {"label": name_of(p.meta)}, row.to_dict(), history(p), p.meals, T.explain_sentences(p.explain, row["t"], 60),
            b.stats, cfg, int(b.calib.get("fa_per_day", 2))))  # fmt: skip
    got = st.session_state.get("pdf")
    if got and got[0] == pid and got[1] == i:
        st.download_button("Export visit summary", got[2], file_name=f"glucotwin_{pid}_{T.clock(row['t']).replace(':', '')}.pdf",
                           mime="application/pdf", key="exp_dl")  # fmt: skip
    footer()
