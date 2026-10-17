"""One-page visit summary PDF (Part 9). matplotlib only, so no extra dependency.

The page repeats what the screen says at one anchor: decision sentence, forecast table, event
probabilities, confidence, data-quality line, the "why" sentences and the disclaimer. A simplified
chart (history, forecast fan, zones) is drawn with matplotlib; the interactive chart stays on
screen. The synthetic patient is labelled synthetic on the page.
"""

from __future__ import annotations

import io
import textwrap
from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.dash import text as T

A4 = (8.27, 11.69)
INK, SLATE, RULE, DAMSON = "#1B2230", "#566174", "#D8DDE5", "#43285E"
ZONE = [  # (low, high, colour, alpha, hatch, label): a hatch and a label on every zone, never colour alone
    (40, 54, "#2B6CB0", 0.30, "xxx", "Very low"), (54, 70, "#2B6CB0", 0.18, "///", "Low"),
    (70, 180, "#9AD3B8", 0.30, None, "In range"), (180, 250, "#E09A12", 0.30, "...", "High"),
    (250, 300, "#8C1D2B", 0.25, "+++", "Very high"),
]  # fmt: skip


def _wrap(s: str, width: int = 92) -> str:
    return "\n".join(textwrap.wrap(s, width=width)) if s else ""


def _chart(
    ax: Any, hist: pd.DataFrame, row: Mapping[str, Any], meals: pd.DataFrame, cfg: Mapping[str, Any]
) -> None:
    now = pd.Timestamp(row["t"])
    x0, x1 = now - pd.Timedelta(hours=cfg["history_hours"]), now + pd.Timedelta(
        minutes=cfg["forecast_min"]
    )
    for lo, hi, c, a, hatch, label in ZONE:
        ax.axhspan(lo, hi, facecolor=c, alpha=a, lw=0)
        if hatch:
            ax.axhspan(lo, hi, facecolor="none", edgecolor=c, hatch=hatch, lw=0, alpha=0.55)
        ax.text(x1, (lo + hi) / 2, label + " ", ha="right", va="center", fontsize=7, color=INK)
    h = hist[(hist["t"] >= x0) & (hist["t"] <= now)]
    ax.plot(h["t"], h["glucose"], color=INK, lw=1.6, label="Glucose history")
    ts = [now] + [now + pd.Timedelta(minutes=m) for m in (30, 60, 120)]
    mean = [row["cgm_now"]] + [row[f"pred|{m}"] for m in (30, 60, 120)]
    lo = [row["cgm_now"]] + [row[f"lo80|{m}"] for m in (30, 60, 120)]
    hi = [row["cgm_now"]] + [row[f"hi80|{m}"] for m in (30, 60, 120)]
    ax.fill_between(ts, lo, hi, color=DAMSON, alpha=0.20, lw=0, label="80% interval")
    ax.plot(ts, mean, color=DAMSON, lw=2.2, marker="o", ms=4, label="Forecast (median)")
    m = meals[(meals["t"] >= x0) & (meals["t"] <= x1)]
    if len(m):
        ax.scatter(
            m["t"],
            [cfg["y_range"][0] + 4] * len(m),
            marker="^",
            color=INK,
            s=28,
            label="Meal",
            zorder=5,
        )
    ax.axvline(now, color=INK, lw=0.8)
    ax.set_xlim(x0, x1)
    ax.set_ylim(*cfg["y_range"])
    ax.set_ylabel("Glucose (mg/dL)", fontsize=9, color=SLATE)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    import matplotlib.dates as mdates

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(fontsize=7, loc="upper left", frameon=False, ncol=4)


def visit_summary_pdf(
    meta: Mapping[str, Any],
    row: Mapping[str, Any],
    hist: pd.DataFrame,
    meals: pd.DataFrame,
    explain_sentences: list[str],
    stats: Mapping[str, Any],
    cfg: Mapping[str, Any],
    alert_rate: int,
) -> bytes:
    """Render the summary of one anchor as a one-page A4 PDF and return its bytes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    d = T.decision(row, cfg)
    conf = T.confidence(row, cfg)
    banner = T.uncertain_banner(conf)
    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor("white")
    y = 0.965
    title = "GlucoTwin visit summary" + ("  (SYNTHETIC patient)" if meta.get("synthetic") else "")
    fig.text(0.08, y, title, fontsize=17, fontweight="bold", color=INK, family="serif", va="top")
    y -= 0.032
    fig.text(0.08, y, f"{meta['label']}   |   forecast made at {pd.Timestamp(row['t']).strftime('%Y-%m-%d %H:%M')}"
             + ("   |   simulated record, assumed profile" if meta.get("synthetic") else ""),
             fontsize=9, color=SLATE, va="top")  # fmt: skip
    y -= 0.035
    fig.text(
        0.08,
        y,
        _wrap(d["sentence"], 56),
        fontsize=14,
        color=INK,
        family="serif",
        va="top",
        fontweight="bold",
    )
    y -= 0.07
    fig.text(0.08, y, T.forecast_line(row, 60), fontsize=10, color=INK, va="top")
    y -= 0.027
    if banner:
        fig.text(0.08, y, _wrap(banner, 95), fontsize=10, color=INK, fontweight="bold", va="top")
        y -= 0.045
    ax = fig.add_axes([0.10, y - 0.30, 0.84, 0.28])
    _chart(ax, hist, row, meals, cfg)
    y -= 0.34
    lines = ["Forecast at three horizons (80% interval from conformal prediction):"]
    for h in (30, 60, 120):
        lines.append(
            f"   +{h:>3} min   {T.mgdl(row.get(f'pred|{h}')):>11}   interval {T.mgdl(row.get(f'lo80|{h}'))} to {T.mgdl(row.get(f'hi80|{h}'))}"
        )
    lines.append("")
    lines.append("Chance within 60 min (calibrated probability):")
    for r in T.readouts(row, 60):
        lines.append(
            f"   Glucose {r['label']:<17} {T.pct(r['p']) if r['p'] is not None else r['note']}"
        )
    st = T.alert_state(row)["on"]
    lines += ["", f"Alerts on now: {', '.join('glucose ' + T.EVENT_TEXT[e][2] for e in st) if st else 'none'}",
              f"Confidence: {conf['label']} (a display rule, not a validated threshold)",
              T.quality_banner(row)]  # fmt: skip
    for e in ("hyper180", "hypo70"):
        ev = T.alert_evidence(stats, e, alert_rate)
        if ev and ev["sensitivity"] is not None:
            lim = " (limited evidence)" if (ev["episodes"] or 0) < 20 else ""
            lines.append(f"Alert testing, glucose {T.EVENT_TEXT[e][2]}: caught {100 * ev['sensitivity']:.0f} % of {ev['episodes']} episodes, "
                         f"median warning {ev['lead_median_min']:.0f} min, about {alert_rate} false alarms a day{lim}.")  # fmt: skip
    if explain_sentences:
        lines += ["", "Why (what moved the +60 min forecast; not a physiological cause):"]
        lines += [f"   {s}" for s in explain_sentences]
    body = "\n".join(_wrap(ln, 100) if not ln.startswith("   ") else ln for ln in lines)
    fig.text(0.08, y, body, fontsize=8.6, color=INK, va="top", family="DejaVu Sans")
    fig.text(0.08, 0.035, _wrap(T.DISCLAIMER, 110), fontsize=8.5, color=INK, va="bottom")
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    return buf.getvalue()
