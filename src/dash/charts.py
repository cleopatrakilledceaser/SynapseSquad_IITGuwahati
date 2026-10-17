"""Plotly figures of the dashboard (Part 9, docs/DESIGN.md sections 2 and 5).

Plotly is imported inside the functions so the rest of ``src.dash`` imports without it. Colours and
patterns follow the design tokens: the forecast wears the brand colour (damson), glucose zones use
four fixed colours that always come with a pattern and a text label.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.dash import units

INK, SLATE, RULE, DAMSON = "#1B2230", "#5A6478", "#E6E2EC", "#43285E"
PAPER, SURFACE = "#F4F2F7", "#FFFFFF"
FONT = "Segoe UI Variable Text, Segoe UI, system-ui, -apple-system, Roboto, Arial, sans-serif"
SERIF = "'Source Serif 4', Georgia, serif"
ZONES = [  # (key, label, pattern colour, pattern shape or None, flat tint under the pattern)
    ("very_low", "Very low, below 54", "#2B6CB0", "x", "rgba(43,108,176,0.10)"),
    ("low", "Low, below 70", "#2B6CB0", "/", "rgba(43,108,176,0.06)"),
    ("range", "In range, 70 to 180", "#9AD3B8", None, "rgba(154,211,184,0.22)"),
    ("high", "High, above 180", "#E09A12", ".", "rgba(224,154,18,0.08)"),
    ("very_high", "Very high, above 250", "#8C1D2B", "+", "rgba(140,29,43,0.07)"),
]

SOLID = {  # flat colour under each zone's pattern in the single-bar view
    "very_low": "rgba(43,108,176,0.55)", "low": "rgba(43,108,176,0.32)",
    "range": "rgba(154,211,184,0.95)", "high": "rgba(224,154,18,0.50)",
    "very_high": "rgba(140,29,43,0.50)",
}  # fmt: skip


def _bounds(cfg: dict[str, Any]) -> list[tuple[float, float]]:
    z, (y0, y1) = cfg["zones"], cfg["y_range"]
    edges = [y0, z["very_low"], z["low"], z["high"], z["very_high"], y1]
    return [(edges[i], edges[i + 1]) for i in range(5)]


def hero_figure(
    hist: pd.DataFrame,
    now: pd.Timestamp,
    now_value: float,
    fc: dict[int, dict[str, float | None]],
    meals: pd.DataFrame,
    cfg: dict[str, Any],
    ghost: dict[str, Any] | None = None,
    height: int = 520,
) -> Any:
    """The hero forecast chart.

    ``hist``: columns t, glucose (real readings). ``fc``: horizon -> mean, lo, hi (mg/dL).
    ``ghost``: optional what-if, ``{"fc": like fc, "traj": (times, values) or None}``.
    """
    import plotly.graph_objects as go

    x0, x1 = now - pd.Timedelta(hours=cfg["history_hours"]), now + pd.Timedelta(
        minutes=cfg["forecast_min"]
    )
    fig = go.Figure()
    hover = (
        "%{x|%H:%M}  %{y:.0f} mg/dL<extra></extra>"
        if units.unit() == units.MGDL
        else "%{x|%H:%M}  %{customdata:.1f} mmol/L<extra></extra>"
    )
    zl = cfg["zones"]
    labels = [
        f"Very low, below {units.num(zl['very_low'])}",
        f"Low, below {units.num(zl['low'])}",
        f"In range, {units.num(zl['low'])} to {units.num(zl['high'])}",
        f"High, above {units.num(zl['high'])}",
        f"Very high, above {units.num(zl['very_high'])}",
    ]
    for (ya, yb), (_, _, colour, shape, tint), label in zip(
        _bounds(cfg), ZONES, labels, strict=True
    ):
        pat = (
            dict(shape=shape, fgcolor=colour, bgcolor=tint, size=6, solidity=0.16)
            if shape
            else None
        )
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0], y=[ya, ya, yb, yb, ya], mode="lines", line=dict(width=0),
            fill="toself", fillpattern=pat, fillcolor=None if pat else tint,
            hoverinfo="skip", showlegend=False,
        ))  # fmt: skip
        fig.add_annotation(
            x=0.008, xref="paper", xanchor="left", y=min(yb - 4, ya + 24), yref="y", yanchor="top", text=label, showarrow=False,
            font=dict(size=11, color=SLATE, family=FONT), bgcolor="rgba(255,255,255,0.85)", borderpad=2,
        )  # fmt: skip
    # history: break the line where readings are more than 30 min apart
    h = hist[(hist["t"] >= x0) & (hist["t"] <= now)].sort_values("t")
    gap = h["t"].diff() > pd.Timedelta(minutes=30)
    ht, hv = [], []
    for t, g, brk in zip(h["t"], h["glucose"], gap, strict=True):
        if brk:
            ht.append(None), hv.append(None)
        ht.append(t), hv.append(g)
    fig.add_trace(go.Scatter(
        x=ht, y=hv, mode="lines", line=dict(color=INK, width=2), name="Glucose history",
        customdata=[None if v is None else units.conv(v) for v in hv], hovertemplate=hover,
    ))  # fmt: skip

    def fan(
        f: dict[int, dict[str, float | None]], dash: str | None, fill: str, name: str, solid: bool
    ) -> None:
        ts = [now] + [now + pd.Timedelta(minutes=m) for m in f]
        mean = [now_value] + [f[m]["mean"] for m in f]
        lo = [now_value] + [f[m]["lo"] if f[m]["lo"] is not None else f[m]["mean"] for m in f]
        hi = [now_value] + [f[m]["hi"] if f[m]["hi"] is not None else f[m]["mean"] for m in f]
        if any(f[m]["lo"] is not None for m in f):
            fig.add_trace(go.Scatter(
                x=ts + ts[::-1], y=hi + lo[::-1], mode="lines", line=dict(width=0), fill="toself",
                fillcolor=fill, hoverinfo="skip", showlegend=solid, name="80% interval",
            ))  # fmt: skip
        fig.add_trace(go.Scatter(
            x=ts, y=mean, mode="lines+markers", name=name,
            line=dict(color=DAMSON, width=3, dash=dash or "solid"),
            marker=dict(size=[0] + [9] * len(f), color=DAMSON, symbol="circle"),
            customdata=[units.conv(v) for v in mean], hovertemplate=hover,
        ))  # fmt: skip

    fan(fc, None, "rgba(67,40,94,0.22)", "Forecast (median)", True)
    if ghost is not None:
        if ghost.get("traj") is not None:
            tt, vv = ghost["traj"]
            fig.add_trace(go.Scatter(x=tt, y=vv, mode="lines", line=dict(color=DAMSON, width=2, dash="dot"),
                                     name="What-if (full twin path)", hoverinfo="skip"))  # fmt: skip
        fan(ghost["fc"], "dash", "rgba(67,40,94,0.10)", "What-if", False)
    m = meals[(meals["t"] >= x0) & (meals["t"] <= x1)]
    if len(m):
        fig.add_trace(go.Scatter(
            x=m["t"], y=[cfg["y_range"][0] + 3] * len(m), mode="markers", name="Meal",
            marker=dict(symbol="triangle-up", size=11, color=INK, line=dict(color="white", width=1)),
            customdata=m["carbs_g"], hovertemplate="Meal %{x|%H:%M}, %{customdata:.0f} g carbohydrate<extra></extra>",
        ))  # fmt: skip
    fig.add_vline(x=now, line=dict(color=INK, width=1))
    fig.add_annotation(x=now, y=cfg["y_range"][1] - 6, yref="y", yanchor="top", xanchor="right", showarrow=False,
                       text=f"<b>Now {units.fmt(now_value)}</b>", font=dict(size=13, color=INK, family=FONT),
                       bgcolor="rgba(255,255,255,0.92)", bordercolor=RULE, borderwidth=1, borderpad=4, xshift=-6)  # fmt: skip
    fig.update_layout(
        height=height, margin=dict(l=56, r=16, t=16, b=64), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK, size=14), hovermode="closest", dragmode=False,
        legend=dict(orientation="h", y=-0.12, x=0, font=dict(size=13)),
        xaxis=dict(range=[x0, x1], tickformat="%H:%M", showgrid=False, linecolor=RULE, ticks="outside"),
        yaxis=dict(range=list(cfg["y_range"]), title=dict(text=f"Glucose ({units.unit()})", font=dict(size=13, color=SLATE)),
                   showgrid=False, linecolor=RULE, ticks="outside", fixedrange=True),
    )  # fmt: skip
    ticks = units.axis_ticks(*cfg["y_range"])
    if ticks:
        fig.update_yaxes(tickmode="array", tickvals=ticks[0], ticktext=ticks[1])
    return fig


def curve_figure(curve: pd.DataFrame, title: str) -> Any:
    """Cold-start curve: mean absolute error against days of own data, with 95% intervals."""
    import plotly.graph_objects as go

    fig = go.Figure()
    style = [(SLATE, "rgba(86,97,116,0.12)", "dash"), (DAMSON, "rgba(67,40,94,0.12)", "solid")]
    for (colour, fill, dash), (name, d) in zip(
        style, curve.groupby("model", sort=False), strict=False
    ):
        d = d.sort_values("k")
        fig.add_trace(go.Scatter(x=list(d["k"]) + list(d["k"])[::-1], y=list(d["hi"]) + list(d["lo"])[::-1],
                                 fill="toself", fillcolor=fill, line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))  # fmt: skip
        fig.add_trace(go.Scatter(x=d["k"], y=d["mae"], mode="lines+markers", name=name,
                                 line=dict(color=colour, width=2.5, dash=dash), marker=dict(size=7)))  # fmt: skip
    fig.update_layout(
        height=380, margin=dict(l=56, r=16, t=16, b=72), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK, size=14), legend=dict(orientation="h", y=-0.28, x=0),
        xaxis=dict(title=dict(text="Days of the patient's own data used (k)", font=dict(size=13, color=SLATE)),
                   showgrid=False, linecolor=RULE, ticks="outside"),
        yaxis=dict(title=dict(text=title, font=dict(size=13, color=SLATE)), gridcolor=RULE, linecolor=RULE),
    )  # fmt: skip
    return fig


def agp_figure(profile: pd.DataFrame, cfg: dict[str, Any], height: int = 430) -> Any:
    """Ambulatory glucose profile: median, 25-75% and 5-95% bands by time of day."""
    import plotly.graph_objects as go

    base = pd.Timestamp("2000-01-01")
    x = [base + pd.Timedelta(minutes=int(m)) for m in profile["minute"]]
    z = cfg["zones"]
    y0, y1 = cfg["y_range"]
    y1 = max(y1, float(profile["q95"].max()) + 15)
    fig = go.Figure()
    fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=z["low"], y1=z["high"],
                  fillcolor="rgba(154,211,184,0.22)", line=dict(width=0), layer="below")  # fmt: skip
    for edge in (z["low"], z["high"]):
        fig.add_hline(y=edge, line=dict(color=SLATE, width=1, dash="dot"))
    fig.add_annotation(
        x=0.008, xref="paper", xanchor="left", y=z["high"] - 4, yref="y", yanchor="top",
        text=f"Target range, {units.num(z['low'])} to {units.num(z['high'])}", showarrow=False,
        font=dict(size=11, color=SLATE, family=FONT), bgcolor="rgba(255,255,255,0.85)", borderpad=2,
    )  # fmt: skip
    for lo, hi, fill, name in (("q05", "q95", "rgba(67,40,94,0.12)", "5 to 95% of readings"),
                               ("q25", "q75", "rgba(67,40,94,0.26)", "25 to 75% of readings")):  # fmt: skip
        fig.add_trace(go.Scatter(
            x=x + x[::-1], y=list(profile[hi]) + list(profile[lo])[::-1], mode="lines",
            line=dict(width=0), fill="toself", fillcolor=fill, name=name, hoverinfo="skip",
        ))  # fmt: skip
    hover = (
        "%{x|%H:%M}  median %{y:.0f} mg/dL<extra></extra>"
        if units.unit() == units.MGDL
        else "%{x|%H:%M}  median %{customdata:.1f} mmol/L<extra></extra>"
    )
    fig.add_trace(go.Scatter(
        x=x, y=profile["q50"], mode="lines", name="Median", line=dict(color=DAMSON, width=3),
        customdata=[units.conv(v) for v in profile["q50"]], hovertemplate=hover,
    ))  # fmt: skip
    fig.update_layout(
        height=height, margin=dict(l=56, r=16, t=16, b=64), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK, size=14), hovermode="closest", dragmode=False,
        legend=dict(orientation="h", y=-0.14, x=0, font=dict(size=13)),
        xaxis=dict(tickformat="%H:%M", dtick=3 * 3600 * 1000, showgrid=False, linecolor=RULE, ticks="outside"),
        yaxis=dict(range=[y0, y1], title=dict(text=f"Glucose ({units.unit()})", font=dict(size=13, color=SLATE)),
                   showgrid=False, linecolor=RULE, ticks="outside", fixedrange=True),
    )  # fmt: skip
    ticks = units.axis_ticks(y0, y1)
    if ticks:
        fig.update_yaxes(tickmode="array", tickvals=ticks[0], ticktext=ticks[1])
    return fig


def tir_figure(shares: dict[str, float], height: int = 150) -> Any:
    """One stacked bar of the five glucose zones (pattern and label on every segment)."""
    import plotly.graph_objects as go

    names = {
        "very_low": "Very low",
        "low": "Low",
        "range": "In range",
        "high": "High",
        "very_high": "Very high",
    }
    fig = go.Figure()
    for key, _, colour, shape, _tint in ZONES:
        v = shares.get(key, 0.0)
        pat = dict(shape=shape, fgcolor=colour, size=6, solidity=0.3) if shape else None
        fig.add_trace(go.Bar(
            x=[v], y=[""], orientation="h", name=names[key], text=[f"{v:.0f}%" if v >= 4 else ""],
            textposition="inside", insidetextanchor="middle",
            marker=dict(color=SOLID[key], pattern=pat, line=dict(color="white", width=1)),
            hovertemplate=f"{names[key]}: {v:.1f}%<extra></extra>",
        ))  # fmt: skip
    fig.update_layout(
        barmode="stack", height=height, margin=dict(l=8, r=8, t=8, b=56), paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE, font=dict(family=FONT, color=INK, size=14),
        legend=dict(orientation="h", y=-0.5, x=0, font=dict(size=13)),
        xaxis=dict(range=[0, 100], showgrid=False, visible=False), yaxis=dict(visible=False), dragmode=False,
    )  # fmt: skip
    return fig
