"""Plotly figures of the dashboard (Part 9, docs/DESIGN.md sections 2 and 5).

Plotly is imported inside the functions so the rest of ``src.dash`` imports without it. Colours and
patterns follow the design tokens: the forecast wears the brand colour (damson), glucose zones use
four fixed colours that always come with a pattern and a text label.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

INK, SLATE, RULE, DAMSON = "#1B2230", "#566174", "#D8DDE5", "#43285E"
PAPER, SURFACE = "#F5F6F8", "#FFFFFF"
FONT = "Atkinson Hyperlegible, 'Segoe UI', system-ui, sans-serif"
SERIF = "'Source Serif 4', Georgia, serif"
ZONES = [  # (low, high, label, colour, pattern shape or None)
    ("very_low", "Very low, below 54", "#2B6CB0", "x"),
    ("low", "Low, below 70", "#2B6CB0", "/"),
    ("range", "In range, 70 to 180", "#9AD3B8", None),
    ("high", "High, above 180", "#E09A12", "."),
    ("very_high", "Very high, above 250", "#8C1D2B", "+"),
]


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

    x0, x1 = now - pd.Timedelta(hours=cfg["history_hours"]), now + pd.Timedelta(minutes=cfg["forecast_min"])
    fig = go.Figure()
    for (ya, yb), (_, label, colour, shape) in zip(_bounds(cfg), ZONES, strict=True):
        pat = (
            dict(shape=shape, fgcolor=colour, bgcolor="rgba(255,255,255,0)", size=7, solidity=0.28)
            if shape
            else None
        )
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0], y=[ya, ya, yb, yb, ya], mode="lines", line=dict(width=0),
            fill="toself", fillpattern=pat, fillcolor=None if pat else "rgba(154,211,184,0.30)",
            hoverinfo="skip", showlegend=False,
        ))  # fmt: skip
        fig.add_annotation(
            x=1, xref="paper", xanchor="right", y=(ya + yb) / 2, yref="y", text=label, showarrow=False,
            font=dict(size=12, color=INK, family=FONT), bgcolor="rgba(255,255,255,0.78)", borderpad=2,
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
        hovertemplate="%{x|%H:%M}  %{y:.0f} mg/dL<extra></extra>",
    ))  # fmt: skip

    def fan(f: dict[int, dict[str, float | None]], dash: str | None, fill: str, name: str, solid: bool) -> None:
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
            hovertemplate="%{x|%H:%M}  %{y:.0f} mg/dL<extra></extra>",
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
    fig.add_annotation(x=now, y=cfg["y_range"][1], yref="y", yanchor="top", xanchor="right", showarrow=False,
                       text=f"Now {now_value:.0f} mg/dL", font=dict(size=13, color=INK, family=FONT))  # fmt: skip
    fig.update_layout(
        height=height, margin=dict(l=56, r=16, t=16, b=64), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK, size=14), hovermode="closest", dragmode=False,
        legend=dict(orientation="h", y=-0.12, x=0, font=dict(size=13)),
        xaxis=dict(range=[x0, x1], tickformat="%H:%M", showgrid=False, linecolor=RULE, ticks="outside"),
        yaxis=dict(range=list(cfg["y_range"]), title=dict(text="Glucose (mg/dL)", font=dict(size=13, color=SLATE)),
                   showgrid=False, linecolor=RULE, ticks="outside", fixedrange=True),
    )  # fmt: skip
    return fig


def curve_figure(curve: pd.DataFrame, title: str) -> Any:
    """Cold-start curve: mean absolute error against days of own data, with 95% intervals."""
    import plotly.graph_objects as go

    fig = go.Figure()
    style = [(SLATE, "rgba(86,97,116,0.12)", "dash"), (DAMSON, "rgba(67,40,94,0.12)", "solid")]
    for (colour, fill, dash), (name, d) in zip(style, curve.groupby("model", sort=False), strict=False):
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
