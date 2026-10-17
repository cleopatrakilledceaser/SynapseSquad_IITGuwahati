"""What-if windows and the precomputed twin grid of the demo patients (D-058).

ML what-ifs run live in the app (fold models are shipped). The twin needs the fitted per-patient
state, so its results are precomputed on a (carbs, slowing ratio) grid for each window, once with
the learned activity effect (Part 4 alpha_on) and once with the ASSUMED walk effect of D-052, both
with the D-056 one-sided activity term. Values between nodes are interpolated in the app and
labelled; ``interp_check`` measures the interpolation error at off-grid points.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.whatif import edits as ed
from src.whatif.engine import twin_run
from src.whatif.plausibility import last_meal

VARIANTS = ("twin_learned", "twin_assumed")


def pick_windows(per: Any, rows: pd.DataFrame, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Replay anchors at the configured offsets after each logged meal with macros."""
    out, seen = [], set()
    tol = pd.Timedelta(minutes=cfg["whatif_tolerance_min"])
    for off in cfg["whatif_offsets_min"]:
        for mid, m in per.meals.iterrows():
            target = m["t"] + pd.Timedelta(minutes=off)
            d = (rows["t"] - target).abs()
            if d.min() > tol:
                continue
            t = rows.loc[d.idxmin(), "t"]
            if (
                last_meal(per, t, 0.0, float(max(cfg["whatif_offsets_min"]) + tol.seconds / 60))
                != mid
            ):
                continue
            if (mid, t) in seen:
                continue
            seen.add((mid, t))
            out.append(
                {"t": t, "meal": mid, "offset_min": float((t - m["t"]).total_seconds() / 60)}
            )
    return sorted(out, key=lambda w: w["t"])


def slow_ratio(m: dict[str, float], w_protein: float) -> float:
    """(fat + w * protein + fiber) / (carbs + 1): the slowing ratio the twin uses."""
    return float((m["fat_g"] + w_protein * m["protein_g"] + m["fiber_g"]) / (m["carbs_g"] + 1.0))


def node_edit(meal: Any, carbs: float, ratio: float, walk: bool) -> list[ed.Edit]:
    """Edit set realizing grid node (carbs, ratio): the twin only sees carbs and the ratio."""
    fat = ratio * (carbs + 1.0)
    e = [ed.set_macros(meal, carbs_g=float(carbs), protein_g=0.0, fat_g=float(fat), fiber_g=0.0)]
    return e + ([ed.add_walk(meal)] if walk else [])


def twin_grid(
    fc: Any, per: Any, win: dict[str, Any], alphas: dict[str, float], grid: dict[str, list[float]]
) -> np.ndarray:
    """Trajectories, shape (variant, walk, carbs, ratio, steps) of one window (NaN if no twin)."""
    cs, rs = grid["carbs_g"], grid["slow_ratio"]
    out = None
    for iv, v in enumerate(VARIANTS):
        for iw, walk in enumerate((False, True)):
            for ic, c in enumerate(cs):
                for ir, r in enumerate(rs):
                    g = twin_run(fc, per, win["t"], node_edit(win["meal"], c, r, walk), alphas[v])[
                        "g"
                    ]
                    if out is None:
                        out = np.full((len(VARIANTS), 2, len(cs), len(rs), len(g)), np.nan)
                    out[iv, iw, ic, ir] = g
    return out


def interp(
    arr: np.ndarray, cs: list[float], rs: list[float], c: float, r: float
) -> tuple[np.ndarray, bool]:
    """Bilinear interpolation of ``arr`` (carbs, ratio, steps) at (c, r); inputs are clamped.

    Returns (values, interpolated) where ``interpolated`` is False only on an exact grid node.
    """
    c, r = float(np.clip(c, cs[0], cs[-1])), float(np.clip(r, rs[0], rs[-1]))
    on_node = c in cs and r in rs

    def idx(x: float, g: list[float]) -> tuple[int, int, float]:
        j = int(np.clip(np.searchsorted(g, x, side="right") - 1, 0, len(g) - 2))
        return j, j + 1, (x - g[j]) / (g[j + 1] - g[j])

    i0, i1, fc = idx(c, cs)
    j0, j1, fr = idx(r, rs)
    v = (
        arr[i0, j0] * (1 - fc) * (1 - fr)
        + arr[i1, j0] * fc * (1 - fr)
        + arr[i0, j1] * (1 - fc) * fr
        + arr[i1, j1] * fc * fr
    )
    return v, not on_node


def interp_check(
    fc: Any,
    per: Any,
    win: dict[str, Any],
    alphas: dict[str, float],
    grid: dict[str, list[float]],
    arr: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> float:
    """Max |exact twin - interpolated| (mg/dL over the whole rollout) at ``n`` random off-grid points."""
    cs, rs = grid["carbs_g"], grid["slow_ratio"]
    worst = 0.0
    for _ in range(n):
        c, r = rng.uniform(cs[0], cs[-1]), rng.uniform(rs[0], rs[-1])
        iv, walk = int(rng.integers(len(VARIANTS))), bool(rng.integers(2))
        ex = twin_run(fc, per, win["t"], node_edit(win["meal"], c, r, walk), alphas[VARIANTS[iv]])[
            "g"
        ]
        ip, _ = interp(arr[iv, int(walk)], cs, rs, c, r)
        worst = max(worst, float(np.nanmax(np.abs(ex - ip))))
    return worst
