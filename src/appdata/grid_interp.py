"""Bilinear interpolation on the twin what-if grid (Part 9, D-058).

Kept in its own module so the app can import it without the training stack (the what-if engine
behind ``whatif_grid`` needs LightGBM and statsmodels).
"""

from __future__ import annotations

import numpy as np


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
