"""Part 5.1 figures (300 dpi, labeled axes with units, Okabe-Ito colorblind-safe palette)."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.mech.figures import COLORS, _save  # noqa: E402

HORIZONS = [30, 60, 120]
SUBSETS = [("overall", "overall"), ("post_meal_3h", "post-meal (3 h)"), ("onset_hyper140", "onset")]


def _bars(ax: Any, t: pd.DataFrame, models: list[str]) -> None:
    wbar = 0.8 / len(models)
    for i, m in enumerate(models):
        d = t[t.model == m].sort_values("horizon_min")
        if d.empty:
            continue
        lo = d.mae - [c[0] for c in d.mae_ci]
        hi = [c[1] for c in d.mae_ci] - d.mae
        ax.bar(np.arange(len(d)) + i * wbar, d.mae, wbar, yerr=[lo, hi], capsize=2,
               color=COLORS[i % 8], hatch="//" if i >= 8 else None, label=m)  # fmt: skip
    ax.set_xticks(np.arange(3) + 0.4 - wbar / 2, [f"+{h}" for h in HORIZONS])
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel("MAE (mg/dL)")


def comparison(res: dict[str, Any], ks: list[int], name: str) -> None:
    """MAE with patient-bootstrap CIs: rows = k, columns = subset (identical rows per k)."""
    fig, axes = plt.subplots(len(ks), 3, figsize=(16, 4.2 * len(ks)), squeeze=False)
    for r, k in enumerate(ks):
        t = pd.DataFrame(res[f"regression_k{k}"])
        models = list(dict.fromkeys(t.model))
        for c, (s, lab) in enumerate(SUBSETS):
            ax = axes[r][c]
            _bars(ax, t[t.subset == s], models)
            ax.set_title(f"k = {k} d, {lab} (n = {res[f'n_patients_k{k}']} patients)")
        axes[r][0].legend(fontsize=6)
    _save(fig, name)


def seed_variance(rows: list[dict[str, Any]], name: str) -> None:
    """Hybrid C pooled MAE per seed (k = 3), with and without the physics loss."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, m in enumerate(sorted({r["model"] for r in rows})):
        for r in [r for r in rows if r["model"] == m]:
            x = HORIZONS.index(r["horizon_min"]) + (i - 0.5) * 0.25
            ax.scatter(np.full(len(r["seed_mae"]), x), r["seed_mae"], color=COLORS[i + 1],
                       s=14, label=m if r["horizon_min"] == 30 else None)  # fmt: skip
            ax.errorbar(x + 0.06, r["mean"], yerr=r["sd"], color=COLORS[i + 1], capsize=3)
    ax.set_xticks(range(3), [f"+{h}" for h in HORIZONS])
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel("MAE per seed (mg/dL)")
    ax.set_title("Hybrid C seed variance (5 seeds, mean +/- SD)")
    ax.legend(fontsize=7)
    _save(fig, name)


def onset_fix(res: dict[str, Any], name: str) -> None:
    """Unweighted vs onset-weighted GRU + FiLM: overall and onset MAE, both cohorts."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    models = ["nn_B_cgm", "nnw_B_cgm", "nn_E_full", "nnw_E_full", "lgbm_E_full_delta",
              "ar_per_patient"]  # fmt: skip
    for r, (coh, rc) in enumerate(res.items()):
        t = pd.DataFrame(rc["regression"])
        for c, (s, lab) in enumerate([("overall", "overall"), ("onset_hyper140", "onset")]):
            ax = axes[r][c]
            _bars(ax, t[t.subset == s], [m for m in models if m in set(t.model)])
            ax.set_title(f"{coh}: {lab} (nnw = onset-weighted loss)")
        axes[r][0].legend(fontsize=6)
    _save(fig, name)
