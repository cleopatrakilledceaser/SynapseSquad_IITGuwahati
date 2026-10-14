"""Part 5.2 figures (300 dpi, labeled axes with units, Okabe-Ito colorblind-safe palette)."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.mech.figures import COLORS, _save  # noqa: E402

COHORT_LABEL = {"cgmacros": "CGMacros", "shanghai_t2dm": "Shanghai T2DM"}
METHOD_LABEL = {
    "none": "population (no adaptation)",
    "scratch": "per-patient ridge (scratch)",
    "resid": "residual ridge",
    "shrink": "shrinkage blend",
    "finetune": "LightGBM fine-tune",
    "rls": "RLS (frozen at k)",
    "rls_cont": "RLS (continuous)",
}


def _band(ax: Any, d: pd.DataFrame, color: str, label: str, ls: str = "-") -> None:
    d = d.sort_values("k")
    ax.plot(d.k, d.mae, ls, marker="o", ms=3, color=color, label=label)
    ax.fill_between(d.k, [c[0] for c in d.mae_ci], [c[1] for c in d.mae_ci], color=color,
                    alpha=0.15, lw=0)  # fmt: skip


def cold_start(res: dict[str, Any], base: str, subset: str, name: str) -> None:
    """MAE vs k with patient-bootstrap 95% bands, with (solid) / without (dashed) static."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), squeeze=False)
    for r, (coh, rc) in enumerate(res.items()):
        for c, h in enumerate((30, 60, 120)):
            ax = axes[r][c]
            key = f"curve_{subset}_{h}"
            if key not in rc:
                ax.set_visible(False)
                continue
            t = pd.DataFrame(rc[key])
            for i, m in enumerate(("none", "shrink", "finetune", "rls", "rls_cont")):
                for st, ls in (("s", "-"), ("ns", "--")):
                    d = t[t.model == f"{base}_{st}_{m}"]
                    if len(d):
                        lab = f"{METHOD_LABEL[m]}, {'with' if st == 's' else 'no'} static"
                        _band(ax, d, COLORS[i], lab, ls)
            d = t[t.model == "twin_s_twin"]
            if len(d):
                _band(ax, d, COLORS[6], "mechanistic twin, static prior", ":")
            ax.set_title(f"{COHORT_LABEL[coh]}, +{h} min (n = {rc[f'n_patients_{subset}_{h}']})")
            ax.set_xlabel("Adaptation days of the patient's own data (k, days)")
            ax.set_ylabel(f"{'Onset ' if subset == 'onset' else ''}MAE (mg/dL)")
    axes[0][0].legend(fontsize=5.5, ncol=1)
    _save(fig, name)


def crossover_plot(res: dict[str, Any], name: str) -> None:
    """Static head start vs the best no-static adapted method; bootstrap crossover k*."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), squeeze=False)
    for r, (coh, rc) in enumerate(res.items()):
        t = pd.DataFrame(rc["curve_overall_60"])
        for c, base in enumerate(("lgbm", "hybrid", "twin")):
            ax = axes[r][c]
            co = rc["crossover_overall_60"][base]
            best = co["best_no_static_method"]
            _band(ax, t[t.model == best], COLORS[5], f"{best} (no static, adapted)")
            if base != "twin":
                _band(ax, t[t.model == f"{base}_ns_none"], COLORS[1], "population, no static", "--")
            ref = "twin_s_twin" if base == "twin" else f"{base}_s_none"
            s0 = t[(t.model == ref) & (t.k == 0)].mae.iloc[0]
            ax.axhline(s0, color=COLORS[0], lw=1, label="population with static, k = 0")
            kx = co["crossover_k"]
            if np.isfinite(kx):
                ax.axvline(kx, color=COLORS[6], ls=":", label=f"crossover k* = {kx:.2f} d")
            ci = co["crossover_ci"]
            txt = (f"head start {co['head_start_mgdl']:.2f} "
                   f"[{co['head_start_ci'][0]:.2f}, {co['head_start_ci'][1]:.2f}] mg/dL\n"
                   f"k* CI {ci if ci is None else [round(x, 2) for x in ci]}; "
                   f"no crossover: {100 * co['frac_no_crossover']:.0f}% of resamples")  # fmt: skip
            ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=6.5)
            ax.set_title(f"{COHORT_LABEL[coh]}, base = {base}, +60 min")
            ax.set_xlabel("Adaptation days (k, days)")
            ax.set_ylabel("MAE (mg/dL)")
            ax.legend(fontsize=6, loc="upper right")  # annotation sits lower left
    _save(fig, name)


def method_comparison(res: dict[str, Any], name: str) -> None:
    """Paired MAE difference vs no adaptation at k = 1/3/5 (+60 min), with 95% CIs."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), squeeze=False)
    for r, (coh, rc) in enumerate(res.items()):
        ax = axes[r][0]
        t = pd.DataFrame(rc["methods_overall_60"])
        t = t[t["diff_vs_none"].notna()] if "diff_vs_none" in t else t.iloc[:0]
        labs = sorted(t.model.unique())
        ks = sorted(t.k.unique())
        wbar = 0.8 / max(len(ks), 1)
        for i, k in enumerate(ks):
            d = t[t.k == k].set_index("model").reindex(labs)
            ci = [c if isinstance(c, list) else [np.nan, np.nan] for c in d.diff_ci]
            lo = d.diff_vs_none - [c[0] for c in ci]
            hi = [c[1] for c in ci] - d.diff_vs_none
            ax.bar(np.arange(len(labs)) + i * wbar, d.diff_vs_none, wbar, yerr=[lo, hi],
                   capsize=2, color=COLORS[i + 1], label=f"k = {k:g} d")  # fmt: skip
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(np.arange(len(labs)) + 0.4 - wbar / 2, labs, rotation=60, ha="right",
                      fontsize=6)  # fmt: skip
        ax.set_ylabel("MAE difference vs no adaptation (mg/dL)\n< 0 = adaptation helps")
        ax.set_title(f"{COHORT_LABEL[coh]}, +60 min (paired patient bootstrap)")
        ax.legend(fontsize=7)
    _save(fig, name)


def gain_distribution(res: dict[str, Any], name: str, k: float = 3.0) -> None:
    """Per-patient gain (MAE none - MAE method) at k = 3 d, +60 min: > 0 = the patient improves."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), squeeze=False)
    for c, (coh, rc) in enumerate(res.items()):
        ax = axes[0][c]
        t = sorted([g for g in rc["gains_overall_60"] if g["k"] == k], key=lambda g: g["model"])
        ax.boxplot([g["gains"] for g in t], showfliers=True, flierprops={"ms": 2})
        ax.set_xticks(range(1, len(t) + 1),
                      [f"{g['model']}\n{g['n_improve']}/{g['n_patients']}" for g in t],
                      rotation=60, ha="right", fontsize=5.5)  # fmt: skip
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Per-patient MAE gain (mg/dL), > 0 = better")
        ax.set_title(f"{COHORT_LABEL[coh]}, k = {k:g} d, +60 min (improved / patients)")
    _save(fig, name)


def all_figures(res: dict[str, Any]) -> None:
    """Every Part 5.2 figure."""
    for base in ("lgbm", "hybrid"):
        cold_start(res, base, "overall", f"personalize_cold_start_{base}")
        cold_start(res, base, "onset", f"personalize_cold_start_onset_{base}")
    crossover_plot(res, "personalize_crossover")
    method_comparison(res, "personalize_method_comparison")
    gain_distribution(res, "personalize_gain_distribution")
