"""Part 4 figures (300 dpi, labeled axes with units, Okabe-Ito colorblind-safe palette)."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.mech.model import FIT_PARAMS  # noqa: E402
from src.utils.config import PROJECT_ROOT  # noqa: E402

FIG = PROJECT_ROOT / "reports" / "figures"
HORIZONS = [30, 60, 120]
COLORS = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
DPI = 300


def _save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=DPI)
    plt.close(fig)


def _mae_table(rows: list[dict], subset: str = "overall") -> pd.DataFrame:
    t = pd.DataFrame(rows)
    return t[t.subset == subset]


def comparison(res: dict[str, Any], name: str, ks: list[int]) -> None:
    """MAE with patient-bootstrap CIs per horizon, one panel per k (identical windows per panel)."""
    fig, axes = plt.subplots(1, len(ks), figsize=(5.2 * len(ks), 4.2), squeeze=False)
    for ax, k in zip(axes[0], ks, strict=True):
        t = _mae_table(res[f"regression_k{k}"])
        ms = list(dict.fromkeys(t.model))
        wbar = 0.8 / len(ms)
        for i, m in enumerate(ms):
            d = t[t.model == m].sort_values("horizon_min")
            lo = d.mae - [c[0] for c in d.mae_ci]
            hi = [c[1] for c in d.mae_ci] - d.mae
            ax.bar(np.arange(3) + i * wbar, d.mae, wbar, yerr=[lo, hi], capsize=2,
                   color=COLORS[i % 8], label=m, hatch="//" if i >= 8 else None)  # fmt: skip
        ax.set_xticks(np.arange(3) + 0.4 - wbar / 2, [f"+{h}" for h in HORIZONS])
        ax.set_xlabel("Forecast horizon (min)")
        ax.set_ylabel("MAE (mg/dL)")
        ax.set_title(f"k = {k} adaptation day(s)")
        ax.legend(fontsize=6)
    _save(fig, name)


def error_vs_k(res: dict[str, Any], name: str) -> None:
    """Twin MAE vs k (pop / static prior, activity on / off), rows common to all k."""
    t = _mae_table(res["twin_vs_k"])
    t = t.assign(k=t.model.str.extract(r"_k(\d)$").astype(int)[0])
    t = t.assign(variant=t.model.str.replace(r"_k\d$", "", regex=True))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    for ax, h in zip(axes, HORIZONS, strict=True):
        for i, (v, g) in enumerate(t[t.horizon_min == h].groupby("variant")):
            g = g.sort_values("k")
            lo = g.mae - [c[0] for c in g.mae_ci]
            hi = [c[1] for c in g.mae_ci] - g.mae
            ax.errorbar(g.k + 0.05 * i, g.mae, yerr=[lo, hi], marker="o", capsize=3,
                        color=COLORS[i % 8], label=v)  # fmt: skip
        ax.set_xlabel("Adaptation days used for parameters (k)")
        ax.set_ylabel("MAE (mg/dL)")
        ax.set_title(f"+{h} min")
        ax.set_xticks([0, 1, 3])
    axes[0].legend(fontsize=7)
    _save(fig, name)


def coverage(res: dict[str, Any], name: str, model: str) -> None:
    """Empirical vs nominal coverage of the twin's Gaussian predictive intervals."""
    t = pd.DataFrame(res["coverage"])
    t = t[t.model == model]
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, h in enumerate(HORIZONS):
        d = t[t.horizon_min == h].sort_values("level")
        lo = d.coverage - [c[0] for c in d.coverage_ci]
        hi = [c[1] for c in d.coverage_ci] - d.coverage
        ax.errorbar(d.level + 0.004 * i, d.coverage, yerr=[lo, hi], marker="o", capsize=3,
                    color=COLORS[i + 1], label=f"+{h} min")  # fmt: skip
    ax.plot([0.75, 1.0], [0.75, 1.0], "k--", lw=1, label="ideal")
    ax.set_xlabel("Nominal interval level")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(model, fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, name)


def recovery(rec_df: pd.DataFrame, summary: dict[str, Any], name: str) -> None:
    """True vs estimated log parameters on SYNTHETIC patients."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, p in zip(axes.ravel(), FIT_PARAMS, strict=True):
        r = rec_df[rec_df.param == p]
        ax.scatter(r.true_log, r.est_log, s=8, color=COLORS[5], alpha=0.6)
        lim = [min(r.true_log.min(), r.est_log.min()), max(r.true_log.max(), r.est_log.max())]
        ax.plot(lim, lim, "k--", lw=1)
        s = summary["params"][p]
        ax.set_title(
            f"{p}: synthetic r = {s['recovery_corr_true_est']:.2f}, "
            f"real retest r = {s['retest_corr_log']:.2f}",
            fontsize=8,
        )
        ax.set_xlabel(f"true log {p}")
        ax.set_ylabel(f"estimated log {p}")
    fig.suptitle("Parameter recovery on SYNTHETIC patients (real meal/activity schedules)")
    _save(fig, name)


def realism(real: pd.DataFrame, syn: pd.DataFrame, name: str) -> None:
    """Distributions of per-patient summary statistics: real CGMacros vs SYNTHETIC."""
    cols = [("mean", "Mean glucose (mg/dL)"), ("cv_pct", "CV (%)"), ("pct_gt180", "Time >180 (%)"),
            ("pct_140_180", "Time 140-180 (%)"), ("meal_peak_rise", "Post-meal rise (mg/dL)"),
            ("meal_time_to_peak_min", "Time to peak (min)")]  # fmt: skip
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax, (c, lab) in zip(axes.ravel(), cols, strict=True):
        ax.boxplot([real[c].dropna(), syn[c].dropna()], tick_labels=["real", "SYNTHETIC"])
        ax.set_ylabel(lab)
    fig.suptitle("Simulator realism: real CGMacros patients vs SYNTHETIC patients")
    _save(fig, name)


def whatif(traj: dict[str, np.ndarray], minutes: np.ndarray, name: str, alpha: float) -> None:
    """SIMULATED what-if: 30 vs 60 g meal, with vs without a walk."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.text(
        0.01, 0.90,
        f"population activity effect α_E = {alpha:g} /min (selected on training patients;\n"
        "grid floor: the data did not support a larger walk effect, D-033)",
        transform=ax.transAxes, ha="left", va="top", fontsize=7,
    )  # fmt: skip
    for i, (lab, g) in enumerate(traj.items()):
        ax.plot(minutes / 60, g, color=COLORS[[1, 6, 2, 5][i % 4]],
                ls="-" if "no walk" in lab else "--", label=lab)  # fmt: skip
    ax.axhline(140, color="grey", ls=":", lw=1)
    ax.axhline(180, color="grey", ls="-.", lw=1)
    ax.set_xlabel("Time after meal (h)")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title("SIMULATED what-if (mechanistic twin, illustrative patient) — not a forecast")
    ax.legend(fontsize=8)
    _save(fig, name)


def example_forecasts(w: pd.DataFrame, model: str, pids: list[str], name: str) -> None:
    """+60-min twin forecasts with 80% / 95% bands vs real readings for a few patients (1 day)."""
    from scipy.stats import norm

    fig, axes = plt.subplots(len(pids), 1, figsize=(10, 3 * len(pids)), squeeze=False)
    for ax, p in zip(axes[:, 0], pids, strict=True):
        d = w[w.patient_id == p].sort_values("t")
        d = d[d.t < d.t.iloc[0] + pd.Timedelta(hours=24)]
        tt = d.t + pd.Timedelta(minutes=60)
        mu, sd = d[f"{model}|60"], np.sqrt(d[f"{model}|var60"])
        for lv, a in ((0.95, 0.15), (0.8, 0.3)):
            z = norm.ppf(0.5 + lv / 2)
            ax.fill_between(tt, mu - z * sd, mu + z * sd, color=COLORS[2], alpha=a, lw=0,
                            label=f"{int(lv * 100)}% interval")  # fmt: skip
        ax.plot(tt, mu, color=COLORS[5], lw=1, label="twin +60 min forecast")
        ax.plot(tt, d["y_60"], ".", color=COLORS[0], ms=3, label="real CGM at t+60")
        ax.set_ylabel("Glucose (mg/dL)")
        ax.set_title(p, fontsize=9)
    axes[-1, 0].set_xlabel("Time (date-shifted, first test day)")
    axes[0, 0].legend(fontsize=7, ncol=4)
    _save(fig, name)


def profiles(prof: pd.DataFrame, name: str) -> None:
    """Profile likelihood curves (real patients, 3 days)."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, p in zip(axes.ravel(), FIT_PARAMS, strict=True):
        for i, (pid, g) in enumerate(prof[prof.param == p].groupby("patient_id")):
            ax.plot(g.log_value, g.delta_m2ll, marker=".", color=COLORS[i % 8], label=pid[-3:])
        ax.axhline(1.92 * 2, color="grey", ls=":", lw=1)
        ax.set_xlabel(f"log {p}")
        ax.set_ylabel("Δ(−2 log L)")
        ax.set_title(p)
    axes[0, 0].legend(fontsize=6, title="patient")
    fig.suptitle("Profile likelihood, real CGMacros patients (first 3 days); dotted = 95% cut")
    _save(fig, name)
