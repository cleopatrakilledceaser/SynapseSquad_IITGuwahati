"""Part 3 figures (300 dpi, labeled axes with units, colorblind-safe Okabe-Ito palette)."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.config import PROJECT_ROOT  # noqa: E402

FIG = PROJECT_ROOT / "reports" / "figures"
HORIZONS = [30, 60, 120]
COLORS = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def _err(d: pd.DataFrame) -> list[np.ndarray]:
    lo = d["mae"] - np.array([c[0] for c in d["mae_ci"]])
    hi = np.array([c[1] for c in d["mae_ci"]]) - d["mae"]
    return [lo.to_numpy(), hi.to_numpy()]


def ablation(results: dict[str, Any], fc: dict[str, Any], dpi: int) -> None:
    """Ablation bars with CIs at +60 min, both targets, per cohort."""
    cohorts = list(results)
    fig, axes = plt.subplots(len(cohorts), 2, figsize=(11, 4 * len(cohorts)), squeeze=False)
    for i, coh in enumerate(cohorts):
        t = pd.DataFrame(results[coh]["ablation"])
        t = t[(t.subset == "overall") & (t.horizon_min == 60)].set_index("model")
        for j, tgt in enumerate(fc["targets"]):
            ms = ["persistence", "ar_per_patient"] + [
                f"lgbm_{n}_{tgt}" for n in fc["ablation"][coh]
            ]
            ms = [m for m in ms if m in t.index]
            d = t.loc[ms]
            axes[i, j].bar(
                np.arange(len(ms)),
                d.mae,
                yerr=_err(d),
                capsize=3,
                color=[COLORS[k % 8] for k in range(len(ms))],
            )
            axes[i, j].set_xticks(
                np.arange(len(ms)),
                [m.replace("lgbm_", "").replace(f"_{tgt}", "") for m in ms],
                rotation=35,
                ha="right",
                fontsize=8,
            )
            axes[i, j].set(
                ylabel="MAE at +60 min (mg/dL), 95% CI",
                title=f"{coh}: ablation (LightGBM, {tgt} target)",
            )
    fig.tight_layout()
    fig.savefig(FIG / "fusion_ablation.png", dpi=dpi)
    plt.close(fig)


def logo(results: dict[str, Any], fc: dict[str, Any], dpi: int) -> None:
    """Paired MAE change when each group is removed from E (delta target)."""
    cohorts = list(results)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(6.5 * len(cohorts), 4.5), squeeze=False)
    for i, coh in enumerate(cohorts):
        p = pd.DataFrame(results[coh]["logo_paired"])
        p = p[(p.subset == "overall") & p.model_1.str.endswith("_delta")]
        names = list(fc["logo"][coh])
        for k, h in enumerate(HORIZONS):
            d = p[p.horizon_min == h].set_index("model_1").loc[[f"lgbm_{n}_delta" for n in names]]
            v = d.diff_mae.to_numpy()
            ci = np.array(d.diff_mae_ci.tolist())
            y = np.arange(len(names)) + (k - 1) * 0.25
            axes[0, i].errorbar(
                v,
                y,
                xerr=[v - ci[:, 0], ci[:, 1] - v],
                fmt="o",
                capsize=2,
                color=COLORS[k + 1],
                label=f"+{h} min",
            )
        axes[0, i].axvline(0, color="#999999", lw=0.8)
        axes[0, i].set_yticks(
            np.arange(len(names)), [n.replace("E_minus_", "minus ") for n in names]
        )
        axes[0, i].set(
            xlabel="MAE change when group removed (mg/dL), paired 95% CI\n"
            "(> 0 = group carries information)",
            title=f"{coh}: leave-one-group-out from E (delta)",
        )
        axes[0, i].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fusion_leave_one_group_out.png", dpi=dpi)
    plt.close(fig)


def tiers(results: dict[str, Any], fc: dict[str, Any], dpi: int) -> None:
    """Tier degradation: MAE vs horizon per tier (delta target) + fingerstick persistence."""
    cohorts = list(results)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(6 * len(cohorts), 4), squeeze=False)
    for i, coh in enumerate(cohorts):
        t = pd.DataFrame(results[coh]["tiers"])
        t = t[t.subset == "overall"]
        ms = ["lgbm_E_full_delta"] + [
            f"lgbm_{n}_delta" for n, s in fc["tiers"][coh].items() if "fingerstick" in s
        ]
        ms += [f"fs_persist_{s}" for s in fc["fingerstick"]["schedules"]]
        for k, m in enumerate(ms):
            d = t[t.model == m].sort_values("horizon_min")
            if d.empty:
                continue
            lab = (
                "T1 full fusion"
                if m == "lgbm_E_full_delta"
                else m.replace("lgbm_", "")
                .replace("_delta", "")
                .replace("fs_persist_", "last fingerstick only, ")
            )
            axes[0, i].errorbar(
                d.horizon_min,
                d.mae,
                yerr=_err(d),
                marker="o",
                capsize=2,
                color=COLORS[k % 8],
                label=lab,
            )
        axes[0, i].set(
            xlabel="Forecast horizon (min)",
            ylabel="MAE (mg/dL), 95% CI",
            title=f"{coh}: input tiers (delta target)",
            xticks=HORIZONS,
        )
        axes[0, i].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fusion_tiers.png", dpi=dpi)
    plt.close(fig)


def post_meal(results: dict[str, Any], fc: dict[str, Any], dpi: int) -> None:
    """CGMacros: MAE overall vs post-meal (3 h) vs rising-onset windows, delta target."""
    t = pd.DataFrame(results["cgmacros"]["ablation"])
    ms = ["persistence", "ar_per_patient"] + [f"lgbm_{n}_delta" for n in fc["ablation"]["cgmacros"]]
    subs = ("overall", "post_meal_3h", "onset_hyper140")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k, h in enumerate(HORIZONS):
        for j, sub in enumerate(subs):
            d = t[(t.subset == sub) & (t.horizon_min == h)].set_index("model")
            d = d.loc[[m for m in ms if m in d.index]]
            axes[k].bar(
                np.arange(len(d)) + (j - 1) * 0.27,
                d.mae,
                width=0.27,
                yerr=_err(d),
                capsize=1.5,
                color=COLORS[j + 1],
                label=sub,
            )
        axes[k].set_xticks(
            np.arange(len(ms)),
            [m.replace("lgbm_", "").replace("_delta", "") for m in ms],
            rotation=35,
            ha="right",
            fontsize=7,
        )
        axes[k].set(ylabel="MAE (mg/dL), 95% CI", title=f"CGMacros +{h} min")
        axes[k].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fusion_post_meal_vs_overall.png", dpi=dpi)
    plt.close(fig)


def seed_variance(results: dict[str, Any], dpi: int) -> None:
    """Neural MAE per seed (CV repeat 0) vs LightGBM E (delta) on the same rows."""
    cohorts = list(results)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(6 * len(cohorts), 4), squeeze=False)
    for i, coh in enumerate(cohorts):
        sr = pd.DataFrame(results[coh]["seed_variance"])
        for k, (m, g) in enumerate(sr.groupby("model")):
            for _, r in g.iterrows():
                xs = np.full(len(r.seed_mae), r.horizon_min) + (k - 0.5) * 6
                axes[0, i].scatter(
                    xs,
                    r.seed_mae,
                    s=16,
                    color=COLORS[k + 5],
                    label=m if r.horizon_min == 30 else None,
                )
        fam = pd.DataFrame(results[coh]["families_r0"])
        for m, ls in (("lgbm_E_full_delta", "--"), ("lgbm_B_cgm_delta", ":")):
            d = fam[(fam.subset == "overall") & (fam.model == m)].sort_values("horizon_min")
            axes[0, i].plot(d.horizon_min, d.mae, ls, color="#000000", lw=1, label=m)
        axes[0, i].set(
            xlabel="Forecast horizon (min)",
            ylabel="MAE (mg/dL), CV repeat 0",
            title=f"{coh}: GRU+FiLM, one point per seed",
            xticks=HORIZONS,
        )
        axes[0, i].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fusion_neural_seed_variance.png", dpi=dpi)
    plt.close(fig)


def importance(cohorts: list[str], dpi: int) -> None:
    """LightGBM gain and mean |SHAP| (E, delta, +60 min), top 20 features per cohort."""
    proc = PROJECT_ROOT / "data" / "processed" / "fusion"
    fig, axes = plt.subplots(len(cohorts), 2, figsize=(12, 6 * len(cohorts)), squeeze=False)
    for i, coh in enumerate(cohorts):
        imp = pd.read_parquet(proc / f"shap_{coh}.parquet")
        imp["gain_pct"] = 100 * imp.gain / imp.gain.sum()
        for j, (col, lab) in enumerate(
            (
                ("gain_pct", "LightGBM gain (% of total)"),
                ("mean_abs_shap", "mean |SHAP| (mg/dL of delta)"),
            )
        ):
            d = imp.sort_values(col).tail(20)
            # One distinct (colour, hatch) per group shown, so no two groups share a style.
            groups = sorted(d.group.dropna().unique())
            style = {g: (COLORS[k % 8], "//" if k >= 8 else "") for k, g in enumerate(groups)}
            bars = axes[i, j].barh(
                d.feature, d[col], color=[style.get(g, ("#999999", ""))[0] for g in d.group]
            )
            for b, g in zip(bars, d.group, strict=True):
                b.set_hatch(style.get(g, ("", ""))[1])
            axes[i, j].set(xlabel=lab, title=f"{coh}: E full fusion, +60 min, top 20")
            axes[i, j].tick_params(axis="y", labelsize=7)
            handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor=style[g][0], hatch=style[g][1])
                for g in groups
            ]
            axes[i, j].legend(handles, groups, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fusion_feature_importance.png", dpi=dpi)
    plt.close(fig)
    import shap

    for coh in cohorts:
        sv = np.load(proc / f"shap_values_{coh}.npy")
        x = pd.read_parquet(proc / f"shap_x_{coh}.parquet")
        plt.figure()
        shap.summary_plot(sv, x, max_display=20, show=False)
        plt.gcf().axes[0].set_xlabel("SHAP value (mg/dL change in predicted delta, +60 min)")
        plt.title(f"{coh}: SHAP summary, E full fusion")
        plt.tight_layout()
        plt.savefig(FIG / f"fusion_shap_summary_{coh}.png", dpi=dpi)
        plt.close("all")


def all_figures(results: dict[str, Any], fc: dict[str, Any], dpi: int) -> None:
    """Write every Part 3 figure."""
    ablation(results, fc, dpi)
    logo(results, fc, dpi)
    tiers(results, fc, dpi)
    if "cgmacros" in results:
        post_meal(results, fc, dpi)
    seed_variance(results, dpi)
    importance(list(results), dpi)
