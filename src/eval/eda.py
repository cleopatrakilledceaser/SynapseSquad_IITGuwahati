"""EDA figures + threshold statistics for Part 1 Step D (``make figures``).

All figures: 300 dpi, labeled axes with units, Okabe-Ito colorblind-safe palette, written to
reports/figures/<name>.png. Statistics go to reports/metrics/eda_summary.json.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.cgm_reconstruct import detect_native  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402
from src.utils.provenance import file_hash, write_metrics  # noqa: E402

OKABE_ITO = {
    "healthy": "#009E73",
    "prediabetes": "#E69F00",
    "t2d": "#D55E00",
    "shanghai_t2d": "#0072B2",
    "grey": "#999999",
    "black": "#000000",
    "sky": "#56B4E9",
}
GROUPS = ("healthy", "prediabetes", "t2d", "shanghai_t2d")
LABELS = {
    "healthy": "CGMacros healthy",
    "prediabetes": "CGMacros prediabetes",
    "t2d": "CGMacros T2D",
    "shanghai_t2d": "Shanghai T2D",
}
PROC = PROJECT_ROOT / "data" / "processed"
FIG = PROJECT_ROOT / "reports" / "figures"


def _save(fig: plt.Figure, name: str, dpi: int) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=dpi)
    plt.close(fig)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load harmonized tables and add a ``group`` column (status x cohort)."""
    static = pd.read_parquet(PROC / "static.parquet")
    ts5 = pd.read_parquet(PROC / "cgmacros_ts5.parquet")
    ts15 = pd.read_parquet(PROC / "common_ts15.parquet")
    meals = pd.read_parquet(PROC / "meals.parquet")
    grp = static.set_index("patient_id").apply(
        lambda r: "shanghai_t2d" if r.dataset == "shanghai_t2dm" else r.diabetes_status, axis=1
    )
    for df in (ts5, ts15, meals):
        df["group"] = df["patient_id"].map(grp)
    return static, ts5, ts15, meals


def real_glucose(ts5: pd.DataFrame, ts15: pd.DataFrame) -> pd.DataFrame:
    """Real readings only: CGMacros on its native 5-min grid, Shanghai on its native 15-min grid."""
    a = ts5.loc[ts5.glucose_is_real, ["patient_id", "group", "glucose"]]
    b = ts15.loc[(ts15.dataset == "shanghai_t2dm") & ts15.glucose_is_real]
    return pd.concat([a, b[["patient_id", "group", "glucose"]]], ignore_index=True)


def band_table(g: pd.DataFrame, clin: dict) -> pd.DataFrame:
    """% time per glucose band; patient-weighted (mean of per-patient %), not row-weighted."""
    lo, sec, hi = clin["hypo_mgdl"], clin["secondary_hyper_mgdl"], clin["hyper_mgdl"]
    bands = {
        f"<{lo}": g.glucose < lo,
        f"{lo}-{sec}": (g.glucose >= lo) & (g.glucose <= sec),
        f"{sec}-{hi}": (g.glucose > sec) & (g.glucose <= hi),
        f">{hi}": g.glucose > hi,
    }
    per_pat = (
        pd.DataFrame({k: v for k, v in bands.items()}).groupby([g.group, g.patient_id]).mean() * 100
    )
    return per_pat.groupby(level=0).mean().reindex(list(GROUPS)).round(2)


def fig_distribution(g: pd.DataFrame, clin: dict, dpi: int) -> dict:
    """Glucose density by group with threshold lines; returns percentiles per group."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(40, 410, 5)
    stats = {}
    for grp in GROUPS:
        x = g.loc[g.group == grp, "glucose"]
        ax.hist(
            x,
            bins=bins,
            density=True,
            histtype="step",
            lw=1.8,
            color=OKABE_ITO[grp],
            label=f"{LABELS[grp]} (n={g.loc[g.group == grp, 'patient_id'].nunique()})",
        )
        stats[grp] = x.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).round(1).to_dict()
    for v, ls in (
        (clin["hypo_mgdl"], ":"),
        (clin["secondary_hyper_mgdl"], "--"),
        (clin["hyper_mgdl"], "-."),
    ):
        ax.axvline(v, color=OKABE_ITO["black"], ls=ls, lw=1)
        ax.text(v + 2, ax.get_ylim()[1] * 0.92, f"{v}", fontsize=8)
    ax.set(
        xlabel="Glucose (mg/dL), real CGM readings only",
        ylabel="Density (1/(mg/dL))",
        title="Glucose distribution by cohort and diabetes status",
    )
    ax.legend(fontsize=8)
    _save(fig, "glucose_distribution", dpi)
    return stats


def fig_tir(tab: pd.DataFrame, dpi: int) -> None:
    """Stacked time-in-range bars (patient-weighted)."""
    colors = [OKABE_ITO["sky"], OKABE_ITO["healthy"], OKABE_ITO["prediabetes"], OKABE_ITO["t2d"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    left = np.zeros(len(tab))
    for col, c in zip(tab.columns, colors, strict=True):
        ax.barh([LABELS[i] for i in tab.index], tab[col], left=left, color=c, label=f"{col} mg/dL")
        left += tab[col].to_numpy()
    ax.set(
        xlabel="Time in band (% of real readings, mean over patients)",
        ylabel="",
        title="Time in glucose bands",
    )
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, "time_in_range", dpi)


def fig_meal_responses(ts5: pd.DataFrame, meals: pd.DataFrame, dpi: int) -> None:
    """Mean real-glucose change -30..+180 min around logged CGMacros breakfasts, per group."""
    fig, ax = plt.subplots(figsize=(7, 4))
    offs = np.arange(-30, 181, 5)
    for grp in ("healthy", "prediabetes", "t2d"):
        traces = []
        m = meals[(meals.group == grp) & (meals.meal_type == "breakfast")]
        for pid, mm in m.groupby("patient_id"):
            d = ts5[ts5.patient_id == pid].set_index("t")
            s = d["glucose"].where(d["glucose_is_real"])
            for t0 in mm.t:
                y = s.reindex(t0.ceil("5min") + pd.to_timedelta(offs, unit="min")).to_numpy()
                if np.isnan(y).mean() < 0.2 and np.isfinite(y[:7]).any():
                    traces.append(y - np.nanmean(y[:7]))
        arr = np.array(traces)
        mu = np.nanmean(arr, axis=0)
        se = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
        ax.plot(offs, mu, color=OKABE_ITO[grp], label=f"{LABELS[grp]} ({len(arr)} breakfasts)")
        ax.fill_between(offs, mu - 1.96 * se, mu + 1.96 * se, color=OKABE_ITO[grp], alpha=0.2)
    ax.axvline(0, color=OKABE_ITO["grey"], lw=1)
    ax.set(
        xlabel="Minutes from logged breakfast start (min)",
        ylabel="Glucose change from pre-meal baseline (mg/dL)",
        title="Mean post-breakfast response (shaded: 95% CI of the mean)",
    )
    ax.legend(fontsize=8)
    _save(fig, "meal_responses", dpi)


def fig_missingness(ts5: pd.DataFrame, dpi: int) -> None:
    """Per-patient hourly map of CGM state (worst state in each hour) on the 5-min grid."""
    from matplotlib.colors import ListedColormap

    pids = sorted(ts5.patient_id.unique())
    hours = int(np.ceil(ts5.days_since_start.max() * 24)) + 1
    img = np.full((len(pids), hours), np.nan)
    for i, pid in enumerate(pids):
        d = ts5[ts5.patient_id == pid]
        h = (d.days_since_start * 24).astype(int).to_numpy()
        state = np.where(d.glucose_missing, 0, np.where(d.glucose_is_real, 2, 1))
        img[i, : h.max() + 1] = pd.Series(state).groupby(h).min().reindex(range(h.max() + 1))
    cmap = ListedColormap([OKABE_ITO["t2d"], OKABE_ITO["prediabetes"], OKABE_ITO["sky"]])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax.set_yticks(range(len(pids)), [p.split("_")[1] for p in pids], fontsize=6)
    ax.set(
        xlabel="Hours since start of recording (h)",
        ylabel="CGMacros participant",
        title="Dexcom CGM availability per hour (worst state in the hour)",
    )
    cb = fig.colorbar(im, ticks=[0, 1, 2])
    cb.ax.set_yticklabels(["missing", "carried <=30 min", "all real"])
    _save(fig, "missingness_map", dpi)


def fig_days_and_wearables(ts5: pd.DataFrame, ts15: pd.DataFrame, dpi: int) -> None:
    """Days per patient (both cohorts) and wearable coverage (CGMacros)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    d5 = ts5.groupby(["patient_id", "group"]).days_since_start.max().reset_index()
    sh = ts15[ts15.dataset == "shanghai_t2dm"]
    dsh = (
        sh.groupby(["patient_id", "period_id"])
        .days_since_start.max()
        .groupby("patient_id")
        .sum()
        .reset_index()
        .assign(group="shanghai_t2d")
    )
    alld = pd.concat([d5, dsh])
    rng = np.random.default_rng(42)
    for i, grp in enumerate(GROUPS):
        x = alld.loc[alld.group == grp, "days_since_start"]
        axes[0].scatter(i + rng.uniform(-0.15, 0.15, len(x)), x, s=12, color=OKABE_ITO[grp])
    axes[0].set_xticks(range(len(GROUPS)), [LABELS[g].replace(" ", "\n", 1) for g in GROUPS])
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].set(ylabel="Recorded days per patient (days)", title="Days per patient")
    cov = (
        ts5.groupby(["patient_id", "group"])
        .agg(
            hr=("hr_missing", lambda s: 100 * (1 - s.mean())),
            act=("activity_missing", lambda s: 100 * (1 - s.mean())),
        )
        .reset_index()
        .sort_values("hr")
    )
    y = np.arange(len(cov))
    axes[1].barh(y, cov.hr, color=[OKABE_ITO[g] for g in cov.group], height=0.8)
    axes[1].scatter(cov.act, y, marker="|", color=OKABE_ITO["black"], s=30, label="activity")
    axes[1].set_yticks(y, [p.split("_")[1] for p in cov.patient_id], fontsize=5)
    axes[1].set(
        xlabel="Grid points with data (%)",
        ylabel="CGMacros participant",
        title="Wearable coverage (bars: heart rate, colour = status)",
    )
    axes[1].legend(fontsize=7, loc="lower right")
    _save(fig, "days_and_wearable_coverage", dpi)


def fig_reconstruction(cfg: dict, pid: str, dpi: int) -> dict:
    """Before/after: raw interpolated Dexcom vs recovered real readings on the 5-min grid."""
    c = cfg["cgm"]
    z = zipfile.ZipFile(PROJECT_ROOT / cfg["paths"]["cgmacros_zip"])
    name = next(n for n in z.namelist() if n.endswith(f"CGMacros-{pid}.csv") and "MACOSX" not in n)
    raw = pd.read_csv(io.BytesIO(z.read(name)), usecols=["Timestamp", "Dexcom GL"])
    t = pd.to_datetime(raw.Timestamp)
    m = ((t - t.iloc[0]).dt.total_seconds() // 60).astype(int).to_numpy()
    v = pd.to_numeric(raw["Dexcom GL"], errors="coerce").to_numpy()
    nat = detect_native(
        m, v, c["dexcom_period_min"], c["max_gap_min"], c["kink_tol"], c["integer_tol"]
    )
    rt = t[nat].reset_index(drop=True)
    k = int(rt.diff().idxmax())
    g0, g1 = rt.iloc[k - 1], rt.iloc[k]
    lo, hi = g0 - pd.Timedelta(hours=6), g1 + pd.Timedelta(hours=6)

    def days(x: pd.Series) -> pd.Series:
        return (x - t.iloc[0]).dt.total_seconds() / 86400

    w = ((t >= lo) & (t <= hi)).to_numpy()
    gap_h = (g1 - g0).total_seconds() / 3600
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    axes[0].plot(days(t[w]), v[w], color=OKABE_ITO["grey"], lw=1)
    axes[0].axvspan(
        *days(pd.Series([g0, g1])),
        color=OKABE_ITO["t2d"],
        alpha=0.15,
        label=f"bridged span: {gap_h:.1f} h with no real reading",
    )
    axes[0].set(
        ylabel="Glucose (mg/dL)",
        title=f"Before: raw file, participant {pid} (1-min rows, linearly interpolated)",
    )
    axes[0].legend(fontsize=8)
    ts5 = pd.read_parquet(PROC / "cgmacros_ts5.parquet")
    gw = ts5[(ts5.patient_id == f"cgmacros_{pid}") & (ts5.t >= lo) & (ts5.t <= hi)]
    axes[1].plot(
        days(gw.t),
        gw.glucose,
        color=OKABE_ITO["sky"],
        lw=1,
        label="5-min grid (carry <= 30 min, else missing)",
    )
    rr = gw[gw.glucose_is_real]
    axes[1].scatter(
        days(rr.t), rr.glucose, s=3, color=OKABE_ITO["black"], label="recovered real readings"
    )
    axes[1].set(
        xlabel=f"Days since start of participant {pid} recording (days)",
        ylabel="Glucose (mg/dL)",
        title="After: the gap stays MISSING (nothing invented)",
    )
    axes[1].legend(fontsize=8)
    _save(fig, "cgm_reconstruction_before_after", dpi)
    return {"pid": pid, "largest_gap_hours": round(gap_h, 1)}


def main() -> None:
    """Generate all Part 1 Step D figures and the EDA summary metrics."""
    cfg = load_config("data")
    dpi = load_config("base")["figures"]["dpi"]
    _, ts5, ts15, meals = load()
    g = real_glucose(ts5, ts15)
    stats = fig_distribution(g, cfg["clinical"], dpi)
    tab = band_table(g, cfg["clinical"])
    fig_tir(tab, dpi)
    fig_meal_responses(ts5, meals, dpi)
    fig_missingness(ts5, dpi)
    fig_days_and_wearables(ts5, ts15, dpi)
    rec = fig_reconstruction(cfg, "018", dpi)
    print(tab.to_string())
    print("percentiles:", stats)
    write_metrics(
        "eda_summary",
        {
            "glucose_percentiles": stats,
            "time_in_bands_pct": tab.reset_index().to_dict("records"),
            "reconstruction_example": rec,
        },
        cfg,
        cfg["seed"],
        file_hash(sorted(PROC.glob("*.parquet"))),
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "Path"]
