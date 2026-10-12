"""Part 2 evaluation (second stage of ``make models``).

Metrics with patient-bootstrap CIs, tables, per-patient metrics and figures, computed from the
predictions in data/processed/predictions/.

Headline numbers = repeated patient-grouped CV: each repeat gives one out-of-fold prediction per
anchor; metrics are computed per repeat and the bootstrap pools patients' sufficient statistics
averaged over repeats (so each patient counts once). Holdout-test numbers are reported separately.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.eval.metrics import (
    bootstrap_event,
    bootstrap_point,
    per_patient_point,
    point_stats,
    reliability,
)
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.provenance import file_hash, write_metrics

PROC = PROJECT_ROOT / "data" / "processed"
PRED = PROC / "predictions"
METRICS = PROJECT_ROOT / "reports" / "metrics"
MODELS = [
    "persistence",
    "linear_trend",
    "exp_smoothing",
    "ar_pooled",
    "ar_per_patient",
    "lgbm_cgm",
    "lgbm_cgm_tod",
]


def pooled_stats(oof: pd.DataFrame, model: str, h: int) -> pd.DataFrame:
    """Per-patient sufficient statistics averaged over CV repeats (each patient counts once)."""
    per_rep = [point_stats(g, f"y_{h}", f"pred_{model}_{h}") for _, g in oof.groupby("repeat")]
    s = pd.concat(per_rep).groupby(level=0).sum()
    return s / oof["repeat"].nunique()


def regression_table(
    oof: pd.DataFrame, horizons: list[int], bcfg: dict[str, Any], exclude: list[str]
) -> list[dict[str, Any]]:
    """MAE/RMSE/MARD/Clarke with CIs per model and horizon (optionally excluding patients)."""
    b = bcfg["bootstrap"]
    d = oof[~oof.patient_id.isin(exclude)]
    rows = []
    for h in horizons:
        for m in MODELS:
            st = pooled_stats(d, m, h)
            r = bootstrap_point(st, b["n_boot"], 42, b["ci"])
            rows.append({"model": m, "horizon_min": h, **r})
    return rows


def event_table(
    oof: pd.DataFrame, cohort: str, horizons: list[int], bcfg: dict[str, Any], exclude: list[str]
) -> list[dict[str, Any]]:
    """AUROC/AUPRC/Brier per event, definition, horizon and scorer, with reliability flags."""
    b, e = bcfg["bootstrap"], bcfg["events"]
    d = oof[(oof["repeat"] == 0) & ~oof.patient_id.isin(exclude)]
    never = set(e["never_report"].get(cohort, []))
    rows = []
    for h in horizons:
        for name in ("hyper180", "hyper140", "hypo70"):
            for defn in e["definitions"]:
                lab = f"{name}_{defn}_{h}"
                scorers = {
                    "persistence_value": f"score_persistence_{lab}",
                    "lgbm_reg_value": f"score_lgbm_cgm_tod_reg_{lab}",
                    "lgbm_classifier": f"prob_lgbm_cgm_tod_{lab}",
                }
                for sname, col in scorers.items():
                    if col not in d:
                        continue
                    r = bootstrap_event(d, lab, col, b["n_boot"], 42, b["ci"], b["auc_bins"])
                    if sname != "lgbm_classifier":
                        r["brier"], r["brier_ci"] = None, None  # value scores are not probabilities
                    rel = reliability(r, e["min_pos_windows"], e["min_pos_patients"])
                    if name in never:
                        rel = "not reported (owner policy)"
                    rows.append(
                        {
                            "event": name,
                            "definition": defn,
                            "horizon_min": h,
                            "scorer": sname,
                            "reliability": rel,
                            **r,
                        }
                    )
    return rows


def group_table(
    oof: pd.DataFrame, static: pd.DataFrame, cohort: str, h: int, bcfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """MAE by diabetes group (CGMacros) or HbA1c group (both), patient-bootstrap CIs."""
    s = static.set_index("patient_id")
    gcfg = bcfg["groups"]
    grp = {
        "diabetes_status": s["diabetes_status"],
        "hba1c_group": pd.cut(
            s["hba1c_pct"], gcfg["hba1c_bins_pct"], labels=gcfg["hba1c_labels"], right=False
        ).astype(str),
    }
    rows = []
    for gname, gmap in grp.items():
        if cohort == "shanghai_t2dm" and gname == "diabetes_status":
            continue
        for gval in sorted(gmap.dropna().unique()):
            pids = set(gmap[gmap == gval].index)
            d = oof[oof.patient_id.isin(pids)]
            if d.empty:
                continue
            for m in ("persistence", "ar_per_patient", "lgbm_cgm_tod"):
                st = pooled_stats(d, m, h)
                r = bootstrap_point(st, bcfg["bootstrap"]["n_boot"], 42, bcfg["bootstrap"]["ci"])
                rows.append(
                    {
                        "grouping": gname,
                        "group": gval,
                        "model": m,
                        "horizon_min": h,
                        "n_patients": r["n_patients"],
                        "mae": r["mae"],
                        "mae_ci": r["mae_ci"],
                    }
                )
    return rows


def per_patient(oof: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Per-patient metrics for every model and horizon (averaged over repeats)."""
    parts = []
    for h in horizons:
        for m in MODELS:
            pp = per_patient_point(pooled_stats(oof, m, h))
            parts.append(pp.assign(model=m, horizon_min=h).reset_index())
    return pd.concat(parts, ignore_index=True)


def figures(
    results: dict[str, Any],
    pp: dict[str, pd.DataFrame],
    oof: dict[str, pd.DataFrame],
    horizons: list[int],
    dpi: int,
) -> None:
    """Error vs horizon, Clarke grid (LightGBM, 60 min), per-patient MAE spread."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.eval.eda import OKABE_ITO
    from src.eval.metrics import clarke_zones

    colors = dict(
        zip(
            MODELS,
            ["#000000", "#999999", "#56B4E9", "#E69F00", "#D55E00", "#009E73", "#0072B2"],
            strict=True,
        )
    )
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    for ax, (coh, r) in zip(np.atleast_1d(axes), results.items(), strict=True):
        t = pd.DataFrame(r["cv_regression"])
        for m in MODELS:
            d = t[t.model == m].sort_values("horizon_min")
            lo = [c[0] for c in d.mae_ci]
            hi = [c[1] for c in d.mae_ci]
            ax.errorbar(
                d.horizon_min + (MODELS.index(m) - 3) * 1.2,
                d.mae,
                yerr=[d.mae - lo, np.array(hi) - d.mae],
                marker="o",
                ms=4,
                capsize=2,
                color=colors[m],
                label=m,
            )
        ax.set(
            xlabel="Forecast horizon (min)",
            ylabel="MAE (mg/dL), 95% patient-bootstrap CI",
            title=f"{coh}: error vs horizon (repeated CV)",
            xticks=horizons,
        )
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "reports/figures/baselines_error_vs_horizon.png", dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(oof), figsize=(5 * len(oof), 5))
    for ax, (coh, d) in zip(np.atleast_1d(axes), oof.items(), strict=True):
        d0 = d[d.repeat == 0].dropna(subset=["y_60"])
        z = clarke_zones(d0["y_60"].to_numpy(), d0["pred_lgbm_cgm_tod_60"].to_numpy())
        for zone, c in zip(
            "ABCDE", ["#009E73", "#56B4E9", "#E69F00", "#D55E00", "#000000"], strict=True
        ):
            k = z == zone
            ax.scatter(
                d0["y_60"][k],
                d0["pred_lgbm_cgm_tod_60"][k],
                s=1,
                alpha=0.3,
                color=c,
                label=f"{zone}: {100 * k.mean():.1f}%",
            )
        ax.plot([0, 400], [0, 400], color="#999999", lw=0.8)
        ax.set(
            xlim=(0, 420),
            ylim=(0, 420),
            xlabel="Reference glucose at +60 min (mg/dL)",
            ylabel="Predicted glucose (mg/dL)",
            title=f"{coh}: Clarke grid, LightGBM, +60 min",
        )
        ax.legend(fontsize=8, markerscale=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "reports/figures/baselines_clarke_grid.png", dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(pp), figsize=(5 * len(pp), 4), sharey=True)
    for ax, (coh, p) in zip(np.atleast_1d(axes), pp.items(), strict=True):
        sel = p[
            (p.horizon_min == 60) & p.model.isin(["persistence", "ar_per_patient", "lgbm_cgm_tod"])
        ]
        data = [
            sel[sel.model == m].mae.to_numpy()
            for m in ("persistence", "ar_per_patient", "lgbm_cgm_tod")
        ]
        ax.boxplot(data, tick_labels=["persistence", "AR per-patient", "LightGBM CGM+ToD"])
        for i, v in enumerate(data):
            ax.scatter(
                np.full(len(v), i + 1) + np.random.default_rng(42).uniform(-0.1, 0.1, len(v)),
                v,
                s=8,
                color=OKABE_ITO["sky"],
                alpha=0.7,
            )
        ax.set(
            ylabel="Per-patient MAE at +60 min (mg/dL)", title=f"{coh}: per-patient error spread"
        )
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "reports/figures/baselines_per_patient_spread.png", dpi=dpi)
    plt.close(fig)


def figures_only() -> None:
    """Regenerate figures from saved metrics / predictions (no bootstrap recomputation)."""
    bcfg, fcfg = load_config("baselines"), load_config("features")
    results, pp, oof = {}, {}, {}
    for coh in bcfg["cohorts"]:
        results[coh] = json.loads((METRICS / f"baselines_{coh}.json").read_text())["metrics"]
        pp[coh] = pd.read_csv(METRICS / f"baselines_per_patient_{coh}.csv")
        oof[coh] = pd.read_parquet(PRED / f"{coh}_cv_oof.parquet")
    figures(
        results, pp, oof, fcfg["windows"]["horizons_min"], load_config("base")["figures"]["dpi"]
    )


def main() -> None:
    """Compute and write all Part 2 metrics, per-patient files and figures."""
    import sys

    if "--figures-only" in sys.argv:
        figures_only()
        return
    bcfg, fcfg = load_config("baselines"), load_config("features")
    horizons = fcfg["windows"]["horizons_min"]
    static = pd.read_parquet(PROC / "static.parquet")
    summary = json.loads((PRED / "run_summary.json").read_text())
    results, pp_all, oof_all = {}, {}, {}
    for coh in bcfg["cohorts"]:
        oof = pd.read_parquet(PRED / f"{coh}_cv_oof.parquet")
        hold = pd.read_parquet(PRED / f"{coh}_holdout_test.parquet").assign(repeat=0)
        excl = bcfg["sensitivity_exclude"].get(coh, [])
        r = {
            "N_hours": summary[coh]["N"],
            "window_selection": summary[coh]["window_selection"],
            "cv_regression": regression_table(oof, horizons, bcfg, []),
            "cv_events": event_table(oof, coh, horizons, bcfg, []),
            "cv_groups_60min": group_table(oof, static, coh, 60, bcfg),
            "holdout_test_regression": regression_table(hold, horizons, bcfg, []),
        }
        if excl:
            r["cv_regression_without_" + "_".join(x.split("_")[1] for x in excl)] = (
                regression_table(oof, horizons, bcfg, excl)
            )
            r["cv_events_without_" + "_".join(x.split("_")[1] for x in excl)] = event_table(
                oof, coh, horizons, bcfg, excl
            )
        pp = per_patient(oof, horizons)
        pp.to_csv(METRICS / f"baselines_per_patient_{coh}.csv", index=False)
        results[coh], pp_all[coh], oof_all[coh] = r, pp, oof
        write_metrics(
            f"baselines_{coh}",
            r,
            {"baselines": bcfg, "features": fcfg},
            42,
            file_hash(sorted((PROC / "features").glob(f"{coh}_N*.parquet"))),
        )
    write_metrics(
        "baselines_checks",
        {"shuffle_check": summary["shuffle_check"]},
        {"baselines": bcfg},
        42,
        file_hash([PRED / "run_summary.json"]),
    )
    figures(results, pp_all, oof_all, horizons, load_config("base")["figures"]["dpi"])
    print("written: reports/metrics/baselines_*.json, figures")


if __name__ == "__main__":
    main()
