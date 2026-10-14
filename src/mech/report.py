"""Part 4 evaluation: identical test windows (day >= 3), paired patient-bootstrap CIs.

Reuses the Part 3 row filter / bootstrap code (src.eval.fusion_report): per horizon, rows where
every compared model has a prediction; CIs resample PATIENTS (1,000 draws). Models:
  persistence, lgbm (population: none of the patient's own days), ar_k1 / ar_k3 (per-patient
  AR refit on exactly the first k days), twin_<init>_act<on|off>_k<k>.
At k = 0 only persistence, LightGBM and the twin are compared (owner rule).
Coverage of 80/95% Gaussian predictive intervals (twin only): per-patient coverage averaged with
patient-bootstrap CIs.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.eval.fusion_report import paired_rows, regression_rows
from src.features.fusion_data import load_table
from src.mech.model import FIT_PARAMS
from src.utils.config import PROJECT_ROOT, load_config

PROC = PROJECT_ROOT / "data" / "processed"
MECH = PROC / "mech"
HORIZONS = [30, 60, 120]


def add_twins(w: pd.DataFrame, cohort: str, tag: str, prefix: str) -> pd.DataFrame:
    """Merge twin forecasts from twin<tag>_<cohort>_fold*.parquet as columns <prefix><model>|h."""
    key = ["period_id", "t"]
    tw = pd.concat(
        [pd.read_parquet(f) for f in sorted(MECH.glob(f"twin{tag}_{cohort}_fold*.parquet"))],
        ignore_index=True,
    )
    clip = load_config("data")["cgm"]["glucose_range_mgdl"][cohort]
    for m, g in tw.groupby("model"):
        cols = {f"mu_{h}": f"{prefix}{m}|{h}" for h in HORIZONS}
        cols |= {f"var_{h}": f"{prefix}{m}|var{h}" for h in HORIZONS}
        w = w.merge(g[key + list(cols)].rename(columns=cols), on=key, how="left")
        for h in HORIZONS:
            w[f"{prefix}{m}|{h}"] = w[f"{prefix}{m}|{h}"].clip(*clip)
    return w


def revision_eval(cohort: str, tag: str) -> dict[str, Any]:
    """Original vs revised spec (D-037) on IDENTICAL rows: MAE per k and paired differences."""
    mc, fc, bc = load_config("mech"), load_config("fusion"), load_config("baselines")
    w = add_twins(build_wide(cohort, mc, fc), cohort, tag, "rev_")
    ks = mc["adapt_days"] if cohort == "cgmacros" else [mc["shanghai"]["k"]]
    act = "on" if cohort == "cgmacros" else "off"
    out: dict[str, Any] = {}
    for k in ks:
        pairs = [
            (f"rev_twin_{i}_act{act}_k{k}", f"twin_{i}_act{act}_k{k}") for i in ("pop", "static")
        ]
        ms = [m for p in pairs for m in p]
        base = ["persistence", "lgbm"] + ([f"ar_k{k}"] if k >= 1 else [])
        wk = common_rows(w, ms + base)
        out[f"regression_k{k}"] = regression_rows(wk, ms + base, cohort, fc, bc)
        out[f"paired_k{k}"] = paired_rows(wk, pairs, cohort, fc, bc)
    out["coverage"] = coverage_rows(
        w, [f"rev_twin_pop_act{act}_k{max(ks)}"], mc["intervals"], bc["bootstrap"]["n_boot"]
    )
    out["fold_choices"] = [
        json.loads(f.read_text()) for f in sorted(MECH.glob(f"twin{tag}_{cohort}_fold*.json"))
    ]
    return out


def build_wide(cohort: str, mc: dict[str, Any], fc: dict[str, Any]) -> pd.DataFrame:
    """One row per test anchor with targets, subsets and every model's forecast (+ twin var)."""
    tab = load_table(cohort, fc["n_hours"], fc)
    keep = ["patient_id", "period_id", "t", "days_since_start", "cgm_now"]
    keep += [c for c in ("meal_minutes_since_last",) if c in tab]
    keep += [f"y_{h}" for h in HORIZONS] + [f"hyper140_any_{h}" for h in HORIZONS]
    w = tab.loc[tab.days_since_start >= mc["test_from_day"], keep].reset_index(drop=True)
    w["repeat"] = 0
    for h in HORIZONS:
        w[f"persistence|{h}"] = w["cgm_now"]
    key = ["period_id", "t"]
    lg = pd.read_parquet(MECH / f"lgbm_{cohort}.parquet")
    w = w.merge(
        lg[key + [f"pred_{h}" for h in HORIZONS]].rename(
            columns={f"pred_{h}": f"lgbm|{h}" for h in HORIZONS}
        ),
        on=key,
        how="left",
    )
    ar = pd.read_parquet(MECH / f"ar_{cohort}.parquet")
    w = w.merge(ar[key + [c for c in ar.columns if "|" in c]], on=key, how="left")
    tw = pd.concat(
        [pd.read_parquet(f) for f in sorted(MECH.glob(f"twin_{cohort}_fold*.parquet"))],
        ignore_index=True,
    )
    for m, g in tw.groupby("model"):
        cols = {f"mu_{h}": f"{m}|{h}" for h in HORIZONS}
        cols |= {f"var_{h}": f"{m}|var{h}" for h in HORIZONS}
        w = w.merge(g[key + list(cols)].rename(columns=cols), on=key, how="left")
    clip = load_config("data")["cgm"]["glucose_range_mgdl"][cohort]
    for c in [c for c in w.columns if c.startswith("twin_") and "|var" not in c]:
        w[c] = w[c].clip(*clip)
    return w


def model_names(w: pd.DataFrame) -> list[str]:
    """Point-forecast model names present in the wide table."""
    return list(dict.fromkeys(c.split("|")[0] for c in w.columns if "|" in c and "|var" not in c))


def common_rows(w: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Blank all predictions on rows where ANY listed model is missing (identical windows)."""
    w = w.copy()
    for h in HORIZONS:
        ok = np.ones(len(w), bool)
        for m in models:
            ok &= w[f"{m}|{h}"].notna().to_numpy()
        for m in models:
            w.loc[~ok, f"{m}|{h}"] = np.nan
    return w


def coverage_rows(w: pd.DataFrame, twins: list[str], levels: list[float], n_boot: int) -> list:
    """Empirical coverage and mean width of central Gaussian intervals, patient bootstrap."""
    rng = np.random.default_rng(42)
    rows = []
    for m in twins:
        for h in HORIZONS:
            d = w[[f"y_{h}", f"{m}|{h}", f"{m}|var{h}", "patient_id"]].dropna()
            if d.empty:
                continue
            sd = np.sqrt(d[f"{m}|var{h}"].to_numpy())
            err = np.abs(d[f"y_{h}"].to_numpy() - d[f"{m}|{h}"].to_numpy())
            for lv in levels:
                z = norm.ppf(0.5 + lv / 2)
                hit = (
                    pd.Series(err <= z * sd)
                    .groupby(d["patient_id"].to_numpy())
                    .agg(["sum", "count"])
                )
                arr = hit.to_numpy(dtype=float)
                idx = rng.integers(0, len(arr), (n_boot, len(arr)))
                s = arr[idx].sum(1)
                draws = s[:, 0] / s[:, 1]
                rows.append(
                    {
                        "model": m,
                        "horizon_min": h,
                        "level": lv,
                        "coverage": float(arr[:, 0].sum() / arr[:, 1].sum()),
                        "coverage_ci": [
                            float(np.percentile(draws, 2.5)),
                            float(np.percentile(draws, 97.5)),
                        ],
                        "mean_width": float(np.mean(2 * z * sd)),
                        "n_rows": int(len(d)),
                        "n_patients": int(len(arr)),
                    }
                )
    return rows


def k_sets(cohort: str, mc: dict[str, Any], twins: list[str]) -> dict[str, list[str]]:
    """Model sets compared on common rows, per k (owner rule: no AR at k = 0)."""
    out = {}
    ks = mc["adapt_days"] if cohort == "cgmacros" else [mc["shanghai"]["k"]]
    for k in ks:
        tw = [t for t in twins if t.endswith(f"_k{k}")]
        base = ["persistence", "lgbm"] + ([f"ar_k{k}"] if k >= 1 else [])
        out[f"k{k}"] = base + tw
    return out


def pairs_for(cohort: str, k: int, twins: list[str]) -> list[tuple[str, str]]:
    """Paired comparisons at one k."""
    best = [t for t in twins if t.endswith(f"_k{k}")]
    ref = f"twin_pop_act{'on' if cohort == 'cgmacros' else 'off'}_k{k}"
    p = [(ref, "persistence"), (ref, "lgbm")]
    if k >= 1:
        p += [(ref, f"ar_k{k}")]
    if cohort == "cgmacros":
        p += [(f"twin_pop_acton_k{k}", f"twin_pop_actoff_k{k}")]
        p += [(f"twin_static_acton_k{k}", f"twin_pop_acton_k{k}")]
    return [x for x in p if x[0] in best or x[0] in ("lgbm",)]


def evaluate(cohort: str) -> dict[str, Any]:
    """All Part 4 tables for one cohort."""
    mc, fc, bc = load_config("mech"), load_config("fusion"), load_config("baselines")
    w = build_wide(cohort, mc, fc)
    models = model_names(w)
    twins = [m for m in models if m.startswith("twin_")]
    res: dict[str, Any] = {
        "n_test_patients": int(w.patient_id.nunique()),
        "n_test_windows": int(len(w)),
        "test_from_day": mc["test_from_day"],
    }
    for kname, ms in k_sets(cohort, mc, twins).items():
        ms = [m for m in ms if m in models]
        wk = common_rows(w, ms)
        res[f"regression_{kname}"] = regression_rows(wk, ms, cohort, fc, bc)
        k = int(kname[1:])
        res[f"paired_{kname}"] = paired_rows(wk, pairs_for(cohort, k, twins), cohort, fc, bc)
    # k-trajectory of the twin on rows common to ALL k (paired k3 vs k0)
    allk = [m for m in twins if m.startswith("twin_pop_acton") or m.startswith("twin_pop_actoff")]
    if allk:
        wk = common_rows(w, allk)
        res["twin_vs_k"] = regression_rows(wk, allk, cohort, fc, bc)
        kp = [(f"twin_pop_actoff_k{k}", "twin_pop_actoff_k0") for k in (1, 3)]
        kp += [(f"twin_static_actoff_k{k}", f"twin_pop_actoff_k{k}") for k in mc["adapt_days"]]
        kp = [p for p in kp if p[0] in twins and p[1] in twins]
        wk = common_rows(w, sorted({m for p in kp for m in p}))
        res["paired_k_and_prior"] = paired_rows(wk, kp, cohort, fc, bc)
    res["coverage"] = coverage_rows(w, twins, mc["intervals"], bc["bootstrap"]["n_boot"])
    if cohort == "cgmacros":  # activity ablation per activity source (METs vs Fitbit intensity)
        src = pd.read_parquet(
            PROC / "cgmacros_ts5.parquet", columns=["patient_id", "activity_source"]
        )
        src = src.dropna().drop_duplicates("patient_id").set_index("patient_id")["activity_source"]
        res["activity_by_source"] = {}
        for s in sorted(src.unique()):
            ws = w[w.patient_id.map(src) == s]
            pr = [(f"twin_pop_acton_k{k}", f"twin_pop_actoff_k{k}") for k in mc["adapt_days"]]
            ws = common_rows(ws, sorted({m for p in pr for m in p}))
            res["activity_by_source"][s] = {
                "n_patients": int(ws.patient_id.nunique()),
                "paired": paired_rows(ws, pr, cohort, fc, bc),
            }
    choices = [json.loads(f.read_text()) for f in sorted(MECH.glob(f"twin_{cohort}_fold*.json"))]
    res["fold_choices"] = choices
    return res


def noise_from_real(grid: pd.DataFrame) -> dict[str, float]:
    """Sensor-noise parameters from REAL CGMacros readings (D-035).

    Residual = real reading - centred 1-h rolling mean (removes slow physiology). phi = lag-1
    autocorrelation of the residual; white sd from first differences. A rough, documented proxy,
    not a validated sensor-error model.
    """
    r = []
    for _, g in grid.groupby("period_id"):
        y = g["glucose"].where(g["glucose_is_real"])
        r.append((y - y.rolling(13, center=True, min_periods=9).mean()).to_numpy())
    res = np.concatenate(r)
    a, b = res[:-1], res[1:]
    ok = ~np.isnan(a) & ~np.isnan(b)
    phi = float(np.clip(np.corrcoef(a[ok], b[ok])[0, 1], 0.0, 0.95))
    sd = float(np.nanstd(res))
    white = float(np.nanstd(b[ok] - a[ok]) / np.sqrt(2) * np.sqrt(1 - phi))
    return {"white_sd": white, "ar_sd": float(np.sqrt(max(sd**2 - white**2, 1.0))), "phi": phi}


def whatif_traj(
    mc: dict[str, Any], fits: pd.DataFrame, alpha: float
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """SIMULATED 30 vs 60 g meal, with / without a walk, for the median fitted patient."""
    from src.mech.model import bounds_array, initial_state, simulate, theta_from_log

    wc, sc = mc["whatif"], mc["solver"]
    lp = np.log(fits[list(FIT_PARAMS)].median().to_numpy(dtype=float))
    th = theta_from_log(lp, mc["params"]["fixed"], alpha)
    n = int(wc["hours"] * 60 / sc["dt_min"]) + 1
    out = {}
    for c in wc["meals_g"]:
        for walk in (False, True):
            carbs, act = np.zeros(n), np.zeros(n)
            carbs[0] = c
            if walk:
                s = int(wc["walk"]["start_after_meal_min"] / sc["dt_min"])
                e = s + int(wc["walk"]["duration_min"] / sc["dt_min"])
                act[s:e] = wc["walk"]["intensity"]
            x0 = initial_state(float(np.exp(lp[0])), th)
            x = simulate(th, x0, carbs, np.full(n, 0.5), act, sc["dt_min"], sc["n_sub"],
                         bounds_array(sc))  # fmt: skip
            out[f"{c} g meal, {'30-min walk' if walk else 'no walk'}"] = x[:, 2]
    return out, np.arange(n) * sc["dt_min"]


def main() -> None:
    """Evaluate both cohorts, simulator realism, what-if; write metrics + figures."""
    from src.features.fusion_data import load_grid
    from src.mech import figures
    from src.mech.inputs import build_records
    from src.mech.simulate import realism_stats, simulate_cohort
    from src.utils.provenance import file_hash, write_metrics

    mc = load_config("mech")
    files = ["cgmacros_ts5", "common_ts15", "meals", "static"]
    dh = file_hash([PROC / f"{f}.parquet" for f in files])
    res = evaluate("cgmacros")
    if (MECH / "twin_shanghai_t2dm_fold0.parquet").exists():
        res_sh = evaluate("shanghai_t2dm")
        write_metrics("mech_shanghai_t2dm", res_sh, mc, mc["seed"], dh)
        figures.comparison(res_sh, "mech_shanghai_comparison", [mc["shanghai"]["k"]])
    grid = load_grid("cgmacros")
    meals = pd.read_parquet(PROC / "meals.parquet")
    meals = meals[meals.dataset == "cgmacros"]
    w_p = mc["params"]["slow_protein_weight"]
    recs = list(build_records(grid, meals, mc["activity"], w_p).values())
    fits = pd.concat([pd.read_parquet(f) for f in MECH.glob("twinfits_cgmacros_fold*.parquet")])
    fits3 = fits[fits.model == "twin_pop_actoff_k3"]
    alpha = float(np.median([c["alpha_on"] for c in res["fold_choices"]]))
    noise = noise_from_real(grid)
    ts, prm = simulate_cohort(fits3, recs, mc, alpha, noise, mc["seed"])
    syn_dir = PROJECT_ROOT / "data" / "synthetic"
    syn_dir.mkdir(parents=True, exist_ok=True)
    ts.to_parquet(syn_dir / "synthetic_mech_cohort.parquet", index=False)
    prm.to_parquet(syn_dir / "synthetic_mech_params.parquet", index=False)
    real_c = np.zeros(len(grid))
    pid_arr = grid.period_id.to_numpy()
    for r in recs:  # grid is sorted by period and time, like the records
        real_c[np.flatnonzero(pid_arr == r.period_id)] = r.carbs
    g_real = grid["glucose"].where(grid["glucose_is_real"])
    st_real = realism_stats(g_real, grid["patient_id"], pd.Series(real_c))
    st_syn = realism_stats(ts["glucose"], ts["patient_id"], ts["carbs_g"])
    cols = [c for c in st_real.columns if c != "patient_id"]
    q = (0.25, 0.5, 0.75)
    res["simulator"] = {
        "synthetic": True,
        "n_synthetic_patients": int(prm.patient_id.nunique()),
        "alpha_E": alpha,
        "noise": noise,
        "phenotype": mc["simulate"]["phenotype"],
        "real_quartiles": st_real[cols].quantile(q).to_dict(),
        "synthetic_quartiles": st_syn[cols].quantile(q).to_dict(),
    }
    traj, minutes = whatif_traj(mc, fits3, alpha)
    dt = mc["solver"]["dt_min"]
    res["whatif_simulated"] = {
        lab: {
            "peak_mgdl": float(g.max()),
            "min_to_peak": float(minutes[g.argmax()]),
            "min_above_140": float((g > 140).sum() * dt),
        }
        for lab, g in traj.items()
    }
    write_metrics("mech_cgmacros", res, mc, mc["seed"], dh)
    figures.comparison(res, "mech_comparison", mc["adapt_days"])
    figures.error_vs_k(res, "mech_error_vs_k")
    figures.coverage(res, "mech_interval_coverage", "twin_pop_acton_k3")
    ident = json.loads((PROJECT_ROOT / "reports/metrics/mech_identifiability.json").read_text())
    rec_df = pd.read_parquet(MECH / "ident_recovery_synthetic.parquet")
    figures.recovery(rec_df, ident["metrics"], "mech_parameter_recovery")
    figures.profiles(
        pd.read_parquet(MECH / "ident_profile_real.parquet"), "mech_profile_likelihood"
    )
    figures.realism(st_real, st_syn, "mech_simulator_realism")
    figures.whatif(traj, minutes, "mech_whatif_simulated", alpha)
    w = build_wide("cgmacros", mc, load_config("fusion"))
    pids = sorted(w.patient_id.unique())[:3]
    figures.example_forecasts(w, "twin_pop_acton_k3", pids, "mech_example_forecasts")
    print("mech report written", flush=True)


def sg_below(mc: dict[str, Any]) -> dict[str, Any]:
    """Revised fits: fraction with SG below the ORIGINAL lower bound, and median SG (k >= 1)."""
    lo0 = 0.99 * mc["params"]["fit"]["SG"]["lo"]  # original bound, 1% tolerance
    tag = mc["revision"]["tag"]
    out = {}
    for cohort in ("cgmacros", "shanghai_t2dm"):
        files = sorted(MECH.glob(f"twinfits{tag}_{cohort}_fold*.parquet"))
        if not files:
            continue
        f = pd.concat([pd.read_parquet(x) for x in files])
        f = f[f.k > 0]
        out[cohort] = {
            m: {
                "frac_below_orig_lo": float(g["SG"].lt(lo0).mean()),
                "median": float(g.SG.median()),
            }
            for m, g in f.groupby("model")
        }
    return out


def main_rev() -> None:
    """Part 4b (D-037): original vs revised spec -> reports/metrics/mech_revision.json.

    The original Part 4 metrics files are not rewritten. At-bound fractions for both specs are
    merged into mech_identifiability.json (keys at_bound, at_bound_rev) with per-patient fitted
    values in reports/metrics/mech_fitted_params{,_rev}.csv.
    """
    from src.mech.identify import write_at_bound
    from src.mech.run_mech import revised
    from src.utils.provenance import file_hash, write_metrics

    mc = load_config("mech")
    tag = mc["revision"]["tag"]
    files = ["cgmacros_ts5", "common_ts15", "meals", "static"]
    dh = file_hash([PROC / f"{f}.parquet" for f in files])
    res: dict[str, Any] = {"spec": mc["revision"]}
    for cohort in ("cgmacros", "shanghai_t2dm"):
        if (MECH / f"twin{tag}_{cohort}_fold0.parquet").exists():
            res[cohort] = revision_eval(cohort, tag)
    sh = mc["shanghai"]["free"]
    fix = mc["revision"]["fix"]
    write_at_bound({"cgmacros": list(FIT_PARAMS), "shanghai_t2dm": sh})
    write_at_bound(
        {
            "cgmacros": [p for p in FIT_PARAMS if p not in fix],
            "shanghai_t2dm": [p for p in sh if p not in fix],
        },
        tag,
        revised(mc)[0],
    )
    res["sg_below_original_bound"] = sg_below(mc)
    write_metrics("mech_revision", res, mc, mc["seed"], dh)
    print("mech revision report written", flush=True)


if __name__ == "__main__":
    import sys

    main_rev() if len(sys.argv) > 1 and sys.argv[1] == "rev" else main()
