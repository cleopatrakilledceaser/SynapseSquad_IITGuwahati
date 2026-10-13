"""Identifiability of the twin's parameters (Part 4).

1. Synthetic recovery: parameters drawn around the config prior (clipped to bounds), simulated
   OPEN LOOP on real CGMacros meal / activity schedules (first 3 days) + Gaussian noise, refit
   with the same MAP procedure. Reports |log error| median / p90 and corr(true, estimate).
2. Laplace on real patients (3-day fits): contraction 1 - posterior var / prior var per
   parameter, and posterior correlations.
3. Profile likelihood (near-flat prior) for a few real patients: each parameter fixed on a log
   grid, others re-optimized; a profile with max delta(-2 log L) < 1.92 is flat.
Synthetic records are labeled ``_synthetic`` and never mixed with real evaluation data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.fusion_data import load_grid
from src.mech.fit import Prior, config_prior, contraction, fit_patient, log_bounds
from src.mech.inputs import Record, build_records
from src.mech.model import FIT_PARAMS, bounds_array, initial_state, simulate, theta_from_log
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.provenance import file_hash, write_metrics

PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROC / "mech"

FLAT_THRESHOLD = 1.92  # chi2(1) 95% / 2: profile likelihood CI half-height


def synthetic_record(
    rec: Record,
    logp: np.ndarray,
    mc: dict[str, Any],
    alpha: float,
    noise_sd: float,
    rng: np.random.Generator,
) -> Record:
    """Copy of ``rec`` with y = open-loop simulation + noise where y was observed (SYNTHETIC)."""
    sc = mc["solver"]
    th = theta_from_log(logp, mc["params"]["fixed"], alpha)
    g0 = rec.y[~np.isnan(rec.y)][0] if (~np.isnan(rec.y)).any() else th[0]
    x = simulate(
        th, initial_state(g0, th), rec.carbs, rec.slow, rec.act, sc["dt_min"], sc["n_sub"],
        bounds_array(sc),
    )  # fmt: skip
    y = np.where(np.isnan(rec.y), np.nan, x[:, 2] + rng.normal(0, noise_sd, len(x)))
    return Record(
        rec.patient_id + "_synthetic", rec.period_id + "_synthetic", rec.t, rec.days, y,
        rec.carbs, rec.slow, rec.act, rec.activity_source,
    )  # fmt: skip


def recovery(recs: list[Record], mc: dict[str, Any], alpha: float) -> pd.DataFrame:
    """Draw-simulate-refit loop; one row per synthetic patient x parameter."""
    ic = mc["identifiability"]
    pr = config_prior(mc["params"])
    lo, hi = log_bounds(mc["params"])
    rng = np.random.default_rng(mc["seed"])
    rows = []
    for i in range(ic["n_synthetic"]):
        rec = recs[i % len(recs)]
        true = np.clip(pr.mu + 0.5 * pr.sd * rng.standard_normal(len(pr.mu)), lo + 0.05, hi - 0.05)
        syn = synthetic_record(rec, true, mc, alpha, ic["noise_sd"], rng)
        fr = fit_patient(syn, mc["fit"]["stage1_days"], pr, mc, alpha)
        con = contraction(fr, pr)
        for j, p in enumerate(FIT_PARAMS):
            rows.append(
                {"synthetic_id": i, "param": p, "true_log": true[j], "est_log": fr.logp[j],
                 "contraction": con[j]}
            )  # fmt: skip
    return pd.DataFrame(rows)


def laplace_real(recs: list[Record], mc: dict[str, Any], alpha: float) -> pd.DataFrame:
    """3-day MAP fits of real patients: contraction and posterior correlations."""
    pr = config_prior(mc["params"])
    rows = []
    for rec in recs:
        fr = fit_patient(rec, mc["fit"]["stage1_days"], pr, mc, alpha)
        con = contraction(fr, pr)
        row: dict[str, Any] = {"patient_id": rec.patient_id, "rmse": fr.rmse}
        row |= {f"con_{p}": con[j] for j, p in enumerate(FIT_PARAMS)}
        row |= {f"log_{p}": fr.logp[j] for j, p in enumerate(FIT_PARAMS)}
        if fr.cov is not None:
            sd = np.sqrt(np.diag(fr.cov))
            corr = fr.cov / np.outer(sd, sd)
            for a in range(len(FIT_PARAMS)):
                for b in range(a + 1, len(FIT_PARAMS)):
                    row[f"corr_{FIT_PARAMS[a]}_{FIT_PARAMS[b]}"] = corr[a, b]
        rows.append(row)
    return pd.DataFrame(rows)


def shifted(rec: Record, start_day: float) -> Record:
    """The record from ``start_day`` on, with days re-zeroed (for test-retest fits)."""
    i = int(np.searchsorted(rec.days, start_day, side="left"))
    return Record(
        rec.patient_id, rec.period_id, rec.t[i:], rec.days[i:] - start_day, rec.y[i:],
        rec.carbs[i:], rec.slow[i:], rec.act[i:], rec.activity_source,
    )  # fmt: skip


def retest(recs: list[Record], mc: dict[str, Any], alpha: float) -> pd.DataFrame:
    """Per patient: MAP on days [0, 3) and on days [3, 6) separately (same prior).

    Laplace contraction assumes independent residuals; CGM residuals are autocorrelated, so
    contraction overstates information. Agreement of two disjoint 3-day windows is a direct,
    assumption-light check of practical identifiability (D-034).
    """
    pr = config_prior(mc["params"])
    k = mc["fit"]["stage1_days"]
    rows = []
    for rec in recs:
        if rec.days[-1] < 2 * k:
            continue
        a = fit_patient(rec, k, pr, mc, alpha)
        b = fit_patient(shifted(rec, k), k, pr, mc, alpha)
        rows.append(
            {"patient_id": rec.patient_id}
            | {f"a_{p}": a.logp[j] for j, p in enumerate(FIT_PARAMS)}
            | {f"b_{p}": b.logp[j] for j, p in enumerate(FIT_PARAMS)}
        )
    return pd.DataFrame(rows)


def profile(rec: Record, mc: dict[str, Any], alpha: float) -> pd.DataFrame:
    """Profile delta(-2 log L) per parameter on a log grid around the (flat-prior) optimum."""
    ic = mc["identifiability"]
    pr = config_prior(mc["params"])
    flat = Prior(pr.mu, np.full_like(pr.sd, 1e3))
    lo, hi = log_bounds(mc["params"])
    k = mc["fit"]["stage1_days"]
    best = fit_patient(rec, k, flat, mc, alpha)
    rows = []
    offs = np.linspace(-1, 1, ic["profile_points"]) * ic["profile_halfwidth_log"]
    for j, p in enumerate(FIT_PARAMS):
        others = [q for q in FIT_PARAMS if q != p]
        for v in np.clip(best.logp[j] + offs, lo[j], hi[j]):
            base = best.logp.copy()
            base[j] = v
            fr = fit_patient(rec, k, flat, mc, alpha, free=others, base_logp=base)
            rows.append(
                {"patient_id": rec.patient_id, "param": p, "log_value": v,
                 "delta_m2ll": 2 * (fr.cost - best.cost)}
            )  # fmt: skip
    return pd.DataFrame(rows)


def summarize(
    rec_df: pd.DataFrame,
    lap: pd.DataFrame,
    prof: pd.DataFrame,
    rt: pd.DataFrame,
    mc: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    """Per-parameter recovery / contraction / profile / test-retest + identifiability verdicts.

    "structurally_identifiable" (ideal data): synthetic corr(true, est) >= 0.5 AND median
    real-data Laplace contraction >= contraction_min AND profile non-flat for >= half of the
    profiled patients.
    "practically_identifiable" (real data): additionally, test-retest correlation of the two
    disjoint 3-day windows across patients >= retest_min.
    """
    cmin = mc["identifiability"]["contraction_min"]
    out: dict[str, Any] = {
        "alpha_E_used": alpha,
        "n_synthetic": int(rec_df.synthetic_id.nunique()),
        "n_real": int(len(lap)),
        "n_profile_patients": int(prof.patient_id.nunique()),
        "params": {},
    }
    for p in FIT_PARAMS:
        r = rec_df[rec_df.param == p]
        err = (r.est_log - r.true_log).abs()
        pp = prof[prof.param == p]
        flat = float((pp.groupby("patient_id").delta_m2ll.max() < FLAT_THRESHOLD).mean())
        corr = float(np.corrcoef(r.true_log, r.est_log)[0, 1])
        con_real = float(lap[f"con_{p}"].median())
        rt_corr = float(np.corrcoef(rt[f"a_{p}"], rt[f"b_{p}"])[0, 1])
        rt_diff = float((rt[f"a_{p}"] - rt[f"b_{p}"]).abs().median())
        structural = bool(con_real >= cmin and flat < 0.5 and corr >= 0.5)
        out["params"][p] = {
            "retest_corr_log": rt_corr,
            "retest_abs_log_diff_median": rt_diff,
            "structurally_identifiable": structural,
            "practically_identifiable": bool(
                structural and rt_corr >= mc["identifiability"]["retest_min"]
            ),
            "recovery_abs_log_err_median": float(err.median()),
            "recovery_abs_log_err_p90": float(err.quantile(0.9)),
            "recovery_corr_true_est": corr,
            "contraction_median_synthetic": float(r.contraction.median()),
            "contraction_median_real": con_real,
            "profile_flat_fraction": flat,
        }
    out["n_retest_patients"] = int(len(rt))
    out["posterior_corr_median_real"] = {
        c[5:]: float(lap[c].median()) for c in lap.columns if c.startswith("corr_")
    }
    return out


def run_identify(alpha: float | None = None) -> dict[str, Any]:
    """All identifiability analyses -> data/processed/mech/ident_*.parquet + metrics JSON."""
    mc = load_config("mech")
    alpha = float(np.median(mc["activity"]["alpha_grid"])) if alpha is None else alpha
    grid = load_grid("cgmacros")
    meals = pd.read_parquet(PROC / "meals.parquet")
    meals = meals[meals.dataset == "cgmacros"]
    w = mc["params"]["slow_protein_weight"]
    recs = list(build_records(grid, meals, mc["activity"], w).values())
    OUT.mkdir(parents=True, exist_ok=True)
    rec_df = recovery(recs, mc, alpha)
    lap = laplace_real(recs, mc, alpha)
    n_prof = mc["identifiability"]["profile_patients"]
    prof = pd.concat([profile(r, mc, alpha) for r in recs[:n_prof]], ignore_index=True)
    rec_df.to_parquet(OUT / "ident_recovery_synthetic.parquet", index=False)
    lap.to_parquet(OUT / "ident_laplace_real.parquet", index=False)
    prof.to_parquet(OUT / "ident_profile_real.parquet", index=False)
    rt = retest(recs, mc, alpha)
    rt.to_parquet(OUT / "ident_retest_real.parquet", index=False)
    summary = summarize(rec_df, lap, prof, rt, mc, alpha)
    dh = file_hash([PROC / "cgmacros_ts5.parquet", PROC / "meals.parquet"])
    write_metrics("mech_identifiability", summary, mc, mc["seed"], dh)
    return summary
