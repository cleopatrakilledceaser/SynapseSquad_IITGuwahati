"""Part 6 runner (``make uncertainty``): intervals + robustness per (design, cohort, fold) job.

Design A = CV repeat 0 (5 folds, headline); B = holdout train / cal / test (confirmatory). Per
job, two arms (src.uncertainty.arms, D-045). Outputs (data/processed/uncertainty/):
  unc_<D>_<cohort>_fold<f>_test.parquet  test patients' region rows: point forecasts, intervals
  unc_<D>_<cohort>_fold<f>_oof.parquet   training patients' out-of-fold rows (event calibration,
                                          alert thresholds; never test patients)
  unc_<D>_<cohort>_fold<f>_rob.parquet   design A: faulted test rows (robustness, D-047)
  unc_<D>_<cohort>_fold<f>.json          roles (patient lists), choices, timings
This module never computes a test metric (that is src.uncertainty.report --final).
Usage: python -m src.uncertainty.run [A|B|all] [cgmacros|shanghai_t2dm|all]
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.features.fusion_data import load_table
from src.features.make_splits import strata_for
from src.hybrid.models import base_columns
from src.hybrid.twin_features import HORIZONS
from src.models.fusion_lgbm import KEYS, folds
from src.personalize.data import eligible, first_period
from src.uncertainty import arms
from src.uncertainty import conformal as cf
from src.uncertainty import models as um
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seed import set_global_seed

PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROC / "uncertainty"
SPLITS = PROC / "splits"


def roles(design: str, cohort: str, fold: int, train: list[str], test: list[str]) -> dict:
    """Patient roles of one job (all disjoint; tested).

    proper = fits of the split arm; cal = conformal scores only; cvplus = CV+ arm fits; test.
    """
    h = pd.read_csv(SPLITS / f"{cohort}_holdout.csv")
    cal_all = set(h.loc[h.split == "calibration", "patient_id"])
    if design == "B":
        train = sorted(h.loc[h.split.isin(["train", "calibration"]), "patient_id"])
        test = sorted(h.loc[h.split == "test", "patient_id"])
    cal = sorted(set(train) & cal_all)
    proper = sorted(set(train) - set(cal))
    return {"proper": proper, "cal": cal, "cvplus": sorted(train), "test": sorted(test)}


def check_roles(r: dict) -> None:
    """Raise if calibration / test patients appear in any fitting set."""
    assert not set(r["cal"]) & set(r["proper"]), "calibration patient in split-arm fit"
    assert not set(r["test"]) & set(r["cvplus"]), "test patient in CV+ fit"
    assert not set(r["test"]) & set(r["cal"]), "test patient used for calibration"
    assert set(r["proper"]) | set(r["cal"]) == set(r["cvplus"])


def region(df: pd.DataFrame, uc: dict) -> np.ndarray:
    """Evaluation region mask (first period rows given; day >= test_from_day)."""
    return df["days_since_start"].to_numpy(float) >= uc["test_from_day"]


def lv(a: float) -> str:
    """'80' / '95' label of a miscoverage level."""
    return str(round(100 * (1 - a)))


def cvplus_arm(ctx: dict, out: pd.DataFrame, oof: pd.DataFrame, ch: dict) -> dict:
    """CV+ arm: Part 5 base on all fold-training patients, inner 3-fold OOF residuals."""
    tab, fp, r, uc, pc, bc = (ctx[k] for k in ("tab", "fp", "r", "uc", "pc", "bc"))
    k, thr = uc["k_finetune"], uc["threads"]
    pop, _, _ = um.fit_population(tab, r["cvplus"], ctx["cols"], ctx["strata"], bc, thr)
    fmap, mods = um.inner_models(tab, r["cvplus"], pop, pc["inner_folds"], ctx["seed"], bc, thr)
    trf, te = ctx["trf"], ctx["te"]
    reg_tr = region(trf, uc)
    oof["inner_fold"] = trf["patient_id"].map(fmap).to_numpy()
    keep = {}
    for h in HORIZONS:
        y = trf[f"y_{h}"].to_numpy(float)
        oof[f"oof_pop|{h}"] = um.oof_predict(trf, fmap, mods, h)
        model_of = {p: mods[fmap[p]][h] for p in r["cvplus"]}
        ftp, _ = um.finetune_frame(trf, h, k, model_of, pc, bc)
        sel = um.select_ft(trf, h, ftp, uc["test_from_day"])
        oof[f"oof_ft|{h}"] = ftp[sel]
        ch[f"cvplus_ft_{h}"] = {"lr": sel[0], "trees": sel[1]}
        out[f"cvp_pop|{h}"] = pop[h].predict(te)
        ftt, boost = um.finetune_frame(te, h, k, {p: pop[h] for p in ctx["test"]}, pc, bc, sel)
        out[f"cvp_ft|{h}"] = ftt[sel]
        keep[h] = {"pop": pop[h], "boost": boost, "sel": sel}
        js = sorted(mods)
        m_pop = np.column_stack([mods[j][h].predict(te) for j in js])
        m_ft = np.column_stack(
            [
                um.finetune_frame(te, h, k, {p: mods[j][h] for p in ctx["test"]}, pc, bc, sel)[0][
                    sel
                ]
                for j in js
            ]
        )
        for name, mm, col in (("pop", m_pop, "oof_pop"), ("ft", m_ft, "oof_ft")):
            res_all = np.abs(y - oof[f"{col}|{h}"].to_numpy(float))
            res = [
                res_all[reg_tr & (oof["inner_fold"].to_numpy() == j) & ~np.isnan(res_all)]
                for j in js
            ]
            ch[f"cvplus_nres_{name}_{h}"] = [int(len(x)) for x in res]
            for a in uc["alphas"]:
                lo, hi = cf.cvplus_interval(mm, res, a)
                out[f"cvp_{name}_lo{lv(a)}|{h}"], out[f"cvp_{name}_hi{lv(a)}|{h}"] = lo, hi
    return keep


def split_arm(ctx: dict, out: pd.DataFrame, oof: pd.DataFrame, ch: dict) -> dict:
    """Split arm: fits on PROPER training patients; calibration patients give scores only."""
    tab, r, uc, pc, bc = (ctx[k] for k in ("tab", "r", "uc", "pc", "bc"))
    k, thr, te = uc["k_finetune"], uc["threads"], ctx["te"]
    pop, in_tr, in_va = um.fit_population(tab, r["proper"], ctx["cols"], ctx["strata"], bc, thr)
    rows = tab[tab.patient_id.isin(r["proper"])]
    qs = arms._quantile_levels(uc["alphas"])
    qm = um.fit_quantiles(
        rows[rows.patient_id.isin(in_tr)], rows[rows.patient_id.isin(in_va)], pop, qs, bc, thr
    )
    fmap, mods = um.inner_models(tab, r["proper"], pop, pc["inner_folds"], ctx["seed"], bc, thr)
    trf = ctx["trf"]
    is_p = trf["patient_id"].isin(r["proper"]).to_numpy()
    prf = trf[is_p]
    cal = ctx["fp"][ctx["fp"].patient_id.isin(r["cal"])]
    reg_c = region(cal, uc)
    keep = {}
    for h in HORIZONS:
        # proper patients' OOF forecasts -> fine-tune choice and ACI gamma (never cal / test)
        o_pop = um.oof_predict(prf, fmap, mods, h)
        ftp, _ = um.finetune_frame(prf, h, k, {p: mods[fmap[p]][h] for p in r["proper"]}, pc, bc)
        sel = um.select_ft(prf, h, ftp, uc["test_from_day"])
        ch[f"split_ft_{h}"] = {"lr": sel[0], "trees": sel[1]}
        o = {"pop": o_pop, "ft": ftp[sel]}
        yp = prf[f"y_{h}"].to_numpy(float)
        for name in ("pop", "ft"):
            col = np.full(len(trf), np.nan)
            col[is_p] = o[name]
            oof[f"split_oof_{name}|{h}"] = col
        # calibration and test forecasts from the proper model
        pred, qpred = {}, {}
        for lab, d in (("cal", cal), ("te", te)):
            ids = sorted(d.patient_id.unique())
            ft_d, boost = um.finetune_frame(d, h, k, {p: pop[h] for p in ids}, pc, bc, sel)
            pred[lab] = {"pop": pop[h].predict(d), "ft": ft_d[sel]}
            qpred[lab] = {q: um.predict_quantile(d, qm[(h, q)], pop[h]) for q in qs}
            if lab == "te":
                keep[h] = {"pop": pop[h], "boost": boost, "sel": sel, "qm": qm, "qs": qs}
        yc, yt = cal[f"y_{h}"].to_numpy(float), te[f"y_{h}"].to_numpy(float)
        for name in ("pop", "ft"):
            out[f"spl_{name}|{h}"] = pred["te"][name]
            sc_cal = np.abs(yc - pred["cal"][name])[reg_c]
            sc_cal = np.sort(sc_cal[~np.isnan(sc_cal)])
            ch[f"split_ncal_{h}"] = int(len(sc_cal))
            sc_te = np.abs(yt - pred["te"][name])
            sc_p = np.abs(yp - o[name])
            for a in uc["alphas"]:
                lab_ = lv(a)
                q = cf.conformal_quantile(sc_cal, a)
                keep[h][f"q_{name}_{lab_}"] = q
                out[f"scp_{name}_lo{lab_}|{h}"] = pred["te"][name] - q
                out[f"scp_{name}_hi{lab_}|{h}"] = pred["te"][name] + q
                g, obj = arms.choose_gamma(
                    prf,
                    sc_p,
                    h,
                    a,
                    uc["aci"]["gamma_grid"],
                    uc["aci_start_day"],
                    uc["test_from_day"],
                )
                ch[f"aci_gamma_{name}_{lab_}_{h}"] = {"gamma": g, "objective": obj}
                lo, hi = arms.aci_patients(
                    te, sc_te, pred["te"][name], h, sc_cal, a, g, uc["aci_start_day"]
                )
                out[f"aci_{name}_lo{lab_}|{h}"], out[f"aci_{name}_hi{lab_}|{h}"] = lo, hi
                # CQR (population quantiles; personalized = shifted by the fine-tune correction)
                shift = {lab: pred[lab]["ft"] - pred[lab]["pop"] for lab in ("cal", "te")}
                s = shift if name == "ft" else {"cal": 0.0, "te": 0.0}
                lo_c = qpred["cal"][a / 2] + s["cal"]
                hi_c = qpred["cal"][1 - a / 2] + s["cal"]
                sc = cf.cqr_scores(lo_c, hi_c, yc)[reg_c]
                qc = cf.conformal_quantile(sc[~np.isnan(sc)], a)
                keep[h][f"qcqr_{name}_{lab_}"] = qc
                out[f"cqr_{name}_lo{lab_}|{h}"] = qpred["te"][a / 2] + s["te"] - qc
                out[f"cqr_{name}_hi{lab_}|{h}"] = qpred["te"][1 - a / 2] + s["te"] + qc
    return keep


def robustness(ctx: dict, keep: dict) -> pd.DataFrame:
    """Design A: faulted test rows scored with the split-arm models (M_pop, M_ft, CQR 80/95)."""
    from src.features.fusion_data import load_grid
    from src.robustness.evaluate import faulted_frame

    uc, te = ctx["uc"], ctx["te"]
    rc = uc["robustness"]
    grid = load_grid(ctx["cohort"])
    meals = pd.read_parquet(PROC / "meals.parquet")
    fcfg = load_config("features")
    step = fcfg["cohorts"][ctx["cohort"]]["step_min"]
    rctx = {
        "tau": uc["test_from_day"],
        "step": step,
        "clip": ctx["clip"],
        "n_h": load_config("fusion")["n_hours"],
        "fcfg": fcfg,
        "dqc": uc["dq"],
        "stale_max": rc["stale_max_min"],
    }
    ev = te[region(te, uc)]
    cases = [("clean", 0.0)] + [
        (f, float(lvl)) for f, s in rc["faults"].items() for lvl in s["levels"]
    ]
    parts = []
    for i, (fault, lvl) in enumerate(cases):
        if fault == "wearable_missing" and ctx["cohort"] != "cgmacros":
            continue  # Shanghai has no wearable stream (n/a, D-047)
        if fault == "timestamp_delay" and lvl % step:
            continue  # not representable on this grid (n/a, D-047)
        spec = rc["faults"].get(fault, {})
        f = "noise" if fault == "clean" else fault
        lv_ = 0.0 if fault == "clean" else lvl
        d = faulted_frame(ev, grid, meals, f, lv_, spec, rctx, ctx["seed"] * 1000 + i)
        d = d.merge(ev[KEYS], on=["period_id", "t"], suffixes=("", "_x"))
        o = d[KEYS + ["available", "dq", "dq_complete", "dq_recency", "dq_plaus", "dq_roc"]].copy()
        o["fault"], o["level"] = fault, lvl
        for h in HORIZONS:
            kh = keep[h]
            o[f"y|{h}"] = d[f"y_{h}"].to_numpy(float)
            o[f"pop|{h}"] = kh["pop"].predict(d)
            ftp = np.full(len(d), np.nan)
            for p, rows in d.groupby("patient_id").indices.items():
                ftp[rows] = um.ft_apply(
                    d.iloc[rows],
                    kh["pop"],
                    kh["boost"].get(p, {}),
                    kh["sel"][0],
                    kh["sel"][1],
                    uc["k_finetune"],
                )
            o[f"ft|{h}"] = ftp
            for a in uc["alphas"]:
                lab_ = lv(a)
                for name in ("pop", "ft"):
                    q = kh[f"q_{name}_{lab_}"]
                    o[f"scp_{name}_lo{lab_}|{h}"] = o[f"{name}|{h}"] - q
                    o[f"scp_{name}_hi{lab_}|{h}"] = o[f"{name}|{h}"] + q
                qlo = um.predict_quantile(d, kh["qm"][(h, a / 2)], kh["pop"])
                qhi = um.predict_quantile(d, kh["qm"][(h, 1 - a / 2)], kh["pop"])
                o[f"cqr_pop_lo{lab_}|{h}"] = qlo - kh[f"qcqr_pop_{lab_}"]
                o[f"cqr_pop_hi{lab_}|{h}"] = qhi + kh[f"qcqr_pop_{lab_}"]
        na = ~o["available"].to_numpy()
        for c in [c for c in o.columns if "|" in c and not c.startswith("y|")]:
            o.loc[na, c] = np.nan
        parts.append(o)
    return pd.concat(parts, ignore_index=True)


def run_job(design: str, cohort: str, fold: int, train: list[str], test: list[str]) -> None:
    """One (design, cohort, fold): both arms (+ robustness in design A) -> parquet + json."""
    stem = OUT / f"unc_{design}_{cohort}_fold{fold}"
    if (stem.parent / f"{stem.name}.json").exists():
        return
    t0 = time.time()
    uc, pc, bc = load_config("uncertainty"), load_config("personalize"), load_config("baselines")
    hc, fc, sc = load_config("hybrid"), load_config("fusion"), load_config("splits")
    tab = load_table(cohort, fc["n_hours"], fc)
    fp = first_period(tab)
    elig = set(eligible(fp, uc["test_from_day"], uc["min_test_anchors"]))
    r = roles(design, cohort, fold, train, test)
    check_roles(r)
    r["test_eval"] = sorted(set(r["test"]) & elig)
    ctx = {
        "tab": tab,
        "fp": fp,
        "r": r,
        "uc": uc,
        "pc": pc,
        "bc": bc,
        "cohort": cohort,
        "cols": base_columns(cohort, hc, tab),
        "strata": strata_for(pd.read_parquet(PROC / "static.parquet"), cohort, sc),
        "seed": 42 + fold if design == "A" else 42,
        "test": r["test_eval"],
        "trf": fp[fp.patient_id.isin(r["cvplus"])].reset_index(drop=True),
        "te": fp[fp.patient_id.isin(r["test_eval"])].reset_index(drop=True),
        "clip": tuple(load_config("data")["cgm"]["glucose_range_mgdl"][cohort]),
    }
    te, trf = ctx["te"], ctx["trf"]
    out = te[KEYS + ["days_since_start", "cgm_now"]].copy()
    oof = trf[KEYS + ["days_since_start", "cgm_now"]].copy()
    for h in HORIZONS:
        out[f"y|{h}"] = te[f"y_{h}"].to_numpy(float)
        oof[f"y|{h}"] = trf[f"y_{h}"].to_numpy(float)
    ch: dict[str, Any] = {"design": design, "cohort": cohort, "fold": fold, "roles": r}
    cvplus_arm(ctx, out, oof, ch)
    keep = split_arm(ctx, out, oof, ch)
    for c in [c for c in out.columns if c.startswith(("cvp_pop|", "cvp_ft|", "spl_"))]:
        out[c] = np.clip(out[c].to_numpy(float), *ctx["clip"])  # point forecasts only
    out = out[region(out, uc)].reset_index(drop=True)
    out.to_parquet(f"{stem}_test.parquet", index=False)
    oof.to_parquet(f"{stem}_oof.parquet", index=False)
    if design == "A":
        t1 = time.time()
        rob = robustness(ctx, keep)
        # consistency check: the CLEAN rebuild must reproduce the stored test forecasts
        cl = rob[rob.fault == "clean"].merge(out[KEYS + ["spl_pop|60"]], on=KEYS)
        diff = np.abs(np.clip(cl["pop|60"], *ctx["clip"]) - cl["spl_pop|60"]).max()
        ch["clean_rebuild_max_abs_diff_60"] = float(diff)
        if not diff < 1e-6:
            raise RuntimeError(f"clean rebuild differs from stored forecasts by {diff:.3f} mg/dL")
        rob.to_parquet(f"{stem}_rob.parquet", index=False)
        ch["robustness_seconds"] = round(time.time() - t1, 1)
    ch["seconds"] = round(time.time() - t0, 1)
    (stem.parent / f"{stem.name}.json").write_text(json.dumps(ch, indent=1, default=str))
    print(f"  {stem.name}: {len(out)} test rows, {ch['seconds']:.0f}s", flush=True)


def jobs(designs: list[str], cohorts: list[str], uc: dict) -> list[tuple]:
    """(design, cohort, fold, train, test) for every job."""
    out = []
    for c in cohorts:
        if "A" in designs:
            for _, f, tr, te in folds(c, [uc["cv_repeat"]]):
                out.append(("A", c, f, tr, te))
        if "B" in designs:
            out.append(("B", c, 0, [], []))
    return out


def main() -> None:
    """Run the requested design(s) / cohort(s) in parallel processes (resumable)."""
    d = sys.argv[1] if len(sys.argv) > 1 else "all"
    c = sys.argv[2] if len(sys.argv) > 2 else "all"
    uc = load_config("uncertainty")
    set_global_seed(uc["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    designs = uc["designs"] if d == "all" else [d]
    cohorts = ["cgmacros", "shanghai_t2dm"] if c == "all" else [c]
    js = jobs(designs, cohorts, uc)
    t0 = time.time()
    print(f"[uncertainty] {len(js)} jobs", flush=True)
    Parallel(n_jobs=uc["n_jobs"])(delayed(run_job)(*j) for j in js)
    print(f"[uncertainty] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
