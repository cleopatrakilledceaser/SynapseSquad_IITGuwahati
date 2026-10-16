"""Part 6 report (D-045..D-047).

Modes:
  python -m src.uncertainty.report dev      TRAINING-patient OOF numbers only (development)
  python -m src.uncertainty.report --final  scores the test windows ONCE -> metrics json + figs
"""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from src.uncertainty import alarm_eval as ae
from src.uncertainty.common import COHORTS, HORIZONS, UNC, boot_ratio, load
from src.uncertainty.events import ece, net_benefit, reliability_curve
from src.uncertainty.intervals import coverage_tables, failures
from src.utils.config import load_config
from src.utils.provenance import file_hash, write_metrics


def dev_summary() -> None:
    """Print training-patient OOF coverage of the CV+ residual quantiles (no test rows)."""
    uc = load_config("uncertainty")
    for c in COHORTS:
        o = load("A", c, "oof")
        reg = o["days_since_start"] >= uc["test_from_day"]
        for h in HORIZONS:
            r = (o[f"y|{h}"] - o[f"oof_pop|{h}"]).abs()[reg].dropna()
            print(f"{c} +{h}: OOF |res| q80 {r.quantile(0.8):.1f} q95 {r.quantile(0.95):.1f}")


def event_section(design: str, cohort: str, uc: dict, bc: dict) -> dict[str, Any]:
    """Calibrated event probabilities (CV+ arm, M_pop) on test onset rows, per fold then pooled."""
    out: dict[str, Any] = {}
    te_all, oof_all = load(design, cohort, "test"), load(design, cohort, "oof")
    lim = uc["events"]["limited_evidence"].get(cohort, [])
    for ev in ("hyper180", "hyper140", "hypo70"):
        for h in HORIZONS:
            parts = []
            for f, te in te_all.groupby("fold"):
                oof = oof_all[oof_all.fold == f]
                p = ae.event_probs(oof, te, ev, h, f"oof_pop|{h}", f"cvp_pop|{h}")
                if p is None:
                    continue
                parts.append(
                    pd.DataFrame(
                        {
                            "patient_id": te.loc[p["mask"], "patient_id"].to_numpy(),
                            "y": p["y"],
                            "raw": p["raw"],
                            "platt": p["platt"],
                            "isotonic": p["isotonic"],
                        }
                    )
                )
            if not parts:
                out[f"{ev}_{h}"] = {"status": "too few training events"}
                continue
            d = pd.concat(parts, ignore_index=True)
            res: dict[str, Any] = {
                "n_rows": int(len(d)),
                "n_pos": int(d.y.sum()),
                "n_patients": int(d.patient_id.nunique()),
                "n_patients_with_events": int(d[d.y == 1].patient_id.nunique()),
                "label": "limited evidence" if ev in lim else "",
            }
            for m in ("raw", "platt", "isotonic"):
                res[m] = {
                    "brier": float(np.mean((d[m] - d.y) ** 2)),
                    "ece": ece(d[m].to_numpy(), d.y.to_numpy(), uc["events"]["ece_bins"]),
                    "reliability": reliability_curve(
                        d[m].to_numpy(), d.y.to_numpy(), uc["events"]["ece_bins"]
                    ),
                }
            res["auroc"] = boot_auc(d, "platt", bc)
            out[f"{ev}_{h}"] = res
    return out


def boot_auc(d: pd.DataFrame, col: str, bc: dict) -> dict[str, Any]:
    """AUROC with a patient-bootstrap CI."""
    from sklearn.metrics import roc_auc_score

    if d.y.nunique() < 2:
        return {"value": float("nan"), "ci": [float("nan")] * 2}
    est = float(roc_auc_score(d.y, d[col]))
    rng = np.random.default_rng(42)
    idx = d.groupby("patient_id").indices
    pids = list(idx)
    draws = []
    for _ in range(bc["n_boot"]):
        rows = np.concatenate([idx[pids[i]] for i in rng.integers(0, len(pids), len(pids))])
        yy = d.y.to_numpy()[rows]
        if yy.min() != yy.max():
            draws.append(roc_auc_score(yy, d[col].to_numpy()[rows]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"value": est, "ci": [float(lo), float(hi)]}


def alarm_section(design: str, cohort: str, uc: dict) -> dict[str, Any]:
    """Alarm burden (60-min alerts) of model / naive / trend systems + decision curves."""
    ac = uc["alerts"]
    step = load_config("features")["cohorts"][cohort]["step_min"]
    te_all, oof_all = load(design, cohort, "test"), load(design, cohort, "oof")
    h = ac["horizon_min"]
    lim = uc["events"]["limited_evidence"].get(cohort, [])
    pts = np.arange(*ac["dca_thresholds"])
    out: dict[str, Any] = {}
    for ev in ac["events"]:
        epi = ae.grid_episodes(cohort, ev, ac, step)
        curves = {s: [] for s in ("model", "naive", "trend")}
        chosen = {s: {r: [] for r in ac["fa_per_day"]} for s in curves}
        dca_rows, n_folds_ok = [], 0
        for f, te in te_all.groupby("fold"):
            oof = oof_all[(oof_all.fold == f) & (oof_all.days_since_start >= 1)].reset_index(
                drop=True
            )
            te = te.reset_index(drop=True)
            res = ae.oof_residuals(oof, f"oof_pop|{h}", h)
            oof = oof[~oof[f"oof_pop|{h}"].isna()].reset_index(drop=True)
            mo, mt = ae.at_risk(oof, epi), ae.at_risk(te, epi)
            o, t = oof[mo].reset_index(drop=True), te[mt].reset_index(drop=True)
            so = ae.scores(o, f"oof_pop|{h}", res, ev, ac)
            stt = ae.scores(t, f"cvp_pop|{h}", res, ev, ac)
            yo, yt = ae.alert_label(o, epi, h), ae.alert_label(t, epi, h)
            if yo.sum() < 5:
                continue
            n_folds_ok += 1
            prob = {}
            for s in curves:
                # threshold chosen on TRAINING OOF rows, applied to test (D-046 rule ii)
                grid_o = ae.threshold_grid(so[s], 60)
                cur_o = ae.sweep(o, so[s], epi, ac, grid_o)
                for r in ac["fa_per_day"]:
                    th = ae.theta_at_rate(cur_o, r)
                    chosen[s][r].append(ae.burden(t, stt[s], th, epi, ac))
                curves[s].append((t, stt[s]))
                raw = s != "model"
                pl = ae.Platt(raw=raw).fit(so[s] / (100.0 if raw else 1.0), yo)
                prob[s] = pl.predict(stt[s] / (100.0 if raw else 1.0))
            dca_rows.append(pd.DataFrame({"y": yt, **prob}))
        if not n_folds_ok:
            out[ev] = {"status": "too few training episodes"}
            continue
        r_ev: dict[str, Any] = {"label": "limited evidence" if ev in lim else ""}
        for s in curves:
            # FROC on pooled test rows (curve points; no tuning)
            t_all = pd.concat([c[0] for c in curves[s]], ignore_index=True)
            s_all = np.concatenate([c[1] for c in curves[s]])
            cur = ae.sweep(t_all, s_all, epi, ac, ae.threshold_grid(s_all, 60))
            pooled = {}
            for r in ac["fa_per_day"]:
                tot = merge_burden(chosen[s][r])
                pooled[str(r)] = ae.summarize_burden(tot) | {"leads": tot["total"]["leads"]}
            r_ev[s] = {
                "froc": [{"theta": th} | v for th, v in cur],
                "sens_at_fa_curve": {str(r): ae.sens_at_rate(cur, r) for r in ac["fa_per_day"]},
                "train_chosen": pooled,
            }
        d = pd.concat(dca_rows, ignore_index=True)
        prev = d.y.mean()
        r_ev["dca"] = {
            "pt": pts.tolist(),
            "treat_all": (prev - (1 - prev) * pts / (1 - pts)).tolist(),
            **{s: net_benefit(d[s].to_numpy(), d.y.to_numpy(), pts).tolist() for s in curves},
            "n_rows": int(len(d)),
            "prevalence": float(prev),
        }
        r_ev["n_episodes_test"] = int(
            sum(len(v) for k, v in epi["onsets"].items() if k in set(te_all.period_id))
        )
        out[ev] = r_ev
    return out


def merge_burden(bs: list[dict]) -> dict:
    """Pool burden() outputs of several folds."""
    tot: dict[str, Any] = {}
    for b in bs:
        for k, v in b["total"].items():
            tot[k] = tot.get(k, [] if k == "leads" else 0) + v
    return {"total": tot}


def risk_coverage(err: np.ndarray, signal: np.ndarray, grid: list[float]) -> list[float]:
    """MAE of the rows kept when retaining the fraction c with the LOWEST signal (most sure)."""
    o = np.argsort(signal, kind="stable")
    e = err[o]
    return [float(e[: max(1, int(round(c * len(e))))].mean()) for c in grid]


def selective_section(design: str, cohort: str, uc: dict, bc: dict) -> dict[str, Any]:
    """Risk-coverage curves for abstaining on wide CQR intervals / low data quality."""
    grid = uc["selective"]["coverage_grid"]
    te = load(design, cohort, "test")
    rob = load(design, cohort, "rob") if design == "A" else None
    out: dict[str, Any] = {"coverage_grid": grid}
    for h in HORIZONS:
        d = te.dropna(subset=[f"y|{h}", f"spl_pop|{h}", f"cqr_pop_lo80|{h}"])
        if rob is not None:
            dq = rob[rob.fault == "clean"][["patient_id", "period_id", "t", "dq"]]
            d = d.merge(dq, on=["patient_id", "period_id", "t"], how="left")
        err = (d[f"y|{h}"] - d[f"spl_pop|{h}"]).abs().to_numpy()
        width = (d[f"cqr_pop_hi80|{h}"] - d[f"cqr_pop_lo80|{h}"]).to_numpy()
        sig = {"cqr_width": width, "random": np.random.default_rng(42).uniform(size=len(d))}
        if "dq" in d:
            dqv = d["dq"].fillna(0).to_numpy()
            sig["low_dq"] = -dqv
            sig["width_and_dq"] = (
                pd.Series(width).rank().to_numpy() + pd.Series(-dqv).rank().to_numpy()
            )
        sig["oracle"] = err
        out[str(h)] = {
            s: {
                "mae": risk_coverage(err, v, grid),
                "aurc": float(np.trapezoid(risk_coverage(err, v, grid)[::-1], grid[::-1])),
            }
            for s, v in sig.items()
        }
        out[str(h)]["n_rows"] = int(len(d))
    return out


def robustness_section(cohort: str, uc: dict, bc: dict) -> dict[str, Any]:
    """MAE / coverage / availability per fault x severity (design A) + DQ vs error."""
    from scipy.stats import spearmanr

    rob = load("A", cohort, "rob")
    out: dict[str, Any] = {"cells": [], "dq": {}}
    for (fault, lvl), d in rob.groupby(["fault", "level"]):
        row: dict[str, Any] = {
            "fault": fault,
            "level": float(lvl),
            "availability": float(d.available.mean()),
            "dq_mean": float(d.dq.mean()),
        }
        for h in HORIZONS:
            ok = d[f"y|{h}"].notna() & d[f"pop|{h}"].notna()
            g = d[ok]
            ae_ = (g[f"y|{h}"] - g[f"pop|{h}"]).abs()
            pp = ae_.groupby(g.patient_id)
            s = boot_ratio(pp.sum(), pp.size(), bc["n_boot"], 42, bc["ci"])
            row[f"mae_{h}"], row[f"mae_ci_{h}"] = s["value"], s["ci"]
            row[f"mae_ft_{h}"] = float((g[f"y|{h}"] - g[f"ft|{h}"]).abs().mean())
            for m in ("scp_pop", "cqr_pop", "scp_ft"):
                for lv in ("80", "95"):
                    lo, hi = g[f"{m}_lo{lv}|{h}"], g[f"{m}_hi{lv}|{h}"]
                    cov = ((g[f"y|{h}"] >= lo) & (g[f"y|{h}"] <= hi)).astype(float)
                    cp = cov.groupby(g.patient_id)
                    c = boot_ratio(cp.sum(), cp.size(), bc["n_boot"], 42, bc["ci"])
                    row[f"cov_{m}_{lv}_{h}"], row[f"cov_ci_{m}_{lv}_{h}"] = c["value"], c["ci"]
        out["cells"].append(row)
    bins = uc["dq"]["bins"]
    for h in HORIZONS:
        g = rob[rob[f"y|{h}"].notna() & rob[f"pop|{h}"].notna()]
        err = (g[f"y|{h}"] - g[f"pop|{h}"]).abs()
        rho = spearmanr(g.dq, err).statistic
        rng = np.random.default_rng(42)
        idx = g.groupby("patient_id").indices
        pids = list(idx)
        draws = []
        for _ in range(200):
            rows = np.concatenate([idx[pids[i]] for i in rng.integers(0, len(pids), len(pids))])
            draws.append(spearmanr(g.dq.to_numpy()[rows], err.to_numpy()[rows]).statistic)
        cut = pd.cut(g.dq, bins, right=False)
        by = err.groupby(cut, observed=True).agg(["mean", "size"])
        out["dq"][str(h)] = {
            "spearman": float(rho),
            "spearman_ci_200boot": [
                float(np.percentile(draws, 2.5)),
                float(np.percentile(draws, 97.5)),
            ],
            "mae_by_dq_bin": {
                str(k): {"mae": float(v["mean"]), "n": int(v["size"])} for k, v in by.iterrows()
            },
            "clean_only_spearman": float(
                spearmanr(g[g.fault == "clean"].dq, err[g.fault == "clean"]).statistic
            ),
        }
    return out


def final() -> None:
    """Score the test windows ONCE (designs A and B), write metrics + figures."""
    from src.uncertainty import figures
    from src.uncertainty.patient_fusion import run as fusion_run

    t0 = time.time()
    uc = load_config("uncertainty")
    bsc = uc["bootstrap"]
    m: dict[str, Any] = {}
    for des in uc["designs"]:
        m[des] = {}
        for c in COHORTS:
            te = load(des, c, "test")
            cov = coverage_tables(te, bsc, uc["min_patients_reliable"])
            r: dict[str, Any] = {
                "n_test_patients": int(te.patient_id.nunique()),
                "coverage": cov,
                "coverage_failures_overall": failures(cov["overall"]),
                "coverage_failures_group": failures(cov["group"]),
                "events": event_section(des, c, uc, bsc),
                "selective": selective_section(des, c, uc, bsc),
                "alarms": alarm_section(des, c, uc),
            }
            if des == "A":
                r["robustness"] = robustness_section(c, uc, bsc)
            m[des][c] = r
            print(f"[final] {des} {c} done ({time.time() - t0:.0f}s)", flush=True)
    pf = {c: fusion_run(c, uc["patient_fusion"], bsc) for c in COHORTS}
    dh = file_hash(sorted(UNC.glob("*.parquet")))
    write_metrics("uncertainty", m, uc, uc["seed"], dh)
    write_metrics("patient_fusion", pf, uc["patient_fusion"], uc["seed"], dh)
    figures.all_figures(m, load_config("base")["figures"]["dpi"])
    figures.fusion(pf, load_config("base")["figures"]["dpi"])
    print(f"[final] {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    """Dev (training OOF only) or --final (test, once)."""
    if "--final" in sys.argv:
        final()
    else:
        dev_summary()


if __name__ == "__main__":
    main()
