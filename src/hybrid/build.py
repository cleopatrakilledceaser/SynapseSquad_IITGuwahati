"""Stage S1 (Part 5): causal twin features for every patient, fold and adaptation level k.

Train/test-consistent scheme (D-038), per outer fold (CV repeat 0):
  * Stage-1 fits of TRAINING patients on their first 3 days (config prior) -> population prior
    (identical to Part 4) and the static ridge prior.
  * Evaluated (test) patients at level k: parameters from ONLY their first k days (MAP with the
    population prior; k = 0: prior mean), plus a static-prior variant (personalization method 6).
  * Training patients at level k: parameters from ONLY their own first k days, with a
    leave-one-out population prior (the prior excludes the patient's own stage-1 fit), so no
    training patient's twin features use parameters fitted on its own data after day k. Rows of a
    training patient enter a hybrid fitted at level k only if day >= k (src.hybrid.models).
  * EKF (q_scale, r): Part 4 choice where Part 4 ran that k (0, 1, 3 CGMacros; 3 Shanghai), else
    chosen on training patients' windows with day >= max(3, k) by Gaussian NLL (Part 4 rule).
  The EKF runs over each whole record; features at anchor t use readings/inputs <= t only.
Output: data/processed/hybrid/twinfeat_<cohort>_fold<f>.parquet (+ traj_* at k = traj_k).
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from src.hybrid.twin_features import record_features
from src.mech.fit import config_prior, population_prior
from src.mech.prior import StaticPrior, static_matrix
from src.mech.protocol import Ctx, fit_pids, select_qr
from src.utils.config import PROJECT_ROOT

PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROC / "hybrid"
MECH = PROC / "mech"


def kname(k: float) -> str:
    """Level label used in file / column names (0, 0.5, 1, ...)."""
    return f"{k:g}"


def _part4_choice(cohort: str, fold: int, variant: str) -> dict[str, Any]:
    f = MECH / f"twin_{cohort}_fold{fold}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def patient_rows(
    c: Ctx,
    pid: str,
    logp: np.ndarray,
    alpha: float,
    q: float,
    r: float,
    hc: dict[str, Any],
    keep_traj: bool,
) -> list[pd.DataFrame]:
    """Twin features at every anchor of every record of one patient."""
    out = []
    for rec in c.by_pat[pid]:
        a = c.anchors[c.anchors.period_id == rec.period_id].sort_values("t")
        if a.empty:
            continue
        f = record_features(rec, a["t"].to_numpy(), logp, c.mc, alpha, q, r, c.nsub, hc, keep_traj)
        f.insert(0, "patient_id", pid)
        out.append(f)
    return out


def fold_features(
    c: Ctx, fold: int, train: list[str], test: list[str], hc: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """(features, trajectories, fitted log-params, choices) for one fold and every k."""
    mc = c.mc
    alpha = float(hc["twin"]["alpha_e"])
    var = hc["twin"]["part4_variant"].replace("act", "")
    floor, feats = mc["fit"]["prior_sd_floor"], mc["prior"]["static_features"]
    cp = config_prior(mc["params"])
    st1 = fit_pids(c, train, mc["fit"]["stage1_days"], lambda _p: cp, alpha)
    tr_lp = np.array([st1[p] for p in train])
    pop = population_prior(tr_lp, floor)
    loo = {p: population_prior(np.delete(tr_lp, i, 0), floor) for i, p in enumerate(train)}
    sp = StaticPrior(mc["prior"]["ridge_alphas"], floor).fit(
        static_matrix(c.static, train, feats), tr_lp
    )
    s_test = dict(zip(test, static_matrix(c.static, test, feats), strict=True))
    p4 = _part4_choice(c.cohort, fold, var)
    ch: dict[str, Any] = {"fold": fold, "n_train": len(train), "n_test": len(test)}
    if f"pop_prior_{var}" in p4:
        ch["pop_mu_maxabs_vs_part4"] = float(
            np.max(np.abs(pop.mu - np.array(p4[f"pop_prior_{var}"]["mu"])))
        )
    feats_out, traj_out, fits = [], [], []
    for k in hc["k_grid"]:
        kn = kname(k)
        t0 = time.time()
        tr_k = fit_pids(c, train, k, lambda p: loo[p], alpha)
        if f"qr_{var}_k{kn}" in p4:
            q, r = p4[f"qr_{var}_k{kn}"]["q_scale"], p4[f"qr_{var}_k{kn}"]["r"]
            src = "part4"
        else:
            cs = replace(c, mc={**mc, "test_from_day": max(hc["twin"]["sel_min_day"], k)})
            q, r, _ = select_qr(cs, train, tr_k, alpha)
            src = "selected"
        ch[f"qr_k{kn}"] = {"q_scale": q, "r": r, "source": src}
        te = {
            "pop": fit_pids(c, test, k, lambda _p: pop, alpha),
            "static": fit_pids(c, test, k, lambda p: sp.prior_for(s_test[p]), alpha),
        }
        jobs = [(p, "train", "pop", tr_k[p]) for p in train]
        jobs += [(p, "test", pr, te[pr][p]) for pr in te for p in test]
        keep = c.cohort == "cgmacros" and k == hc["twin"]["traj_k"]
        for p, role, prior, lp in jobs:
            kt = keep and prior == "pop"
            for f in patient_rows(c, p, lp, alpha, q, r, hc["twin"], kt):
                f["k"], f["role"], f["prior"] = k, role, prior
                if kt:
                    tcols = [x for x in f.columns if x.startswith("tr_")]
                    traj_out.append(f[["period_id", "t", "role"] + tcols])
                    f = f.drop(columns=tcols)
                feats_out.append(f)
            fits.append({"patient_id": p, "k": k, "role": role, "prior": prior, "logp": lp})
        ch[f"seconds_k{kn}"] = round(time.time() - t0, 1)
    fdf = pd.concat(feats_out, ignore_index=True)
    tdf = pd.concat(traj_out, ignore_index=True) if traj_out else pd.DataFrame()
    pdf = pd.DataFrame(fits)
    pdf = pd.concat([pdf.drop(columns="logp"), pd.DataFrame(np.vstack(pdf.logp))], axis=1)
    pdf.columns = [str(x) for x in pdf.columns]
    return fdf, tdf, pdf, ch
