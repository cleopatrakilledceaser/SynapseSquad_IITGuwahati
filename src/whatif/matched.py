"""Matched-meal validation (D-053): predicted vs observed carb effects on real meal pairs.

Pairs are built from the meal log + anchor inputs only (cgm_now, clock time, macros). Observed
differences (outcomes y) are attached by ``observed`` for the pairs the caller passes; the
development stage passes training-patient pairs only, the final stage every pair once.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.hybrid.twin_features import HORIZONS
from src.whatif import edits as ed
from src.whatif.engine import FoldCtx, Period, edited_rows, ml_forecast, twin_run
from src.whatif.twin import at_horizons


def clean_meals(meals: pd.DataFrame, tab: pd.DataFrame, mc: dict) -> pd.DataFrame:
    """Candidates: meal type, amount in range, no other meal nearby, anchor <= 15 min after."""
    m = meals[(meals.meal_type == mc["meal_type"]) & meals.macros_available.astype(bool)].copy()
    lo, hi = mc["amount_pct"]
    m = m[~m.amount_missing.astype(bool) & m.amount_pct_raw.between(lo, hi)]
    keep = []
    for i, r in m.iterrows():
        o = meals[(meals.period_id == r.period_id) & (meals.index != i)]
        near = (o.t > r.t - pd.Timedelta(hours=mc["clean_before_h"])) & (
            o.t < r.t + pd.Timedelta(hours=mc["clean_after_h"])
        )
        keep.append(not near.any())
    m = m[keep].reset_index().rename(columns={"index": "meal"}).sort_values("t")
    m["period_id"] = m["period_id"].astype(str)
    a = tab[["period_id", "t", "cgm_now", "days_since_start"]].sort_values("t")
    a = a.assign(period_id=a["period_id"].astype(str))
    m = pd.merge_asof(
        m,
        a.assign(anchor_t=a["t"]),
        on="t",
        by="period_id",
        direction="forward",
        tolerance=pd.Timedelta(minutes=mc["anchor_max_after_min"]),
    )
    m = m[m.anchor_t.notna() & m.cgm_now.notna()].copy()
    m["clock_h"] = m.t.dt.hour + m.t.dt.minute / 60
    return m.reset_index(drop=True)


def _match(a: pd.Series, b: pd.Series, tol: dict, mc: dict, clock: float, cgm: float) -> bool:
    if abs(a.carbs_g - b.carbs_g) < mc["min_carb_diff_g"]:
        return False
    if any(abs(np.nan_to_num(a[k]) - np.nan_to_num(b[k])) > v for k, v in tol.items()):
        return False
    return abs(a.clock_h - b.clock_h) <= clock and abs(a.cgm_now - b.cgm_now) <= cgm


def _order(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    return (a, b) if a.carbs_g < b.carbs_g else (b, a)


def within_pairs(m: pd.DataFrame, mc: dict) -> list[dict]:
    """Primary: same-patient pairs (ordered so B has more carbs)."""
    out = []
    for pid, g in m.groupby("patient_id"):
        rows = [r for _, r in g.iterrows()]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if _match(a, b, mc["macro_tol_g"], mc, mc["clock_tol_h"], mc["cgm_tol"]):
                    lo, hi = _order(a, b)
                    out.append({"design": "within", "cluster": pid, "A": lo, "B": hi})
    return out


def cross_pairs(m: pd.DataFrame, static: pd.DataFrame, mc: dict) -> list[dict]:
    """Secondary: cross-patient pairs; one (closest clock time) pair per patient pair."""
    cc = mc["cross"]
    s = static.set_index("patient_id")
    m = m.assign(status=m.patient_id.map(s.diabetes_status), a1c=m.patient_id.map(s.hba1c_pct))
    best: dict[tuple, tuple] = {}
    rows = [r for _, r in m.iterrows()]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if a.patient_id == b.patient_id or a.status != b.status:
                continue
            if not abs(a.a1c - b.a1c) <= cc["hba1c_tol"]:
                continue
            if not _match(a, b, cc["macro_tol_g"], mc, cc["clock_tol_h"], cc["cgm_tol"]):
                continue
            lo, hi = _order(a, b)
            key, dclock = (lo.patient_id, hi.patient_id), abs(a.clock_h - b.clock_h)
            if key not in best or dclock < best[key][0]:
                best[key] = (dclock, lo, hi)
    return [{"design": "cross", "cluster": k[0], "A": v[1], "B": v[2]} for k, v in best.items()]


def ml_side(fc: FoldCtx, per: Period, side: pd.Series, carbs: tuple[float, float]) -> dict:
    """ML forecasts in one meal's context with carbs set to (A, B): form -> (2, 3)."""
    row = per.rows[per.rows["t"] == side.anchor_t]
    sets = [[ed.set_macros(side.meal, carbs_g=float(c))] for c in carbs]
    d = pd.concat([edited_rows(fc, per, row, s) for s in sets], ignore_index=True)
    return ml_forecast(fc, d, per.patient_id)


def tw_side(fc: FoldCtx, per: Period, side: pd.Series, carbs: tuple[float, float]) -> np.ndarray:
    """Twin (learned alpha) forecasts at +30/+60/+120 in one meal's context: (2, 3)."""
    out = []
    for c in carbs:
        s = [ed.set_macros(side.meal, carbs_g=float(c))]
        hz = at_horizons(twin_run(fc, per, side.anchor_t, s, fc.alpha_on), HORIZONS)
        out.append([hz[h][0] for h in HORIZONS])
    return np.array(out)


def predicted(fcs: dict, pers: dict, pair: dict, fold_of: dict, twin_min_day: float) -> dict:
    """Edit-based predicted differences (+30/+60/+120, peak), mean over the two contexts."""
    a, b = pair["A"], pair["B"]
    carbs = (float(a.carbs_g), float(b.carbs_g))
    sums: dict[str, list] = {}
    twin_ok = min(a.days_since_start, b.days_since_start) >= twin_min_day
    for side in (a, b):
        fc, per = fcs[fold_of[side.patient_id]], pers[side.period_id]
        for n, f in ml_side(fc, per, side, carbs).items():
            sums.setdefault(n, []).append(f)
        if twin_ok and side.patient_id in fc.twins:
            sums.setdefault("twin_learned", []).append(tw_side(fc, per, side, carbs))
    out = {}
    for n, fs in sums.items():
        if len(fs) != 2:
            continue
        diff = np.mean([f[1] - f[0] for f in fs], axis=0)
        peak = float(np.mean([f[1].max() - f[0].max() for f in fs]))
        out[n] = {**{h: float(diff[i]) for i, h in enumerate(HORIZONS)}, "peak": peak}
    return out


def observed(tab: pd.DataFrame, pair: dict) -> dict[Any, float]:
    """(y_B(+h) - cgm_B) - (y_A(+h) - cgm_A); peak over the three horizons. NaN if missing."""
    v = {}
    for s in ("A", "B"):
        p = pair[s]
        r = tab[(tab.period_id == p.period_id) & (tab.t == p.anchor_t)].iloc[0]
        v[s] = {h: float(r[f"y_{h}"]) - float(r["cgm_now"]) for h in HORIZONS}
    out: dict[Any, float] = {h: v["B"][h] - v["A"][h] for h in HORIZONS}
    out["peak"] = max(v["B"].values()) - max(v["A"].values())
    return out


def pair_record(pair: dict, pred: dict, obs: dict) -> list[dict]:
    """Long records (design, cluster, model, target, predicted, observed)."""
    base = {
        "design": pair["design"],
        "cluster": pair["cluster"],
        "pid_A": pair["A"].patient_id,
        "pid_B": pair["B"].patient_id,
        "carbs_A": float(pair["A"].carbs_g),
        "carbs_B": float(pair["B"].carbs_g),
    }
    out = []
    for n, p in pred.items():
        for k, v in p.items():
            out.append(base | {"model": n, "target": str(k), "predicted": v, "observed": obs[k]})
    return out


def metrics(d: pd.DataFrame, n_boot: int, seed: int, level: float) -> dict[str, Any]:
    """Pearson, Spearman, sign agreement, OLS slope / intercept, means; cluster bootstrap CIs."""
    d = d.dropna(subset=["predicted", "observed"])

    def stats(x: pd.DataFrame) -> np.ndarray:
        p, o = x.predicted.to_numpy(float), x.observed.to_numpy(float)
        if len(p) < 3 or np.std(p) == 0 or np.std(o) == 0:
            return np.full(7, np.nan)
        slope, icpt = np.polyfit(p, o, 1)
        return np.array([
            np.corrcoef(p, o)[0, 1], pd.Series(p).corr(pd.Series(o), method="spearman"),
            np.mean(np.sign(p) == np.sign(o)), slope, icpt, p.mean(), o.mean(),
        ])  # fmt: skip

    names = ["pearson", "spearman", "sign_agreement", "slope", "intercept", "mean_pred", "mean_obs"]
    est = stats(d)
    cl = d.cluster.unique()
    rng = np.random.default_rng(seed)
    groups = {c: d[d.cluster == c] for c in cl}
    draws = np.array([
        stats(pd.concat([groups[c] for c in rng.choice(cl, len(cl))])) for _ in range(n_boot)
    ])  # fmt: skip
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out: dict[str, Any] = {"n_pairs": int(len(d)), "n_clusters": int(len(cl))}
    for i, k in enumerate(names):
        ok = draws[:, i][~np.isnan(draws[:, i])]
        ci = (
            [float(np.percentile(ok, lo)), float(np.percentile(ok, hi))]
            if len(ok)
            else [np.nan] * 2
        )
        out[k] = {"value": float(est[i]), "ci": ci}
    return out
