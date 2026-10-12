"""Forecast and event metrics, Clarke error grid, and PATIENT-level bootstrap CIs (Part 2).

Bootstrap: resample patients with replacement and recompute each metric from the pooled rows of
the resampled patients (patients, not rows, are the unit; rule S3). Point metrics use per-patient
sufficient statistics; AUROC/AUPRC use per-patient score histograms (``auc_bins`` bins), so a
1,000-resample bootstrap stays fast (exact up to binning of the scores).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def clarke_zones(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Clarke error grid zone per point (Clarke et al., Diabetes Care 1987), mg/dL."""
    ref = np.asarray(ref, dtype=float)
    pred = np.asarray(pred, dtype=float)
    z = np.full(ref.shape, "B", dtype="<U1")
    a = ((ref <= 70) & (pred <= 70)) | (np.abs(pred - ref) <= 0.2 * ref)
    e = ((ref >= 180) & (pred <= 70)) | ((ref <= 70) & (pred >= 180))
    c = ((ref >= 70) & (ref <= 290) & (pred >= ref + 110)) | (
        (ref >= 130) & (ref <= 180) & (pred <= (7 / 5) * ref - 182)
    )
    d = (
        ((ref >= 240) & (pred >= 70) & (pred <= 180))
        | ((ref <= 175 / 3) & (pred >= 70) & (pred <= 180))
        | ((ref >= 175 / 3) & (ref <= 70) & (pred >= (6 / 5) * ref))
    )
    z[d] = "D"
    z[c] = "C"
    z[e] = "E"
    z[a] = "A"
    return z


def point_stats(df: pd.DataFrame, y: str, p: str) -> pd.DataFrame:
    """Per-patient sufficient statistics for MAE / RMSE / MARD / Clarke A and A+B."""
    d = df[[y, p, "patient_id"]].dropna()
    err = d[p] - d[y]
    zones = clarke_zones(d[y].to_numpy(), d[p].to_numpy())
    st = pd.DataFrame(
        {
            "patient_id": d["patient_id"].to_numpy(),
            "n": 1.0,
            "abs": err.abs().to_numpy(),
            "sq": (err**2).to_numpy(),
            "ard": (err.abs() / d[y]).to_numpy(),
            "zA": (zones == "A").astype(float),
            "zAB": np.isin(zones, ("A", "B")).astype(float),
        }
    )
    return st.groupby("patient_id").sum()


def point_from_stats(s: pd.DataFrame) -> dict[str, float]:
    """Pooled metrics from (possibly resampled) per-patient sums."""
    n = s["n"].sum()
    return {
        "mae": float(s["abs"].sum() / n),
        "rmse": float(np.sqrt(s["sq"].sum() / n)),
        "mard_pct": float(100 * s["ard"].sum() / n),
        "clarke_A_pct": float(100 * s["zA"].sum() / n),
        "clarke_AB_pct": float(100 * s["zAB"].sum() / n),
        "n": float(n),
    }


def per_patient_point(stats: pd.DataFrame) -> pd.DataFrame:
    """Per-patient MAE / RMSE / MARD / Clarke from the sufficient statistics."""
    n = stats["n"]
    return pd.DataFrame(
        {
            "n": n,
            "mae": stats["abs"] / n,
            "rmse": np.sqrt(stats["sq"] / n),
            "mard_pct": 100 * stats["ard"] / n,
            "clarke_A_pct": 100 * stats["zA"] / n,
        }
    )


def _ci(v: list[float], ci: float) -> list[float]:
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return [float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))]


def bootstrap_point(stats: pd.DataFrame, n_boot: int, seed: int, ci: float) -> dict[str, Any]:
    """Point estimates + percentile CIs by resampling PATIENTS with replacement."""
    rng = np.random.default_rng(seed)
    est = point_from_stats(stats)
    arr = stats[["n", "abs", "sq", "ard", "zA", "zAB"]].to_numpy()
    k = len(arr)
    idx = rng.integers(0, k, (n_boot, k))
    sums = arr[idx].sum(axis=1)  # (n_boot, 6)
    n = sums[:, 0]
    draws = {
        "mae": sums[:, 1] / n,
        "rmse": np.sqrt(sums[:, 2] / n),
        "mard_pct": 100 * sums[:, 3] / n,
        "clarke_A_pct": 100 * sums[:, 4] / n,
        "clarke_AB_pct": 100 * sums[:, 5] / n,
    }
    out: dict[str, Any] = {"n_rows": int(est["n"]), "n_patients": int(k)}
    for m, v in draws.items():
        out[m] = est[m]
        out[f"{m}_ci"] = _ci(list(v), ci)
    return out


def event_hist(df: pd.DataFrame, y: str, p: str, bins: int) -> tuple[np.ndarray, ...]:
    """Per-patient histograms of positive / negative scores, plus per-patient Brier sums."""
    d = df[[y, p, "patient_id"]].dropna()
    pids = np.sort(d["patient_id"].unique())
    idx = np.searchsorted(pids, d["patient_id"].to_numpy())
    b = np.clip((d[p].to_numpy() * bins).astype(int), 0, bins - 1)
    pos = d[y].to_numpy() == 1
    hp = np.zeros((len(pids), bins))
    hn = np.zeros((len(pids), bins))
    np.add.at(hp, (idx[pos], b[pos]), 1)
    np.add.at(hn, (idx[~pos], b[~pos]), 1)
    brier = np.zeros(len(pids))
    np.add.at(brier, idx, (d[p].to_numpy() - d[y].to_numpy()) ** 2)
    return pids, hp, hn, brier


def auc_from_hist(hp: np.ndarray, hn: np.ndarray) -> tuple[float, float]:
    """AUROC (ties count 0.5) and AUPRC (step-wise average precision) from score histograms."""
    tp_b, fp_b = hp[::-1], hn[::-1]  # descending score
    n_pos, n_neg = tp_b.sum(), fp_b.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    tp, fp = np.cumsum(tp_b), np.cumsum(fp_b)
    tp0 = np.r_[0, tp[:-1]]
    auroc = float(np.sum(fp_b * (tp0 + 0.5 * tp_b)) / (n_pos * n_neg))
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0)
    ap = float(np.sum((tp_b / n_pos) * prec))
    return auroc, ap


def bootstrap_event(
    df: pd.DataFrame, y: str, p: str, n_boot: int, seed: int, ci: float, bins: int
) -> dict[str, Any]:
    """AUROC / AUPRC / Brier: exact point estimates + patient-bootstrap CIs."""
    d = df[[y, p, "patient_id"]].dropna()
    pids, hp, hn, brier = event_hist(d, y, p, bins)
    out: dict[str, Any] = {
        "n_rows": int(len(d)),
        "n_pos": int(hp.sum()),
        "n_patients": int(len(pids)),
        "n_patients_with_events": int((hp.sum(axis=1) > 0).sum()),
        "prevalence": float(hp.sum() / max(len(d), 1)),
    }
    if hp.sum() == 0 or hn.sum() == 0:
        return out | {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan")}
    out["auroc"] = float(roc_auc_score(d[y], d[p]))
    out["auprc"] = float(average_precision_score(d[y], d[p]))
    out["brier"] = float(brier.sum() / len(d))
    rng = np.random.default_rng(seed)
    n_rows = hp.sum(axis=1) + hn.sum(axis=1)
    draws: dict[str, list[float]] = {"auroc": [], "auprc": [], "brier": []}
    for _ in range(n_boot):
        ix = rng.integers(0, len(pids), len(pids))
        a, ap = auc_from_hist(hp[ix].sum(0), hn[ix].sum(0))
        draws["auroc"].append(a)
        draws["auprc"].append(ap)
        draws["brier"].append(float(brier[ix].sum() / n_rows[ix].sum()))
    for m, v in draws.items():
        out[f"{m}_ci"] = _ci(v, ci)
    return out


def reliability(ev: dict[str, Any], min_pos: int, min_pat: int) -> str:
    """Owner rule: 'reliable' only with >= min_pos positive windows AND >= min_pat patients."""
    ok = ev.get("n_pos", 0) >= min_pos and ev.get("n_patients_with_events", 0) >= min_pat
    return "reliable" if ok else "not reliable"
