"""Test whether post-meal glucose tracks logged carbs or carbs x (amount eaten / 100) (D-016).

Sample: CGMacros meals with logged carbs > 0 and an amount, isolated (no other logged meal within
``isolation_min`` before or after), with real Dexcom readings for the baseline and the window.
Response: iAUC (mg/dL*min, above the pre-meal baseline) and peak rise over 0-120 min.

Comparisons (Spearman, and within-patient Spearman to remove between-person differences):
- A: all eligible meals; B: only meals with amount <= 100 %; C: only meals with amount > 100 %.
In C the two carb definitions differ most, so it decides whether >100 % means extra servings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.provenance import file_hash, write_metrics  # noqa: E402

PROC = ROOT / "data" / "processed"


def responses(meals: pd.DataFrame, ts: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute baseline, iAUC and peak rise per eligible meal using real readings only."""
    mc = cfg["meals"]
    rows = []
    for pid, m in meals.groupby("patient_id"):
        g = ts[ts.patient_id == pid].set_index("t")
        times = m["t"].sort_values().to_numpy()
        for _, r in m.iterrows():
            others = times[times != r.t.to_datetime64()]
            dt = np.abs((others - r.t.to_datetime64()) / np.timedelta64(1, "m"))
            if dt.size and dt.min() < mc["isolation_min"]:
                continue
            base = g.loc[r.t - pd.Timedelta(minutes=mc["baseline_window_min"]) : r.t]
            win = g.loc[r.t : r.t + pd.Timedelta(minutes=mc["response_window_min"])]
            if base.glucose_is_real.sum() < 2 or win.glucose_is_real.mean() < 0.8:
                continue
            b = base.loc[base.glucose_is_real, "glucose"].mean()
            w = win.loc[win.glucose_is_real, "glucose"]
            mins = (w.index - r.t).total_seconds() / 60
            inc = np.clip(w.to_numpy() - b, 0, None)
            rows.append(
                {
                    "patient_id": pid,
                    "carbs": r.carbs_g,
                    "amount": r.amount_pct_raw,
                    "iauc": float(np.trapezoid(inc, mins)),
                    "peak_rise": float(w.max() - b),
                }
            )
    return pd.DataFrame(rows)


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation (NaN-safe)."""
    ok = x.notna() & y.notna()
    return float(x[ok].rank().corr(y[ok].rank())) if ok.sum() > 3 else float("nan")


def within(df: pd.DataFrame, xcol: str, ycol: str) -> float:
    """Spearman after ranking within patient (removes between-person level differences)."""
    rx = df.groupby("patient_id")[xcol].rank(pct=True)
    ry = df.groupby("patient_id")[ycol].rank(pct=True)
    return spearman(rx, ry)


def main() -> None:
    """Run the analysis, print a compact table, write metrics."""
    cfg = load_config("data")
    meals = pd.read_parquet(PROC / "meals.parquet")
    meals = meals[
        (meals.dataset == "cgmacros") & (meals.carbs_g > 0) & meals.amount_pct_raw.notna()
    ]
    ts = pd.read_parquet(PROC / "cgmacros_ts5.parquet")
    r = responses(meals, ts, cfg)
    r["carbs_x_amount"] = r.carbs * r.amount / 100
    res = {}
    for label, sub in {
        "A_all": r,
        "B_amount_le100": r[r.amount <= 100],
        "C_amount_gt100": r[r.amount > 100],
    }.items():
        for y in ("iauc", "peak_rise"):
            res[f"{label}|{y}"] = {
                "n_meals": len(sub),
                "n_patients": int(sub.patient_id.nunique()),
                "rho_logged_carbs": round(spearman(sub.carbs, sub[y]), 3),
                "rho_carbs_x_amount": round(spearman(sub.carbs_x_amount, sub[y]), 3),
                "within_rho_logged": round(within(sub, "carbs", y), 3),
                "within_rho_x_amount": round(within(sub, "carbs_x_amount", y), 3),
            }
    print(pd.DataFrame(res).T.to_string())
    write_metrics(
        "amount_consumed_analysis", res, cfg, cfg["seed"], file_hash(sorted(PROC.glob("*.parquet")))
    )


if __name__ == "__main__":
    main()
