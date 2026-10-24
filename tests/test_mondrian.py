"""Group-conditional conformal (D-076): quantiles, fallback, and coverage on toy data."""

import numpy as np
import pandas as pd

from src.uncertainty import mondrian as mo
from src.uncertainty.intervals import coverage_tables


def toy(seed: int = 0):
    """Two groups with different error scales; 4 folds; test and OOF frames."""
    rng = np.random.default_rng(seed)
    rows_te, rows_oof = [], []
    for f in range(4):
        for p in range(10):
            g = "t2d" if p % 3 == 0 else "healthy"
            sd = 30.0 if g == "t2d" else 10.0
            for _ in range(150):
                y = rng.normal(150, 20)
                rows_oof.append(
                    {
                        "fold": f,
                        "patient_id": f"tr{f}_{p}",
                        "group": g,
                        "days_since_start": 8.0,
                        **{f"y|{h}": y for h in mo.HORIZONS},
                        **{f"oof_pop|{h}": y + rng.normal(0, sd) for h in mo.HORIZONS},
                    }
                )
            for k in range(60):
                y = rng.normal(150, 20)
                rows_te.append(
                    {
                        "fold": f,
                        "patient_id": f"te{f}_{p}",
                        "period_id": "x",
                        "t": k,
                        "group": g,
                        **{f"y|{h}": y for h in mo.HORIZONS},
                        **{f"cvp_pop|{h}": y + rng.normal(0, sd) for h in mo.HORIZONS},
                    }
                )
    return pd.DataFrame(rows_te), pd.DataFrame(rows_oof)


def test_group_quantiles_and_fallback():
    res = np.r_[np.full(300, 1.0), np.full(300, 10.0), np.full(10, 50.0)]
    grp = np.r_[["a"] * 300, ["b"] * 300, ["c"] * 10]
    pid = np.r_[[f"a{i % 3}" for i in range(300)], [f"b{i % 3}" for i in range(300)],
                [f"c{i % 3}" for i in range(10)]]  # fmt: skip
    pooled, q = mo.group_quantiles(res, grp, pid, 0.2)
    assert q["a"]["q"] == 1.0 and q["b"]["q"] == 10.0
    assert q["c"]["fallback"] and q["c"]["q"] == pooled


def test_mondrian_restores_group_coverage():
    te, oof = toy()
    uc = {"test_from_day": 7, "alphas": [0.2]}
    te2, _ = mo.add_intervals(te, oof, uc)
    cov = coverage_tables(mo.keep_cols(te2), {"n_boot": 50, "ci": 0.95}, 5)
    g = {(r["method"], r["group"]): r["coverage"] for r in cov["group"] if r["horizon_min"] == 60}
    assert g[("cvc", "t2d")] < 0.72  # pooled band too narrow for the noisy group
    assert abs(g[("cvm", "t2d")] - 0.80) < 0.04 and abs(g[("cvm", "healthy")] - 0.80) < 0.04
