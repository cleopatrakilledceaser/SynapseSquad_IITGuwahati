"""(f) Shuffle test: a model trained on labels shuffled in time must drop to chance (synthetic).

The real-data version (CGMacros, hyper180 within 60 min) runs in ``make models`` and is reported
in reports/metrics/baselines_checks.json.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.lgbm import LGBMForecaster
from src.utils.config import load_config


def _toy(n_pat: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        x = rng.normal(140, 35, 400)
        y_next = x + rng.normal(0, 10, 400)
        rows.append(
            pd.DataFrame(
                {
                    "patient_id": f"p{p}",
                    "cgm_now": x,
                    "f2": rng.normal(size=400),
                    "y_60": y_next,
                    "ev_60": (y_next > 180).astype(float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


N_SHUFFLES = 10


def test_shuffled_labels_give_chance_auroc() -> None:
    """Over several shuffles the MEAN test AUROC must be about 0.5 (single shuffles vary: a model
    fitted to noise can land anywhere near 0.5 +/- 0.15 on a small test set)."""
    base = load_config("baselines")["lightgbm"]
    cfg = {
        **base,
        "classifier_events": ["ev"],
        "params": {**base["params"], "n_jobs": 2, "n_estimators": 200},
    }
    tr, va, te = _toy(12, 1), _toy(4, 2), _toy(6, 3)
    m = LGBMForecaster(["cgm_now", "f2"], cfg, seed=42)
    m.fit(tr, val=va, horizons=[60], events=True)
    auc_true = roc_auc_score(te["ev_60"], m.predict_events(te)["ev_60"])
    rng = np.random.default_rng(42)
    aucs = []
    for _ in range(N_SHUFFLES):
        a, b = tr.copy(), va.copy()
        a["ev_60"] = rng.permutation(a["ev_60"].to_numpy())
        b["ev_60"] = rng.permutation(b["ev_60"].to_numpy())
        ms = LGBMForecaster(["cgm_now", "f2"], cfg, seed=42)
        ms.fit(a, val=b, horizons=[60], events=True)
        aucs.append(roc_auc_score(te["ev_60"], ms.predict_events(te)["ev_60"]))
    assert auc_true > 0.85
    assert abs(float(np.mean(aucs)) - 0.5) < 0.05
