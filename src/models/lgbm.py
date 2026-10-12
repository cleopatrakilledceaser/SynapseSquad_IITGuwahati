"""LightGBM baselines (Part 2).

Regression on the glucose change y(t+h) - glucose(t), plus event classifiers. Early stopping and
leaf-count selection use the INNER validation patients only (never the outer test fold). LightGBM
handles NaN natively, so no imputation is needed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass
class LGBMForecaster:
    """Per-horizon LightGBM regressors (+ optional event classifiers) on a fixed feature list."""

    features: list[str]
    cfg: dict[str, Any]
    name: str = "lgbm"
    seed: int = 42
    models: dict = field(default_factory=dict)
    classifiers: dict = field(default_factory=dict)
    chosen: dict = field(default_factory=dict)

    def _params(self, leaves: int) -> dict[str, Any]:
        return {**self.cfg["params"], "num_leaves": leaves, "random_state": self.seed}

    def _fit_best(self, est_cls, xt, yt, xv, yv):  # type: ignore[no-untyped-def]
        best, best_score, best_leaves = None, np.inf, None
        grid = (
            self.cfg["classifier_num_leaves_grid"]
            if est_cls is lgb.LGBMClassifier
            else self.cfg["num_leaves_grid"]
        )
        for leaves in grid:
            m = est_cls(**self._params(leaves))
            m.fit(
                xt,
                yt,
                eval_X=(xv,),
                eval_y=(yv,),
                callbacks=[lgb.early_stopping(self.cfg["early_stopping_rounds"], verbose=False)],
            )
            score = m.best_score_["valid_0"]
            score = float(next(iter(score.values())))
            if score < best_score:
                best, best_score, best_leaves = m, score, leaves
        return best, {"num_leaves": best_leaves, "n_iter": int(best.best_iteration_)}

    def fit(self, tab, grid=None, val=None, horizons=None, events=True, **_):  # type: ignore[no-untyped-def]
        xt, xv = tab[self.features], val[self.features]
        for h in horizons:
            ok_t = tab[f"y_{h}"].notna().to_numpy()
            ok_v = val[f"y_{h}"].notna().to_numpy()
            yt = (tab[f"y_{h}"] - tab["cgm_now"]).to_numpy()
            yv = (val[f"y_{h}"] - val["cgm_now"]).to_numpy()
            self.models[h], self.chosen[f"reg_{h}"] = self._fit_best(
                lgb.LGBMRegressor, xt[ok_t], yt[ok_t], xv[ok_v], yv[ok_v]
            )
            if not events:
                continue
            for ev in self.cfg["classifier_events"]:
                col = f"{ev}_{h}"
                lt, lv = tab[col].notna().to_numpy(), val[col].notna().to_numpy()
                ct, cv = tab[col].to_numpy()[lt], val[col].to_numpy()[lv]
                if ct.sum() < 10 or cv.sum() < 1 or ct.sum() == len(ct):
                    continue  # too few events to train a classifier
                self.classifiers[col], self.chosen[f"clf_{col}"] = self._fit_best(
                    lgb.LGBMClassifier, xt[lt], ct, xv[lv], cv
                )
        return self

    def predict(self, tab, grid=None, horizons=None):  # type: ignore[no-untyped-def]
        x = tab[self.features]
        return {h: tab["cgm_now"].to_numpy() + self.models[h].predict(x) for h in horizons}

    def predict_events(self, tab: pd.DataFrame) -> dict[str, np.ndarray]:
        x = tab[self.features]
        return {col: m.predict_proba(x)[:, 1] for col, m in self.classifiers.items()}
