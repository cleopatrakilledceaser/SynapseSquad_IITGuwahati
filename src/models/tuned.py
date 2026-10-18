"""Part 10 (D-065): tuned LightGBM forecaster.

Random search over the loss (squared, absolute, Huber) and the main hyperparameters. Candidates
are scored by MAE on the INNER validation patients, the best ``top_k`` are averaged. The candidate
list is a pure function of the config, so every fold and horizon sees the same candidates and the
status-quo setting is always candidate 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.models.lgbm import LGBMForecaster

OBJECTIVES: dict[str, dict[str, Any]] = {
    "l2": {"objective": "regression"},
    "l1": {"objective": "l1"},
    "huber10": {"objective": "huber", "alpha": 10.0},
    "huber25": {"objective": "huber", "alpha": 25.0},
}


def trial_params(tcfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Candidate list: the status-quo trial first, then seeded random draws from the space."""
    rng = np.random.default_rng(tcfg["seed"])
    trials = [dict(tcfg["status_quo"])]
    for _ in range(int(tcfg["n_trials"]) - 1):
        trials.append({k: v[int(rng.integers(len(v)))] for k, v in tcfg["space"].items()})
    return trials


def lgb_params(
    trial: dict[str, Any], tcfg: dict[str, Any], seed: int, n_jobs: int
) -> dict[str, Any]:
    """LightGBM keyword arguments for one trial. The early-stopping metric is always MAE."""
    rest = {k: v for k, v in trial.items() if k != "objective"}
    return {
        **tcfg["fixed"], **rest, **OBJECTIVES[trial["objective"]],
        "metric": "l1", "random_state": seed, "n_jobs": n_jobs,
    }  # fmt: skip


@dataclass
class TunedLGBM(LGBMForecaster):
    """Per-horizon average of the ``top_k`` best LightGBM candidates (by inner-validation MAE)."""

    tcfg: dict[str, Any] = field(default_factory=dict)
    n_jobs: int = 2
    members: dict = field(default_factory=dict)

    def fit(self, tab, grid=None, val=None, horizons=None, events=False, **_):  # type: ignore[no-untyped-def]
        """Fit every candidate per horizon, keep the ``top_k`` by inner-validation MAE."""
        import lightgbm as lgb

        xt, xv = tab[self.features], val[self.features]
        trials = trial_params(self.tcfg)
        for h in horizons:
            yt, yv = self.encode(tab, h), self.encode(val, h)
            ok_t, ok_v = ~np.isnan(yt), ~np.isnan(yv)
            fitted = []
            for i, t in enumerate(trials):
                if t["objective"] in self.tcfg.get("skip_objectives", ()):
                    continue  # D-072: infeasible with monotone constraints (LightGBM)
                m = lgb.LGBMRegressor(**lgb_params(t, self.tcfg, self.seed, self.n_jobs))
                m.fit(
                    xt[ok_t], yt[ok_t],
                    eval_X=(xv[ok_v],), eval_y=(yv[ok_v],),
                    callbacks=[
                        lgb.early_stopping(self.tcfg["early_stopping_rounds"], verbose=False)
                    ],
                )  # fmt: skip
                fitted.append((float(m.best_score_["valid_0"]["l1"]), i, m))
            fitted.sort(key=lambda r: (r[0], r[1]))
            top = fitted[: int(self.tcfg["top_k"])]
            self.members[h] = [m for _, _, m in top]
            ens = np.mean([m.predict(xv[ok_v]) for m in self.members[h]], axis=0)
            self.chosen[f"reg_{h}"] = {
                "status_quo_inner_mae": next(s for s, i, _ in fitted if i == 0),
                "ensemble_inner_mae": float(np.mean(np.abs(ens - yv[ok_v]))),
                "selected": [
                    {"trial": i, "inner_mae": s, "n_iter": int(m.best_iteration_), **trials[i]}
                    for s, i, m in top
                ],
            }
        return self

    def predict(self, tab, grid=None, horizons=None):  # type: ignore[no-untyped-def]
        """Mean of the kept members per horizon, decoded to absolute glucose."""
        x = tab[self.features]
        return {
            h: self.decode(tab, np.mean([m.predict(x) for m in self.members[h]], axis=0))
            for h in horizons
        }
