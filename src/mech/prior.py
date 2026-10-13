"""Static-data prior for the twin's log parameters (Part 4).

Ridge regression from standardized static features (BMI, HbA1c, fasting glucose, fasting insulin,
age, sex) to each log parameter. Fitted on TRAINING patients of one CV fold only; the ridge alpha
is chosen per parameter by leave-one-out over those training patients (never the evaluated one).
Missing static values: training-patient median. Prior sd = LOO residual sd (floored).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.mech.fit import Prior


def static_matrix(static: pd.DataFrame, pids: list[str], features: list[str]) -> np.ndarray:
    """(n_patients, n_features) static matrix; sex -> sex_male (1 = M)."""
    s = static.set_index("patient_id").loc[pids].copy()
    s["sex_male"] = (s["sex"].astype(str) == "M").astype(float)
    return s[features].to_numpy(dtype=float)


@dataclass
class StaticPrior:
    """Ridge map static features -> log parameters, fitted on training patients only."""

    alphas: list[float]
    sd_floor: float
    med: np.ndarray | None = None
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    models: list[Ridge] = field(default_factory=list)
    chosen_alpha: list[float] = field(default_factory=list)
    resid_sd: np.ndarray | None = None

    def _x(self, s: np.ndarray) -> np.ndarray:
        s = np.where(np.isnan(s), self.med, s)
        return (s - self.mean) / self.scale

    def fit(self, s: np.ndarray, logps: np.ndarray) -> StaticPrior:
        """Fit; alpha per parameter by leave-one-out MSE over the given (training) patients."""
        self.med = np.nanmedian(s, axis=0)
        filled = np.where(np.isnan(s), self.med, s)
        sd = filled.std(0)
        self.mean, self.scale = filled.mean(0), np.where(sd > 0, sd, 1.0)
        x = self._x(s)
        n = len(x)
        sds = []
        for j in range(logps.shape[1]):
            best_mse, best_a = np.inf, self.alphas[0]
            for a in self.alphas:
                loo = np.empty(n)
                for i in range(n):
                    m = np.ones(n, bool)
                    m[i] = False
                    loo[i] = Ridge(alpha=a).fit(x[m], logps[m, j]).predict(x[i : i + 1])[0]
                mse = float(np.mean((loo - logps[:, j]) ** 2))
                if mse < best_mse:
                    best_mse, best_a = mse, a
            self.chosen_alpha.append(float(best_a))
            self.models.append(Ridge(alpha=best_a).fit(x, logps[:, j]))
            sds.append(max(float(np.sqrt(best_mse)), self.sd_floor))
        self.resid_sd = np.array(sds)
        return self

    def prior_for(self, s_row: np.ndarray) -> Prior:
        """Patient-specific prior (mean = ridge prediction, sd = LOO residual sd)."""
        x = self._x(s_row[None, :])
        return Prior(np.array([m.predict(x)[0] for m in self.models]), self.resid_sd.copy())
