"""Population A0 forms for what-ifs (D-050): unconstrained (`unc`) and monotone (`mono`).

Both use the Part 6 CV+ arm protocol (``src.uncertainty.models``): population model on all
fold-training patients (inner split seed 42), inner 3-fold refits (seed 42 + fold) and their
out-of-fold absolute residuals (training patients, first period, day >= band_region_from_day).
``mono`` only adds LightGBM ``monotone_constraints``. No test patient enters any fit.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.hybrid.twin_features import HORIZONS
from src.personalize.data import PopH
from src.uncertainty import conformal as cf
from src.uncertainty import models as um

FORMS = ("unc", "mono")


def constraint_vector(cols: list[str], mcfg: dict[str, Any]) -> list[int]:
    """+1 / -1 / 0 per column (configs/whatif.yaml ``monotone``); unknown names raise."""
    inc, dec = list(mcfg["increasing"]), list(mcfg["decreasing"])
    missing = [c for c in inc + dec if c not in cols]
    if missing:
        raise ValueError(f"monotone columns not in the feature list: {missing}")
    return [1 if c in inc else -1 if c in dec else 0 for c in cols]


def form_bc(bc: dict[str, Any], form: str, cols: list[str], mcfg: dict[str, Any]) -> dict:
    """Baseline config of a form (deep copy; ``mono`` adds monotone constraints)."""
    out = copy.deepcopy(bc)
    if form == "mono":
        out["lightgbm"]["params"]["monotone_constraints"] = constraint_vector(cols, mcfg)
    elif form != "unc":
        raise ValueError(f"unknown form {form}")
    return out


@dataclass
class Form:
    """One fitted form of one fold."""

    name: str
    pop: dict[int, PopH]
    fmap: dict[str, int]
    inner: dict[int, dict[int, PopH]]
    res: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)  # h -> fold -> |res|
    oof: dict[int, np.ndarray] = field(default_factory=dict)  # h -> OOF forecast (trf order)

    def predict(self, df: pd.DataFrame, h: int, pid: str | None = None) -> np.ndarray:
        """Forecast of the full model (pid None) or of the inner model that did not see pid."""
        if pid is None:
            return self.pop[h].predict(df)
        return self.inner[self.fmap[pid]][h].predict(df)

    def split_q(self, h: int, pid: str, a: float) -> float:
        """Split-conformal half-width from OOF residuals of training patients NOT in pid's fold."""
        j = self.fmap[pid]
        r = np.concatenate([v for k, v in self.res[h].items() if k != j])
        return cf.conformal_quantile(r, a)

    def band(
        self, df: pd.DataFrame, h: int, a: float, pid: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Observational band: CV+ (pid None, test patients) or inner split band (pid given)."""
        if pid is not None:
            p, q = self.predict(df, h, pid), self.split_q(h, pid, a)
            return p - q, p + q
        js = sorted(self.inner)
        m = np.column_stack([self.inner[j][h].predict(df) for j in js])
        return cf.cvplus_interval(m, [self.res[h][j] for j in js], a)


def fit_form(
    form: str,
    tab: pd.DataFrame,
    trf: pd.DataFrame,
    train: list[str],
    ctx: dict[str, Any],
) -> Form:
    """Fit one form on the fold-training patients (ctx: cols, strata, bc, wc, inner_folds, seed).

    ``ctx["tcfg"]`` (D-072) selects the tuned population model, as in Part 6.
    """
    wc = ctx["wc"]
    bc = form_bc(ctx["bc"], form, ctx["cols"], wc["monotone"])
    thr = wc["threads"]
    pop, _, _ = um.fit_population(tab, train, ctx["cols"], ctx["strata"], bc, thr, ctx.get("tcfg"))
    fmap, mods = um.inner_models(tab, train, pop, ctx["inner_folds"], ctx["seed"], bc, thr)
    f = Form(form, pop, fmap, mods)
    reg = trf["days_since_start"].to_numpy(float) >= wc["band_region_from_day"]
    fold_of = trf["patient_id"].map(fmap).to_numpy()
    for h in HORIZONS:
        o = um.oof_predict(trf, fmap, mods, h)
        f.oof[h] = o
        r = np.abs(trf[f"y_{h}"].to_numpy(float) - o)
        f.res[h] = {j: r[reg & (fold_of == j) & ~np.isnan(r)] for j in sorted(mods)}
    return f


def oof_guard(f: Form, trf: pd.DataFrame, part6_oof: pd.DataFrame, tol: float) -> float:
    """Max |unc OOF - Part 6 oof_pop| over training rows; raises above tol (D-050 guard)."""
    keys = ["patient_id", "period_id", "t"]
    d = trf[keys].copy()
    for h in HORIZONS:
        d[f"new|{h}"] = f.oof[h]
    m = d.merge(part6_oof, on=keys, how="inner", validate="one_to_one")
    if len(m) != len(d):
        raise RuntimeError(f"Part 6 OOF rows missing: {len(d) - len(m)}")
    diff = max(float(np.nanmax(np.abs(m[f"new|{h}"] - m[f"oof_pop|{h}"]))) for h in HORIZONS)
    if not diff <= tol:
        raise RuntimeError(f"unc refit differs from Part 6 OOF forecasts by {diff:.3g} mg/dL")
    return diff
