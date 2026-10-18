"""SHAP contributions of the population LightGBM (Part 9, D-057).

The forecast of a ``PopH`` model is ``cgm_now + booster(x)`` (delta target). TreeSHAP of the
booster (LightGBM ``pred_contrib``, restricted to the ``n_iter`` trees the model really uses)
therefore explains the predicted CHANGE in mg/dL: base value + sum of contributions = predicted
delta (tested). ``shap_package_values`` recomputes the same numbers with the ``shap`` package for
the consistency test and for plots. For the tuned ``PopEns`` (D-072) every function averages
its members (exact: SHAP values of a mean are the mean of the SHAP values).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.personalize.data import PopEns


def contributions(model: Any, x: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-feature TreeSHAP of the predicted delta (mg/dL) and the base value per row."""
    if isinstance(model, PopEns):
        parts = [contributions(m, x) for m in model.members]
        phi = sum(p for p, _ in parts) / len(parts)
        return phi, np.mean([b for _, b in parts], axis=0)
    b = model.est.booster_
    raw = b.predict(x[model.cols], num_iteration=model.n_iter, pred_contrib=True)
    phi = pd.DataFrame(raw[:, :-1], columns=model.cols, index=x.index)
    return phi, raw[:, -1]


def used_features(model: Any) -> list[str]:
    """Features with positive split gain in the first ``n_iter`` trees (the model really uses)."""
    if isinstance(model, PopEns):
        used = set().union(*(used_features(m) for m in model.members))
        return [c for c in model.cols if c in used]
    g = model.est.booster_.feature_importance(importance_type="gain", iteration=model.n_iter)
    return [c for c, v in zip(model.cols, g, strict=True) if v > 0]


def shap_package_values(model: Any, x: pd.DataFrame) -> np.ndarray:
    """Same contributions via ``shap.TreeExplainer`` on the first ``n_iter`` trees."""
    import shap

    if isinstance(model, PopEns):
        return np.mean([shap_package_values(m, x) for m in model.members], axis=0)

    ex = shap.TreeExplainer(model.est.booster_)
    return np.asarray(ex.shap_values(x[model.cols], tree_limit=model.n_iter))


def global_importance(phi: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Mean |SHAP| (mg/dL of the predicted delta) over the given rows, largest first."""
    s = phi.abs().mean().sort_values(ascending=False).head(top)
    return s.rename("mean_abs_shap").rename_axis("feature").reset_index()
