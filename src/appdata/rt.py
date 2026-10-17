"""Pure-python LightGBM text-model runtime for the app's live what-if (Part 9, D-058).

Predicts +30 / +60 / +120 min with the boosters exported by ``src.appdata.export`` and recomputes
the split-conformal band from the shipped out-of-fold residuals, with NOTHING but numpy / pandas
(and numpy-only conformal code): the app bundle must import and run with no LightGBM, no
scikit-learn and no training data. The exported text is LightGBM's own ``model_to_string`` of the
``n_iter`` trees the model really uses, so the numbers equal the in-memory model bit for bit
(checked by the builder's ``rtform_max_abs_diff`` guard).

Forecast = base column (``cgm_now``) + sum of the leaf values on the row's path (delta target,
exactly as ``PopH.predict``).
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.uncertainty.conformal import cvplus_interval, conformal_quantile

# keys of one tree block in LightGBM's text format (the ones we need to walk the tree)
_TREE_KEYS = (
    "num_leaves", "num_cat", "split_feature", "threshold", "decision_type", "left_child",
    "right_child", "leaf_value", "missing_type", "split_gain",  # fmt: skip
)
# decision_type bit 1: "default_left" = a missing value goes to the left child (D-058)
_DEFAULT_LEFT = 2


class Tree:
    """One parsed LightGBM regression tree (num_cat = 0: no categorical splits in these models)."""

    __slots__ = ("feat", "thr", "dec", "left", "right", "leaf")

    def __init__(self, t: dict[str, list[str]]) -> None:
        if int(t.get("num_cat", ["0"])[0]) != 0:
            raise ValueError("categorical splits are not supported by the app runtime")
        self.feat = np.asarray(t["split_feature"], np.int32)
        self.thr = np.asarray(t["threshold"], float)
        self.dec = np.asarray(t["decision_type"], np.int32)
        self.left = np.asarray(t["left_child"], np.int32)
        self.right = np.asarray(t["right_child"], np.int32)
        self.leaf = np.asarray(t["leaf_value"], float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Booster output (delta mg/dL) of every row of ``X`` (columns = model columns)."""
        out = np.empty(len(X))
        for i, row in enumerate(np.asarray(X, float)):
            node = 0
            while self.left[node] >= 0:
                v = row[self.feat[node]]
                go_left = bool(self.dec[node] & _DEFAULT_LEFT) if np.isnan(v) else bool(v <= self.thr[node])
                node = int(self.left[node] if go_left else self.right[node])
            out[i] = self.leaf[-node - 1]
        return out


def parse_model_text(text: str) -> list[Tree]:
    """Every tree of a LightGBM ``model_to_string`` dump, in order."""
    trees: list[Tree] = []
    cur: dict[str, list[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Tree="):
            if "leaf_value" in cur:
                trees.append(Tree(cur))
            cur = {}
        elif line.startswith("end of trees"):
            break
        elif "=" in line:
            k, _, v = line.partition("=")
            if k in _TREE_KEYS:
                cur[k] = v.split()
    if "leaf_value" in cur:
        trees.append(Tree(cur))
    if not trees:
        raise ValueError("no trees parsed from the model string")
    return trees


def read_model(path: Path) -> list[Tree]:
    """Trees of one exported ``.txt.gz`` booster."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return parse_model_text(fh.read())


@dataclass
class RtForm:
    """One form (unc / mono) as shipped: full model + CV+ ensemble + OOF residuals, no LightGBM."""

    name: str
    base: str
    cols: list[str]
    horizons: list[int]
    pop: dict[int, list[Tree]]
    inner: dict[int, dict[int, list[Tree]]] = field(default_factory=dict)
    res: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def load(cls, models: Path, name: str) -> RtForm:
        """Read ``{name}_{h}_pop.txt.gz`` (+ inner models and residuals) from a models folder."""
        meta = json.loads((models / "models.json").read_text("utf-8"))
        spec = meta["forms"][name]
        hs = [int(h) for h in meta["horizons"]]
        base = spec["base"][str(hs[0])]
        pop = {h: read_model(models / f"{name}_{h}_pop.txt.gz") for h in hs}
        zf = np.load(models / "residuals.npz")
        inner: dict[int, dict[int, list[Tree]]] = {}
        res: dict[int, dict[int, np.ndarray]] = {h: {} for h in hs}
        for j in spec["inner"]:
            inner[j] = {h: read_model(models / f"{name}_{h}_in{j}.txt.gz") for h in hs}
            for h in hs:
                res[h][j] = np.asarray(zf[f"{name}_{h}_{j}"], float)
        return cls(name, base, list(meta["cols"]), hs, pop, inner, res)

    def _raw(self, X: np.ndarray, h: int, j: int | None = None) -> np.ndarray:
        trees = self.pop[h] if j is None else self.inner[j][h]
        out = np.zeros(len(X))
        for tr in trees:
            out += tr.predict(X)
        return out

    def predict(self, df: pd.DataFrame, h: int) -> np.ndarray:
        """Absolute glucose forecast (mg/dL) = base column + booster output."""
        X = df[self.cols].to_numpy(float)
        return df[self.base].to_numpy(float) + self._raw(X, h)

    def split_q(self, h: int, a: float) -> float:
        """Split-conformal half-width from OOF residuals of every training patient."""
        r = np.concatenate([v for v in self.res[h].values()])
        return conformal_quantile(r, a)

    def band(self, df: pd.DataFrame, h: int, a: float) -> tuple[np.ndarray, np.ndarray]:
        """CV+ 1 - a band from the inner models and their OOF residuals (Part 6 rule)."""
        X = df[self.cols].to_numpy(float)
        base = df[self.base].to_numpy(float)
        js = sorted(self.inner)
        m = np.column_stack([base + self._raw(X, h, j) for j in js])
        return cvplus_interval(m, [self.res[h][j] for j in js], a)