"""Pure-python LightGBM text-model runtime for the app's live what-if (Part 9, D-058).

Predicts +30 / +60 / +120 min with the boosters exported by ``src.appdata.export`` and recomputes
the CV+ band from the shipped out-of-fold residuals, with numpy / pandas and the repo's own
conformal and what-if edit code: the app bundle must import and run with no LightGBM and no
training data. The exported text is LightGBM's own ``model_to_string`` of the
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
from typing import Any

import numpy as np
import pandas as pd

from src.personalize.data import MEMBER_SEP
from src.uncertainty.conformal import conformal_quantile, cvplus_interval
from src.uncertainty.events import exceed_prob
from src.whatif import edits as ed

HORIZONS = (30, 60, 120)

# keys of one tree block in LightGBM's text format (the ones we need to walk the tree)
_TREE_KEYS = (
    "num_leaves", "num_cat", "split_feature", "threshold", "decision_type", "left_child",
    "right_child", "leaf_value", "missing_type", "split_gain",  # fmt: skip
)
# decision_type bit 1: "default_left" = a missing value goes to the left child (D-058)
_DEFAULT_LEFT = 2


class Forest:
    """All trees of one booster as flat arrays; every tree and row is walked in one vectorized pass.

    Split rule = LightGBM's numerical split: a value goes left if ``x <= threshold``. ``decision_type``
    bit 1 is "default left"; bits 2-3 are the missing type (0 none, 1 zero, 2 NaN). With missing type
    NaN a NaN follows the default direction; with type zero, NaN becomes 0 and a (near) zero follows
    the default direction; with type none, NaN becomes 0 and is compared as a number. Categorical
    splits are not used by these models and raise.
    """

    def __init__(self, trees: list[dict[str, list[str]]]) -> None:
        """Flatten the parsed trees of one booster."""
        feat, thr, dec, left, right, leaf, roots = [], [], [], [], [], [], []
        n_node = n_leaf = 0
        for t in trees:
            if int(t.get("num_cat", ["0"])[0]) != 0:
                raise ValueError("categorical splits are not supported by the app runtime")
            lv = np.asarray(t["leaf_value"], float)
            lc = np.asarray(t.get("left_child", []), np.int64)
            rc = np.asarray(t.get("right_child", []), np.int64)
            k = len(lc)  # internal nodes (0 for a single-leaf tree)

            # child ids: >= 0 internal node, < 0 leaf (-1 - leaf index); make them global
            def glob(c: np.ndarray, nn: int = n_node, nl: int = n_leaf) -> np.ndarray:
                return np.where(c >= 0, c + nn, -1 - ((-1 - c) + nl))

            feat.append(np.asarray(t.get("split_feature", []), np.int64))
            thr.append(np.asarray(t.get("threshold", []), float))
            dec.append(np.asarray(t.get("decision_type", []), np.int64))
            left.append(glob(lc))
            right.append(glob(rc))
            roots.append(n_node if k > 0 else -1 - n_leaf)
            leaf.append(lv)
            n_node += k
            n_leaf += len(lv)
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)  # noqa: E731
        self.feat, self.thr, self.dec = cat(feat, np.int64), cat(thr, float), cat(dec, np.int64)
        self.left, self.right = cat(left, np.int64), cat(right, np.int64)
        self.leaf, self.roots = cat(leaf, float), np.asarray(roots, np.int64)
        self.default_left = (self.dec & _DEFAULT_LEFT) != 0
        self.missing = (self.dec >> 2) & 3

    def __len__(self) -> int:
        """Number of trees."""
        return len(self.roots)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Sum of the leaf values of every tree for every row of ``X`` (booster output, mg/dL)."""
        X = np.asarray(X, float)
        n, T = len(X), len(self.roots)
        cur = np.repeat(self.roots[:, None], n, axis=1).ravel()
        row = np.tile(np.arange(n), T)
        act = np.flatnonzero(cur >= 0)
        while act.size:
            nd = cur[act]
            v = X[row[act], self.feat[nd]]
            mt = self.missing[nd]
            nan = np.isnan(v)
            v = np.where(nan & (mt != 2), 0.0, v)  # NaN is a number (0) unless the type is NaN
            miss = (nan & (mt == 2)) | ((mt == 1) & (np.abs(v) <= 1e-35))
            go_left = np.where(miss, self.default_left[nd], v <= self.thr[nd])
            cur[act] = np.where(go_left, self.left[nd], self.right[nd])
            act = act[cur[act] >= 0]
        return self.leaf[-1 - cur].reshape(T, n).sum(axis=0)


def parse_model_text(text: str) -> Forest:
    """Every tree of a LightGBM ``model_to_string`` dump, in order.

    A tuned ensemble (D-072) is exported as member dumps joined by ``MEMBER_SEP``; its forest is
    all members' trees with leaf values divided by the member count, so the sum is their mean.
    """
    parts = text.split(MEMBER_SEP)
    if len(parts) == 1:
        return Forest(_parse_trees(text))
    trees = []
    for part in parts:
        for t in _parse_trees(part):
            t["leaf_value"] = [repr(float(v) / len(parts)) for v in t["leaf_value"]]
            trees.append(t)
    return Forest(trees)


def _parse_trees(text: str) -> list[dict[str, list[str]]]:
    """Parsed tree blocks of one LightGBM dump."""
    trees: list[dict[str, list[str]]] = []
    cur: dict[str, list[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Tree="):
            if "leaf_value" in cur:
                trees.append(cur)
            cur = {}
        elif line.startswith("end of trees"):
            break
        elif "=" in line:
            k, _, v = line.partition("=")
            if k in _TREE_KEYS:
                cur[k] = v.split()
    if "leaf_value" in cur:
        trees.append(cur)
    if not trees:
        raise ValueError("no trees parsed from the model string")
    return trees


def read_model(path: Path) -> Forest:
    """Forest of one exported ``.txt.gz`` booster."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return parse_model_text(fh.read())


@dataclass
class RtForm:
    """One form (unc / mono) as shipped: full model + CV+ ensemble + OOF residuals, no LightGBM."""

    name: str
    base: str
    cols: list[str]
    horizons: list[int]
    pop: dict[int, Forest]
    inner: dict[int, dict[int, Forest]] = field(default_factory=dict)
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
        inner: dict[int, dict[int, Forest]] = {}
        res: dict[int, dict[int, np.ndarray]] = {h: {} for h in hs}
        for j in spec["inner"]:
            inner[j] = {h: read_model(models / f"{name}_{h}_in{j}.txt.gz") for h in hs}
            for h in hs:
                res[h][j] = np.asarray(zf[f"{name}_{h}_{j}"], float)
        return cls(name, base, list(meta["cols"]), hs, pop, inner, res)

    def _raw(self, X: np.ndarray, h: int, j: int | None = None) -> np.ndarray:
        forest = self.pop[h] if j is None else self.inner[j][h]
        return forest.predict(X)

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


def load_forms(root: Path) -> dict[str, RtForm]:
    """Both forms (``unc``, ``mono``) from ``app/demo/models``."""
    meta = json.loads((root / "models.json").read_text("utf-8"))
    return {name: RtForm.load(root, name) for name in meta["forms"]}


@dataclass
class Calibration:
    """Platt maps + OOF residuals of the fold's training patients (Part 6 event probabilities)."""

    res: dict[int, np.ndarray]
    platt: dict[str, dict[str, float] | None]
    events: dict[str, tuple[float, bool]]

    def prob(self, event: str, h: int, pred: np.ndarray) -> np.ndarray:
        """Calibrated P(event within h min) from a forecast; NaN if no map exists."""
        mp = self.platt.get(f"{event}|{h}")
        if mp is None:
            return np.full(np.shape(pred), np.nan)
        thr, above = self.events[event]
        s = np.clip(exceed_prob(np.asarray(pred, float), self.res[h], thr, above), 1e-4, 1 - 1e-4)
        z = mp["coef"] * np.log(s / (1 - s)) + mp["intercept"]
        return 1.0 / (1.0 + np.exp(-z))


def load_calibration(root: Path) -> Calibration:
    """Load ``calib.npz`` / ``calib.json`` written by the app-data builder."""
    meta = json.loads((root / "calib.json").read_text("utf-8"))
    r = np.load(root / "calib.npz")
    return Calibration(
        {h: r[f"res_{h}"] for h in HORIZONS},
        meta["platt"],
        {k: (float(v[0]), bool(v[1])) for k, v in meta["events"].items()},
    )


@dataclass
class PatientState:
    """What a live what-if needs for one demo patient (shipped under app/demo/patients)."""

    pid: str
    rows: pd.DataFrame  # stored anchor rows (model inputs, no outcomes)
    grid: pd.DataFrame  # period grid prefix up to the last replay anchor
    meals: pd.DataFrame  # meal log up to the last replay anchor
    ectx: dict[str, Any]
    kcal_per_g: dict[str, float]
    walk: dict[str, Any]
    cols: list[str]


def edited_forecast(
    form: RtForm,
    st: PatientState,
    t: pd.Timestamp,
    edits: list[ed.Edit],
    alpha: float = 0.2,
    clamp: tuple[float, float] = (40.0, 400.0),
) -> dict[str, Any]:
    """Forecast, CV+ band and edited feature row at anchor ``t`` under ``edits`` (clamped)."""
    row = st.rows[st.rows["t"] == t]
    if len(row) != 1:
        raise ValueError("t is not an anchor of this patient")
    m = ed.apply_meals(st.meals, edits, st.kcal_per_g)
    g = ed.apply_walks(st.grid, st.meals, edits, st.walk)
    d = ed.recompute(row, g, m, st.cols, st.ectx)
    out: dict[str, Any] = {"row": d, "mean": [], "lo": [], "hi": [], "raw_mean": []}
    for h in HORIZONS:
        p = float(form.predict(d, h)[0])
        lo, hi = form.band(d, h, alpha)
        out["raw_mean"].append(p)
        out["mean"].append(float(np.clip(p, *clamp)))
        out["lo"].append(float(np.clip(lo[0], *clamp)))
        out["hi"].append(float(np.clip(hi[0], *clamp)))
    return out
