"""Vectorized LightGBM-text runtime (src.appdata.rt.Forest) vs a scalar reference (D-058)."""

from __future__ import annotations

import numpy as np

from src.appdata.rt import Forest, parse_model_text


def random_tree(rng: np.random.Generator, n_feat: int, n_leaves: int) -> dict[str, list[str]]:
    """Random binary tree in LightGBM's text layout (internal nodes >= 0, leaves < 0)."""
    k = n_leaves - 1
    left, right = [0] * k, [0] * k
    free = [(0, "l"), (0, "r")] if k else []
    leaf_ids = list(range(n_leaves))
    for i in range(1, k):
        j = int(rng.integers(len(free)))
        parent, side = free.pop(j)
        (left if side == "l" else right)[parent] = i
        free += [(i, "l"), (i, "r")]
    for parent, side in free:
        (left if side == "l" else right)[parent] = -1 - leaf_ids.pop()
    dec = []
    for _ in range(k):
        mt = int(rng.integers(3))  # 0 none, 1 zero, 2 NaN
        dec.append(mt * 4 + 2 * int(rng.integers(2)))
    return {
        "num_leaves": [str(n_leaves)], "num_cat": ["0"],
        "split_feature": [str(int(rng.integers(n_feat))) for _ in range(k)],
        "threshold": [repr(float(rng.normal())) for _ in range(k)],
        "decision_type": [str(d) for d in dec],
        "left_child": [str(v) for v in left], "right_child": [str(v) for v in right],
        "leaf_value": [repr(float(rng.normal())) for _ in range(n_leaves)],
    }  # fmt: skip


def reference(trees: list[dict[str, list[str]]], X: np.ndarray) -> np.ndarray:
    """Scalar walk with LightGBM's missing-value rules."""
    out = np.zeros(len(X))
    for t in trees:
        leaf = [float(v) for v in t["leaf_value"]]
        if "left_child" not in t or not t["left_child"]:
            out += leaf[0]
            continue
        for i, row in enumerate(X):
            n = 0
            while n >= 0:
                d = int(t["decision_type"][n])
                mt, dleft = (d >> 2) & 3, bool(d & 2)
                v = row[int(t["split_feature"][n])]
                if np.isnan(v) and mt != 2:
                    v = 0.0
                if (np.isnan(v) and mt == 2) or (mt == 1 and abs(v) <= 1e-35):
                    left = dleft
                else:
                    left = v <= float(t["threshold"][n])
                n = int(t["left_child"][n] if left else t["right_child"][n])
            out[i] += leaf[-1 - n]
    return out


def test_forest_matches_scalar_reference() -> None:
    rng = np.random.default_rng(0)
    trees = [random_tree(rng, 6, int(rng.integers(2, 14))) for _ in range(60)]
    trees.append({"num_leaves": ["1"], "num_cat": ["0"], "leaf_value": ["0.25"]})
    X = rng.normal(size=(40, 6))
    X[rng.random(X.shape) < 0.2] = np.nan
    X[rng.random(X.shape) < 0.1] = 0.0
    got = Forest(trees).predict(X)
    assert np.max(np.abs(got - reference(trees, X))) < 1e-12


def test_parse_model_text_reads_trees_and_stops_at_end() -> None:
    text = "\n".join(
        [
            "tree", "version=v3", "Tree=0", "num_leaves=2", "num_cat=0", "split_feature=0",
            "threshold=0.5", "decision_type=2", "left_child=-1", "right_child=-2",
            "leaf_value=1.5 -2.5", "", "Tree=1", "num_leaves=1", "num_cat=0", "leaf_value=0.5",
            "", "end of trees", "feature_importances:",
        ]
    )  # fmt: skip
    f = parse_model_text(text)
    assert len(f) == 2
    assert np.allclose(f.predict(np.array([[0.0], [1.0], [np.nan]])), [2.0, -2.0, 2.0])
