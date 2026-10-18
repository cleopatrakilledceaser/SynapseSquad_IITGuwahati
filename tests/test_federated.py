"""Part 8c building blocks (D-063): gradients, FedAvg/FedProx, DP clipping and accounting, layouts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.federated import core as fc

CFG = {"rounds": 25, "local_epochs": 1, "batch": 64, "lr": 0.05}


def toy(n: int = 600, seed: int = 0, shift: float = 0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(shift, 1.0, (n, 4))
    y = np.stack([x[:, 0] * 0.8 - x[:, 1] * 0.3, np.tanh(x[:, 2]) + 0.2 * x[:, 3]], axis=1)
    y = y + rng.normal(0, 0.05, y.shape)
    return x, y, np.ones_like(y)


def mse(w, shapes, data):
    out, _ = fc.forward(fc.unflatten(w, shapes), data[0])
    return float((((out - data[1]) * data[2]) ** 2).sum() / data[2].sum())


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    p = fc.init_params(4, [5, 3], 2, rng)
    x, y, m = toy(40)
    m[::3, 1] = 0.0  # some missing targets
    _, g = fc.loss_grad(p, x, y, m)
    gv, w, sh = fc.flatten(g), fc.flatten(p), fc.shapes_of(p)
    for i in rng.choice(len(w), 12, replace=False):
        e = np.zeros_like(w)
        e[i] = 1e-6
        num = (mse(w + e, sh, (x, y, m)) - mse(w - e, sh, (x, y, m))) / 2e-6
        assert num == pytest.approx(gv[i], rel=1e-4, abs=1e-7)


def test_masked_targets_are_ignored():
    p = fc.init_params(4, [4], 2, np.random.default_rng(0))
    x, y, m = toy(30)
    m[:, 1] = 0.0
    y2 = y.copy()
    y2[:, 1] = 999.0
    assert fc.loss_grad(p, x, y, m)[0] == pytest.approx(fc.loss_grad(p, x, y2, m)[0])


def test_flatten_roundtrip():
    p = fc.init_params(3, [4], 2, np.random.default_rng(0))
    back = fc.unflatten(fc.flatten(p), fc.shapes_of(p))
    assert all(np.array_equal(a, b) for a, b in zip(p, back, strict=True))


def test_fedavg_iid_close_to_central_and_beats_untrained():
    sets = [toy(300, seed=s) for s in range(4)]
    p0 = fc.init_params(4, [8], 2, np.random.default_rng(0))
    sh, w0 = fc.shapes_of(p0), fc.flatten(p0)
    cfg = {**CFG, "rounds": 100}
    wf = fc.federated_train(sets, sh, w0, cfg, np.random.default_rng(1))
    wc = fc.central_train(sets, sh, w0, cfg, np.random.default_rng(2))
    test = toy(400, seed=99)
    e0, ef, ec = (mse(w, sh, test) for w in (w0, wf, wc))
    assert ef < 0.1 * e0 and ec < ef * 1.0 + 1e-9 and ef < 8 * ec + 0.01


def test_fedprox_with_huge_mu_stays_near_global():
    sets = [toy(200, seed=s) for s in range(2)]
    p0 = fc.init_params(4, [4], 2, np.random.default_rng(0))
    sh, w0 = fc.shapes_of(p0), fc.flatten(p0)
    cfg = {**CFG, "rounds": 3}
    near = fc.federated_train(sets, sh, w0, cfg, np.random.default_rng(1), mu=5.0)
    free = fc.federated_train(sets, sh, w0, cfg, np.random.default_rng(1), mu=0.0)
    assert np.linalg.norm(near - w0) < np.linalg.norm(free - w0)


def test_clip_update_bounds_norm_and_keeps_direction():
    d = np.array([3.0, 4.0])
    c = fc.clip_update(d, 1.0)
    assert np.linalg.norm(c) == pytest.approx(1.0) and np.allclose(c / np.linalg.norm(c), d / 5.0)
    assert np.array_equal(fc.clip_update(d, 10.0), d)


def test_clip_update_bounds_non_finite_updates():
    for bad in (np.array([np.inf, 1.0]), np.array([np.nan, 1.0])):
        c = fc.clip_update(bad, 1.0)
        assert np.isfinite(c).all() and np.linalg.norm(c) <= 1.0


def test_dp_with_zero_noise_equals_clipped_mean_and_noise_changes_result():
    sets = [toy(100, seed=s) for s in range(3)]
    p0 = fc.init_params(4, [4], 2, np.random.default_rng(0))
    sh, w0 = fc.shapes_of(p0), fc.flatten(p0)
    cfg = {**CFG, "rounds": 2}
    a = fc.federated_train(sets, sh, w0, cfg, np.random.default_rng(5), dp={"clip": 0.01, "z": 0.0})
    # every round moves at most `clip` in norm
    assert np.linalg.norm(a - w0) <= 2 * 0.01 + 1e-9
    b = fc.federated_train(sets, sh, w0, cfg, np.random.default_rng(5), dp={"clip": 0.01, "z": 5.0})
    assert not np.allclose(a, b)


def test_epsilon_known_value_and_monotone():
    assert fc.epsilon_gaussian(1, 1.0, 1e-5) == pytest.approx(5.30, abs=0.02)
    e = [fc.epsilon_gaussian(40, z, 1e-5) for z in (0.5, 1.0, 2.0, 4.0)]
    assert e == sorted(e, reverse=True)
    assert fc.epsilon_gaussian(80, 1.0, 1e-5) > fc.epsilon_gaussian(40, 1.0, 1e-5)
    assert fc.epsilon_gaussian(10, 0.0, 1e-5) == float("inf")


def pool(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"patient_id": [f"p{i}" for i in range(n)],
                         "cohort": ["a" if i < n // 2 else "b" for i in range(n)],
                         "hba1c": rng.uniform(5, 11, n)})  # fmt: skip


def test_partition_covers_every_patient_once():
    p = pool()
    for scheme, k in (("iid", 5), ("hba1c", 5), ("dataset", 2)):
        c = fc.partition(p, scheme, k, np.random.default_rng(1))
        got = sorted(x for v in c.values() for x in v)
        assert got == sorted(p["patient_id"]) and len(c) == k


def test_hba1c_partition_is_ordered_and_dataset_partition_is_pure():
    p = pool()
    c = fc.partition(p, "hba1c", 4, np.random.default_rng(1))
    h = p.set_index("patient_id")["hba1c"]
    means = [h[v].mean() for v in c.values()]
    assert means == sorted(means)
    d = fc.partition(p, "dataset", 2, np.random.default_rng(1))
    cohort = p.set_index("patient_id")["cohort"]
    assert all((cohort[v] == cohort[v].iloc[0]).all() for v in d.values())


def test_holdout_disjoint_and_nonempty():
    p = pool()
    c = fc.partition(p, "iid", 4, np.random.default_rng(1))
    tr, te = fc.holdout(c, 0.2, np.random.default_rng(2))
    for k in c:
        assert tr[k] and te[k] and not set(tr[k]) & set(te[k])
        assert sorted(tr[k] + te[k]) == sorted(c[k])


def test_federated_stats_equal_pooled_stats_and_ignore_nan():
    rng = np.random.default_rng(0)
    parts = [rng.normal(i, 1 + i, (50, 3)) for i in range(3)]
    parts[1][5, 0] = np.nan
    mean, sd = fc.federated_stats(parts)
    allx = np.concatenate(parts)
    assert np.allclose(mean, np.nanmean(allx, axis=0)) and np.allclose(sd, np.nanstd(allx, axis=0))
    z = fc.standardize(np.array([[np.nan, 0.0, 0.0]]), mean, sd)
    assert z[0, 0] == 0.0
