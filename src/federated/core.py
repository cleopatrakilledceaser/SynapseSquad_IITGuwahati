"""Part 8c building blocks (D-063): NumPy MLP, FedAvg / FedProx, client-level DP, accounting.

Everything here is pure NumPy so it is unit-tested without a deep-learning framework. The model is
a small multi-output MLP trained on masked squared error (missing targets are ignored).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

Params = list[np.ndarray]
Client = tuple[np.ndarray, np.ndarray, np.ndarray]  # X, Y, mask


def init_params(d_in: int, hidden: list[int], d_out: int, rng: np.random.Generator) -> Params:
    """He-initialised weights and zero biases: [W1, b1, W2, b2, ...]."""
    sizes = [d_in, *hidden, d_out]
    out: Params = []
    for a, b in zip(sizes[:-1], sizes[1:], strict=True):
        out += [rng.normal(0.0, math.sqrt(2.0 / a), (a, b)), np.zeros(b)]
    return out


def shapes_of(params: Params) -> list[tuple[int, ...]]:
    """Shapes of the parameter arrays (needed to unflatten a vector)."""
    return [p.shape for p in params]


def flatten(params: Params) -> np.ndarray:
    """All parameters as one vector."""
    return np.concatenate([p.ravel() for p in params])


def unflatten(vec: np.ndarray, shapes: list[tuple[int, ...]]) -> Params:
    """Inverse of :func:`flatten`."""
    out, i = [], 0
    for s in shapes:
        n = int(np.prod(s))
        out.append(vec[i : i + n].reshape(s))
        i += n
    return out


def forward(params: Params, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Network output and the list of layer inputs (for backprop)."""
    acts = [x]
    a = x
    n_layers = len(params) // 2
    for i in range(n_layers):
        z = a @ params[2 * i] + params[2 * i + 1]
        a = np.maximum(z, 0.0) if i < n_layers - 1 else z
        acts.append(a)
    return a, acts


def loss_grad(
    params: Params, x: np.ndarray, y: np.ndarray, m: np.ndarray
) -> tuple[float, list[np.ndarray]]:
    """Masked mean squared error and its gradient (same layout as ``params``)."""
    out, acts = forward(params, x)
    den = max(float(m.sum()), 1.0)
    diff = (out - y) * m
    loss = float((diff**2).sum() / den)
    d = 2.0 * diff / den
    grads: list[np.ndarray] = [None] * len(params)  # type: ignore[list-item]
    n_layers = len(params) // 2
    for i in range(n_layers - 1, -1, -1):
        grads[2 * i] = acts[i].T @ d
        grads[2 * i + 1] = d.sum(axis=0)
        if i > 0:
            d = (d @ params[2 * i].T) * (acts[i] > 0)
    return loss, grads


def local_train(
    w0: np.ndarray,
    shapes: list[tuple[int, ...]],
    data: Client,
    epochs: int,
    batch: int,
    lr: float,
    rng: np.random.Generator,
    mu: float = 0.0,
    w_ref: np.ndarray | None = None,
) -> np.ndarray:
    """Minibatch SGD from ``w0``; ``mu`` > 0 adds the FedProx term mu (w - w_ref)."""
    x, y, m = data
    w = w0.copy()
    n = len(x)
    if n == 0:
        return w
    ref = w0 if w_ref is None else w_ref
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s : s + batch]
            _, g = loss_grad(unflatten(w, shapes), x[idx], y[idx], m[idx])
            gv = flatten(g)
            if mu > 0.0:
                gv = gv + mu * (w - ref)
            w = w - lr * gv
    return w


def clip_update(delta: np.ndarray, clip: float) -> np.ndarray:
    """Scale ``delta`` so its L2 norm is at most ``clip``.

    A non-finite update (local training diverged) becomes a zero update, so the norm bound that
    the DP accounting relies on holds for every input.
    """
    if not np.isfinite(delta).all():
        return np.zeros_like(delta)
    norm = float(np.linalg.norm(delta))
    return delta if norm <= clip else delta * (clip / norm)


def federated_train(
    clients: list[Client],
    shapes: list[tuple[int, ...]],
    w0: np.ndarray,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    mu: float = 0.0,
    dp: dict[str, float] | None = None,
) -> np.ndarray:
    """FedAvg (``mu`` = 0) or FedProx for ``rounds``; every client takes part in every round.

    Without ``dp`` the server averages updates weighted by sample count. With ``dp`` (keys
    ``clip``, ``z``) each client's update is clipped to norm ``clip``, the clipped updates are
    averaged with equal weight, and Gaussian noise of standard deviation ``z * clip / n_clients``
    is added to the average (client-level DP, full participation).
    """
    w = w0.copy()
    sizes = np.array([len(c[0]) for c in clients], float)
    active = [i for i, s in enumerate(sizes) if s > 0]
    for _ in range(cfg["rounds"]):
        deltas = []
        for i in active:
            wi = local_train(
                w, shapes, clients[i], cfg["local_epochs"], cfg["batch"], cfg["lr"], rng, mu, w
            )
            deltas.append(wi - w)
        if dp is None:
            wt = sizes[active] / sizes[active].sum()
            w = w + np.tensordot(wt, np.array(deltas), axes=1)
        else:
            clipped = np.array([clip_update(d, dp["clip"]) for d in deltas])
            noise = rng.normal(0.0, dp["z"] * dp["clip"] / len(active), w.shape)
            w = w + clipped.mean(axis=0) + noise
    return w


def central_train(
    clients: list[Client],
    shapes: list[tuple[int, ...]],
    w0: np.ndarray,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Train on the pooled data of all clients for the same number of epochs as FedAvg."""
    x = np.concatenate([c[0] for c in clients])
    y = np.concatenate([c[1] for c in clients])
    m = np.concatenate([c[2] for c in clients])
    epochs = cfg["rounds"] * cfg["local_epochs"]
    return local_train(w0, shapes, (x, y, m), epochs, cfg["batch"], cfg["lr"], rng)


def epsilon_gaussian(rounds: int, z: float, delta: float) -> float:
    """(epsilon, delta)-DP of ``rounds`` full-participation Gaussian releases, noise multiplier z.

    Renyi DP of one release is alpha / (2 z^2); it adds over rounds; conversion
    epsilon = min over alpha > 1 of rounds * alpha / (2 z^2) + log(1/delta) / (alpha - 1).
    No amplification by subsampling is claimed.
    """
    if z <= 0:
        return math.inf
    log_inv = math.log(1.0 / delta)
    a = 1.0 + math.sqrt(2.0 * z * z * log_inv / rounds)  # closed-form minimiser
    return rounds * a / (2.0 * z * z) + log_inv / (a - 1.0)


def partition(
    patients: pd.DataFrame, scheme: str, k: int, rng: np.random.Generator
) -> dict[str, list[str]]:
    """Assign patients to clinics. ``patients`` has patient_id, cohort, hba1c (may be NaN)."""
    ids = patients["patient_id"].to_numpy()
    if scheme == "dataset":
        return {str(c): list(g["patient_id"]) for c, g in patients.groupby("cohort", sort=True)}
    if scheme == "iid":
        order = rng.permutation(len(ids))
        return {f"c{j}": [str(ids[i]) for i in order[j::k]] for j in range(k)}
    if scheme == "hba1c":
        h = patients["hba1c"].astype(float).fillna(patients["hba1c"].median())
        order = np.argsort(h.to_numpy(), kind="stable")
        return {
            f"c{j}": [str(ids[i]) for i in chunk]
            for j, chunk in enumerate(np.array_split(order, k))
        }
    raise ValueError(f"unknown scheme {scheme}")


def holdout(
    clinics: dict[str, list[str]], fraction: float, rng: np.random.Generator
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Per-clinic patient split into (train, test); every clinic keeps at least one of each."""
    tr, te = {}, {}
    for name, pids in clinics.items():
        p = list(rng.permutation(pids))
        n_te = min(max(1, round(fraction * len(p))), max(len(p) - 1, 0))
        te[name], tr[name] = [str(x) for x in p[:n_te]], [str(x) for x in p[n_te:]]
    return tr, te


def federated_stats(parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation per column from per-clinic sums only (NaN ignored)."""
    n = sum(np.sum(~np.isnan(p), axis=0) for p in parts).astype(float)
    s1 = sum(np.nansum(p, axis=0) for p in parts)
    s2 = sum(np.nansum(p * p, axis=0) for p in parts)
    n = np.maximum(n, 1.0)
    mean = s1 / n
    var = np.maximum(s2 / n - mean**2, 0.0)
    sd = np.sqrt(var)
    return mean, np.where(sd > 1e-9, sd, 1.0)


def standardize(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Standardize with the given statistics; missing values become 0 (the mean)."""
    z = (x - mean) / sd
    return np.where(np.isnan(z), 0.0, z)
