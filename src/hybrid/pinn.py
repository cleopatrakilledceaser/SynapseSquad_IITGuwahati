"""Hybrid C (Part 5, D-041): GRU + FiLM forecaster with an ODE-consistency (physics) loss.

Network: the Part 3 GRU + FiLM (src.models.neural.make_model) over the causal 6-h sequence
(CGM, wearable, meals, time of day; 15-min steps); FiLM input = glycemic static + the twin's
filtered state at t (k = 3 twin features). Output: the glucose trajectory at t + 5, ..., t + 120
min (24 steps, delta to the current CGM value).
Loss = L1 on REAL CGM readings at those steps (scaled per step on training rows)
     + lambda * mean |r_j| / phys_scale over the trajectory, with the trapezoid residual
       r_j = (G_j - G_{j-1}) / dt - (f_{j-1} + f_j) / 2,  f = -S (G - Gb) - X G + Ra,
where S = SG + aE E, X and Ra come from the twin's open-loop rollout from t (no future meals,
src.hybrid.twin_features), so the physics term uses only information available at t.
lambda per fold on inner-validation patients (seed 0, supervised L1 at 30/60/120); then 5 seeds.
lambda = 0 is reported too (same network and dense supervision, no physics).
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.features.fusion_data import load_grid, load_table
from src.features.make_splits import strata_for
from src.hybrid.build import OUT, PROC
from src.hybrid.models import twin_table
from src.models.experiment import inner_split
from src.models.fusion_lgbm import KEYS, folds
from src.models.neural import CHANNELS, Scaler, build_sequences, make_model
from src.utils.config import load_config

STATE_FEATS = ["tw_Q1", "tw_Q2", "tw_U", "tw_X", "tw_E", "tw_Ra", "tw_Gf", "tw_innov_mean_1h"]


def ode_residual(g: Any, g0: Any, x: Any, ra: Any, s: Any, gb: Any, dt: float) -> Any:
    """Trapezoid residual (n, J) of dG/dt = -S (G - Gb) - X G + Ra (torch or numpy arrays).

    g: (n, J) predicted G at steps 1..J; g0: (n,) G at t; x / ra / s: (n, J + 1) at steps 0..J;
    gb: (n,). Returns r_j for j = 1..J (mg/dL/min).
    """
    if isinstance(g, np.ndarray):
        full = np.concatenate([g0[:, None], g], 1)
    else:
        import torch

        full = torch.cat([g0[:, None], g], 1)
    f = -s * (full - gb[:, None]) - x * full + ra
    return (full[:, 1:] - full[:, :-1]) / dt - 0.5 * (f[:, 1:] + f[:, :-1])


def dense_targets(grid: pd.DataFrame, anchors: pd.DataFrame, n_steps: int) -> np.ndarray:
    """(n, n_steps) REAL CGM readings at t + 1..n_steps grid steps (NaN if not real / absent)."""
    out = np.full((len(anchors), n_steps), np.nan, dtype=np.float32)
    for pid, rows in anchors.groupby("period_id").indices.items():
        g = grid[grid.period_id == pid]
        val = np.where(g["glucose_is_real"], g["glucose"], np.nan).astype(np.float32)
        pos = np.searchsorted(g["t"].to_numpy(), anchors["t"].to_numpy()[rows])
        ix = pos[:, None] + np.arange(1, n_steps + 1)[None, :]
        ok = ix < len(g)
        blk = np.full(ix.shape, np.nan, dtype=np.float32)
        blk[ok] = val[ix[ok]]
        out[rows] = blk
    return out


def _prep(
    arrays: dict[str, np.ndarray], tr: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Torch tensors (scalers fitted on training rows only) + per-step target mean / sd."""
    import torch

    seq, st, yd, g0 = arrays["seq"], arrays["static"], arrays["dense"], arrays["g0"]
    scaler = Scaler().fit(seq[tr])
    with np.errstate(all="ignore"):
        s_mu = np.nan_to_num(np.nanmean(st[tr], 0))
        s_sd = np.nan_to_num(np.nanstd(st[tr], 0))
    s_sd = np.where(s_sd < 1e-6, 1.0, s_sd)
    stz = np.concatenate([np.nan_to_num((st - s_mu) / s_sd), np.isnan(st)], 1)
    dlt = yd - g0[:, None]
    y_mu, y_sd = np.nanmean(dlt[tr], 0), np.nanstd(dlt[tr], 0)

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))

    ten = {
        "xs": t(scaler.transform(seq)),
        "ss": t(stz),
        "yz": t((dlt - y_mu) / y_sd),
        "g0": t(np.nan_to_num(g0)),
        "gb": t(np.nan_to_num(arrays["Gb"])),
        "X": t(np.nan_to_num(arrays["X"])),
        "Ra": t(np.nan_to_num(arrays["Ra"])),
        "S": t(np.nan_to_num(arrays["S"])),
        "mu": t(y_mu),
        "sd": t(y_sd),
    }
    return ten, y_mu, y_sd


def _fit(
    arrays: dict[str, np.ndarray],
    tr: np.ndarray,
    va: np.ndarray,
    ev: np.ndarray | None,
    lam: float,
    seed: int,
    pc: dict[str, Any],
    ncfg: dict[str, Any],
    hidden: int,
) -> tuple[np.ndarray | None, float, int, float]:
    """Train one network -> (test predictions 30/60/120, inner-val MAE, best epoch, phys |r|)."""
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(pc["threads"])
    ten, y_mu, y_sd = _prep(arrays, tr)
    xs, ss, yz = ten["xs"], ten["ss"], ten["yz"]
    model = make_model(xs.shape[2], ss.shape[1], hidden, ncfg["grid"]["dropout"][0], yz.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=ncfg["lr"], weight_decay=ncfg["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    dt, hidx, g0 = float(pc["step_min"]), arrays["hidx"], arrays["g0"]
    tr_i, va_i = torch.from_numpy(np.where(tr)[0]), np.where(va)[0]

    def predict(idx: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            out = [
                model(xs[idx[i : i + 4096]], ss[idx[i : i + 4096]]).numpy()
                for i in range(0, len(idx), 4096)
            ]
        return np.concatenate(out) * y_sd + y_mu + g0[idx, None]

    def val_mae() -> float:
        p, y = predict(va_i)[:, hidx], arrays["y"][va_i]
        ok = ~np.isnan(y)
        return float(np.mean(np.abs(p[ok] - y[ok])))

    def phys(b: Any, out: Any) -> Any:
        g = ten["g0"][b, None] + out * ten["sd"] + ten["mu"]
        r = ode_residual(g, ten["g0"][b], ten["X"][b], ten["Ra"][b], ten["S"][b], ten["gb"][b], dt)
        return r.abs().mean()

    best, state, bad, best_ep = np.inf, None, 0, 0
    for epoch in range(ncfg["max_epochs"]):
        model.train()
        perm = tr_i[torch.randperm(len(tr_i), generator=gen)]
        for i in range(0, len(perm), ncfg["batch_size"]):
            b = perm[i : i + ncfg["batch_size"]]
            out = model(xs[b], ss[b])
            y = yz[b]
            ok = ~torch.isnan(y)
            loss = torch.nn.functional.l1_loss(out[ok], y[ok])
            if lam > 0:
                loss = loss + lam * phys(b, out) / pc["phys_scale"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = val_mae()
        if v < best - 1e-4:
            best, bad, best_ep = v, 0, epoch
            state = {k: x.clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= ncfg["patience"]:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        vb = torch.from_numpy(va_i[:4096])
        pr = float(phys(vb, model(xs[vb], ss[vb])))
    pred = predict(np.where(ev)[0])[:, hidx] if ev is not None else None
    return pred, best, best_ep, pr


def static_cols(tab: pd.DataFrame, hc: dict[str, Any]) -> list[str]:
    """Glycemic static columns of the A0 feature set (CGMacros)."""
    fc = load_config("fusion")
    grp = hc["lgbm"]["groups"]["cgmacros"]["static"]
    keep = [f"static_{s}" for g in grp for s in fc["groups"]["cgmacros"][g]["static"]]
    return [c for c in keep if c in tab.columns]


def fold_arrays(
    fold: int, tab: pd.DataFrame, seq_all: np.ndarray, dense: np.ndarray, hc: dict[str, Any]
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Inputs for one fold: training rows (day >= k, own first-k-day twin) + test rows (day >= 3).

    The twin state (FiLM input) and rollout (physics loss) come from parameters fitted on the
    patient's first k days only (src.hybrid.build), for training and test patients alike.
    """
    pc, k = hc["pinn"], hc["pinn"]["k"]
    n_steps = pc["horizon_max_min"] // pc["step_min"]
    tr = pd.read_parquet(OUT / f"traj_cgmacros_fold{fold}.parquet")
    parts = []
    for role, min_day in (("train", k), ("test", hc["test_from_day"])):
        tw = twin_table("cgmacros", fold, k, role)[["period_id", "t"] + STATE_FEATS]
        trj = tr[tr.role == role].drop(columns="role")
        base = tab[["period_id", "t", "days_since_start"]].reset_index()
        d = base.merge(tw, on=["period_id", "t"]).merge(trj, on=["period_id", "t"])
        d = d[d.days_since_start >= min_day]
        d["role"] = role
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    idx = d["index"].to_numpy()  # positions in ``tab`` (tab has a RangeIndex)
    st = tab[static_cols(tab, hc)].to_numpy(float)[idx]
    arrays = {
        "seq": seq_all[idx],
        "static": np.concatenate([st, d[STATE_FEATS].to_numpy(float)], 1),
        "dense": dense[idx],
        "y": tab[[f"y_{h}" for h in (30, 60, 120)]].to_numpy(float)[idx],
        "g0": tab["cgm_now"].to_numpy(float)[idx],
        "X": d[[f"tr_X_{j}" for j in range(n_steps + 1)]].to_numpy(float),
        "Ra": d[[f"tr_Ra_{j}" for j in range(n_steps + 1)]].to_numpy(float),
        "S": d[[f"tr_S_{j}" for j in range(n_steps + 1)]].to_numpy(float),
        "Gb": d["tr_Gb"].to_numpy(float),
        "hidx": np.array([h // pc["step_min"] - 1 for h in (30, 60, 120)]),
    }
    meta = tab.iloc[idx][KEYS + ["days_since_start"]].reset_index(drop=True)
    meta["role"] = d["role"].to_numpy()
    return arrays, meta


def _inputs(hc: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Anchor table, causal 6-h sequences (Part 3 channels) and dense real-CGM targets."""
    fc, pc = load_config("fusion"), hc["pinn"]
    tab = load_table("cgmacros", fc["n_hours"], fc).reset_index(drop=True)
    grid = load_grid("cgmacros")
    meals = pd.read_parquet(PROC / "meals.parquet")
    chans = [ch for grp in pc["channels"] for ch in CHANNELS[grp]]
    seq = build_sequences(grid, meals, tab, fc["n_hours"], 5, 15, chans)
    dense = dense_targets(grid, tab, pc["horizon_max_min"] // pc["step_min"])
    return tab, seq, dense


def stage_pinn() -> None:
    """Hybrid C: lambda per fold on inner-val patients (seed 0), then 5 seeds + lambda = 0."""
    out_p = OUT / "pinn_cgmacros_k3.parquet"
    if out_p.exists():
        return
    hc, bc, sc, fc = (load_config(n) for n in ("hybrid", "baselines", "splits", "fusion"))
    pc, ncfg = hc["pinn"], fc["neural"]
    sel = json.loads((PROC / "fusion" / "neural_cgmacros_selection.json").read_text())
    hidden = int(sel["E_full"]["hidden"])  # Part 3 choice for the CGMacros E config
    strata = strata_for(pd.read_parquet(PROC / "static.parquet"), "cgmacros", sc)
    t0 = time.time()
    tab, seq, dense = _inputs(hc)
    print(f"  inputs {time.time() - t0:.0f}s", flush=True)
    res_all, choices = [], {"hidden": hidden}
    for _, fno, tr_p, _te in folds("cgmacros", [hc["cv_repeat"]]):
        arrays, meta = fold_arrays(fno, tab, seq, dense, hc)
        in_tr, in_va = inner_split(tr_p, strata, bc, 42 + hc["cv_repeat"])
        trm = (meta.role == "train").to_numpy() & meta.patient_id.isin(in_tr).to_numpy()
        vam = (meta.role == "train").to_numpy() & meta.patient_id.isin(in_va).to_numpy()
        evm = (meta.role == "test").to_numpy()
        grid = pc["lambda_grid"]
        sel_r = Parallel(n_jobs=pc["parallel"])(
            delayed(_fit)(arrays, trm, vam, None, lam, 0, pc, ncfg, hidden) for lam in grid
        )
        scores = {str(lam): float(r[1]) for lam, r in zip(grid, sel_r, strict=True)}
        lam = float(min(grid, key=lambda x: scores[str(x)]))
        choices[f"fold{fno}"] = {"lambda_scores": scores, "lambda": lam}
        runs = [(f"C_lam{lam:g}", lam, s) for s in pc["seeds"]]
        if lam != 0.0:
            runs += [("C_lam0", 0.0, s) for s in pc["seeds"]]
        res = Parallel(n_jobs=pc["parallel"])(
            delayed(_fit)(arrays, trm, vam, evm, la, s, pc, ncfg, hidden) for _, la, s in runs
        )
        clip = tuple(load_config("data")["cgm"]["glucose_range_mgdl"]["cgmacros"])
        for (_name, la, s), (pred, _best, ep, pr) in zip(runs, res, strict=True):
            o = meta.loc[evm, KEYS].copy()
            o["fold"], o["seed"], o["lam"], o["best_epoch"], o["phys_resid"] = fno, s, la, ep, pr
            o["variant"] = "C" if la == lam else "C_lam0"
            for j, h in enumerate((30, 60, 120)):
                o[f"pred_{h}"] = np.clip(pred[:, j], *clip)
            res_all.append(o)
        print(f"  fold {fno}: lambda={lam:g} {scores} {time.time() - t0:.0f}s", flush=True)
    choices["seconds"] = round(time.time() - t0, 1)
    (OUT / "pinn_cgmacros_k3.json").write_text(json.dumps(choices, indent=1))
    pd.concat(res_all, ignore_index=True).to_parquet(out_p, index=False)
