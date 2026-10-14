"""Neural fusion forecaster (Part 3): GRU over the dynamic window + FiLM static conditioning.

Input per anchor t: the last N hours of grid rows ENDING AT t (causal), subsampled to 15 min, with
channels [glucose (real only), real-mask, HR, activity, meal macros logged in the bin, tod
sin/cos] and a static vector. Missing values -> training-fold mean (scaler fitted on training
patients only) plus explicit mask channels (glucose mask; static NaN mask).

FiLM (Perez et al., AAAI 2018): gamma, beta = MLP(static); h = (1 + gamma) * GRU_state + beta,
then a multi-horizon head (30/60/120 min) predicts the scaled delta (or absolute) target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MEAL_COLS = {
    "carbs": "carbs_g",
    "protein": "protein_g",
    "fat": "fat_g",
    "fiber": "fiber_g",
    "kcal": "kcal",
}
CHANNELS = {
    "cgm": ["glu", "glu_mask"],
    "wearable": ["hr", "act"],
    "meals": list(MEAL_COLS),
    "tod": ["tod_sin", "tod_cos"],
}


def _period_channels(g: pd.DataFrame, m: pd.DataFrame, channels: list[str], k: int) -> np.ndarray:
    """Per-row channel matrix for one period's sorted grid; meals summed over trailing k rows."""
    feats: dict[str, np.ndarray] = {}
    real = g["glucose_is_real"].to_numpy()
    feats["glu"] = np.where(real, g["glucose"].to_numpy(dtype=float), np.nan)
    feats["glu_mask"] = real.astype(float)
    feats["hr"] = g["hr"].to_numpy(dtype=float)
    feats["act"] = g["activity_intensity"].to_numpy(dtype=float)
    mins = (g["t"].dt.hour * 60 + g["t"].dt.minute).to_numpy()
    feats["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    feats["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
    if any(c in channels for c in MEAL_COLS):
        tt = g["t"].to_numpy()
        idx = np.searchsorted(tt, m["t"].to_numpy(), side="left")  # first row at/after the meal
        ok = idx < len(g)
        for ch, col in MEAL_COLS.items():
            v = np.zeros(len(g))
            np.add.at(v, idx[ok], np.nan_to_num(m[col].to_numpy(dtype=float))[ok])
            cs = np.r_[0.0, np.cumsum(v)]
            j = np.arange(len(g))
            feats[ch] = cs[j + 1] - cs[np.maximum(j + 1 - k, 0)]
    return np.column_stack([feats[c] for c in channels]).astype(np.float32)


def build_sequences(
    grid: pd.DataFrame,
    meals: pd.DataFrame,
    anchors: pd.DataFrame,
    n_hours: int,
    grid_step: int,
    seq_step: int,
    channels: list[str],
) -> np.ndarray:
    """(n_anchors, L, C) causal sequences at times t - (L-1)*seq_step, ..., t."""
    seq_len = n_hours * 60 // seq_step
    k = seq_step // grid_step
    out = np.full((len(anchors), seq_len, len(channels)), np.nan, dtype=np.float32)
    offs = np.arange(seq_len - 1, -1, -1) * k
    for pid, rows in anchors.groupby("period_id").indices.items():
        g = grid[grid.period_id == pid]
        mat = _period_channels(g, meals[meals.period_id == pid], channels, k)
        pos = np.searchsorted(g["t"].to_numpy(), anchors["t"].to_numpy()[rows])
        ix = pos[:, None] - offs[None, :]
        ok = ix >= 0
        seq = np.full((len(rows), seq_len, len(channels)), np.nan, dtype=np.float32)
        seq[ok] = mat[ix[ok]]
        out[rows] = seq
    return out


@dataclass
class Scaler:
    """Per-channel mean/std fitted on training rows only; NaN -> 0 after scaling (= the mean)."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> Scaler:
        """Fit on (..., C) training data."""
        flat = x.reshape(-1, x.shape[-1])
        with np.errstate(all="ignore"):
            self.mean = np.nan_to_num(np.nanmean(flat, axis=0))
            sd = np.nan_to_num(np.nanstd(flat, axis=0))
        self.std = np.where(sd < 1e-6, 1.0, sd)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Standardise and replace NaN with 0."""
        return np.nan_to_num((x - self.mean) / self.std).astype(np.float32)


def make_model(n_dyn: int, n_static: int, hidden: int, dropout: float, n_out: int) -> Any:
    """GRU + FiLM + multi-horizon head (torch imported lazily)."""
    import torch
    from torch import nn

    class FiLMGRU(nn.Module):
        """GRU encoder; the static vector modulates the final state via FiLM."""

        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(n_dyn, hidden, batch_first=True)
            self.use_static = n_static > 0
            if self.use_static:
                self.film = nn.Sequential(
                    nn.Linear(n_static, hidden), nn.ReLU(), nn.Linear(hidden, 2 * hidden)
                )
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out)
            )

        def forward(self, seq: torch.Tensor, stat: torch.Tensor) -> torch.Tensor:
            _, h = self.gru(seq)
            h = h[-1]
            if self.use_static:
                gamma, beta = self.film(stat).chunk(2, dim=-1)
                h = (1 + gamma) * h + beta
            return self.head(h)

    return FiLMGRU()


@dataclass
class NeuralForecaster:
    """Train / predict wrapper; targets scaled per horizon on training rows; L1 loss."""

    hidden: int
    dropout: float
    cfg: dict[str, Any]
    seed: int
    target: str = "delta"

    def _static(self, s: np.ndarray) -> np.ndarray:
        if s.shape[1] == 0:
            return np.zeros((len(s), 0), np.float32)
        z = np.nan_to_num((s - self.st_mean) / self.st_std)
        return np.concatenate([z, np.isnan(s)], axis=1).astype(np.float32)

    def _y(self, y: np.ndarray, base: np.ndarray) -> np.ndarray:
        return y - base[:, None] if self.target == "delta" else y

    def fit(
        self,
        tr: tuple[np.ndarray, ...],
        va: tuple[np.ndarray, ...],
        w_tr: np.ndarray | None = None,
        w_va: np.ndarray | None = None,
    ) -> NeuralForecaster:
        """Fit with early stopping. ``tr``/``va`` = (seq, static, y, base) arrays.

        ``w_tr`` / ``w_va``: optional (n, n_horizons) loss weights (Part 5 onset-weighted loss,
        D-042); the validation loss for early stopping is then the weighted L1 as well. None
        (default) = the unweighted Part 3 loss, identical code path.
        """
        import torch

        seq_tr, st_tr, y_tr, base_tr = tr
        seq_va, st_va, y_va, base_va = va
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.cfg["threads"])
        self.scaler = Scaler().fit(seq_tr)
        if st_tr.shape[1]:
            with np.errstate(all="ignore"):
                self.st_mean = np.nan_to_num(np.nanmean(st_tr, axis=0))
                sd = np.nan_to_num(np.nanstd(st_tr, axis=0))
            self.st_std = np.where(sd < 1e-6, 1.0, sd)
        yt = self._y(y_tr, base_tr)
        self.y_mean, self.y_std = np.nanmean(yt, axis=0), np.nanstd(yt, axis=0)

        def t(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))

        xs_tr, ss_tr = t(self.scaler.transform(seq_tr)), t(self._static(st_tr))
        xs_va, ss_va = t(self.scaler.transform(seq_va)), t(self._static(st_va))
        yz_tr = t((yt - self.y_mean) / self.y_std)
        yz_va = t((self._y(y_va, base_va) - self.y_mean) / self.y_std)
        self.model = make_model(
            seq_tr.shape[2], ss_tr.shape[1], self.hidden, self.dropout, y_tr.shape[1]
        )
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"]
        )
        wt_tr = None if w_tr is None else t(w_tr)
        wt_va = None if w_va is None else t(w_va)
        gen = torch.Generator().manual_seed(self.seed)
        best, state, bad, bs = np.inf, None, 0, self.cfg["batch_size"]
        self.best_epoch, self.val_curve = 0, []
        for epoch in range(self.cfg["max_epochs"]):
            self.model.train()
            perm = torch.randperm(len(xs_tr), generator=gen)
            for i in range(0, len(perm), bs):
                b = perm[i : i + bs]
                y = yz_tr[b]
                ok = ~torch.isnan(y)
                p = self.model(xs_tr[b], ss_tr[b])
                if wt_tr is None:
                    loss = torch.nn.functional.l1_loss(p[ok], y[ok])
                else:
                    wb = wt_tr[b][ok]
                    loss = (wb * torch.abs(p[ok] - y[ok])).sum() / wb.sum().clamp_min(1e-6)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            v = self._val_loss(xs_va, ss_va, yz_va, wt_va)
            self.val_curve.append(v)
            if v < best - 1e-4:
                best, bad, self.best_epoch = v, 0, epoch
                state = {k: x.clone() for k, x in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    break
        self.model.load_state_dict(state)
        self.best_val = best
        return self

    def _val_loss(self, xs: Any, ss: Any, yz: Any, w: Any = None) -> float:
        import torch

        self.model.eval()
        tot, n = 0.0, 0.0
        with torch.no_grad():
            for i in range(0, len(xs), 4096):
                y = yz[i : i + 4096]
                ok = ~torch.isnan(y)
                p = self.model(xs[i : i + 4096], ss[i : i + 4096])
                if w is None:
                    tot += float(torch.abs(p[ok] - y[ok]).sum())
                    n += int(ok.sum())
                else:
                    wb = w[i : i + 4096][ok]
                    tot += float((wb * torch.abs(p[ok] - y[ok])).sum())
                    n += float(wb.sum())
        return tot / max(n, 1)

    def predict(self, seq: np.ndarray, st: np.ndarray, base: np.ndarray) -> np.ndarray:
        """(n, n_horizons) absolute glucose forecasts."""
        import torch

        self.model.eval()
        xs = torch.from_numpy(self.scaler.transform(seq))
        ss = torch.from_numpy(self._static(st))
        with torch.no_grad():
            raw = np.concatenate(
                [
                    self.model(xs[i : i + 4096], ss[i : i + 4096]).numpy()
                    for i in range(0, len(xs), 4096)
                ]
            )
        raw = raw * self.y_std + self.y_mean
        return raw + base[:, None] if self.target == "delta" else raw
