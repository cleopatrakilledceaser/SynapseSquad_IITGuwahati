"""Part 8a helpers (D-061): shared features, evaluation region, causal few-shot correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.transfer import core

CFG = {
    "tod_columns": ["tod_sin", "tod_cos"],
    "common_static": ["age_years", "bmi"],
    "feature_sets": {"cgm_tod": ["cgm", "tod"], "cgm_tod_static": ["cgm", "tod", "static"]},
}
FS = {"min_pairs": 24, "offset_n0": 288, "ridge_lambda": 1000.0,
      "ridge_features": ["cgm_now", "cgm_delta_30", "tod_sin", "tod_cos"]}  # fmt: skip


def table(n_pat: int = 3, days: int = 6, bias: float = 20.0, seed: int = 0) -> pd.DataFrame:
    """15-min anchors for a few patients; the 'population forecast' is off by ``bias``."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        t = pd.date_range("2026-01-01", periods=days * 96, freq="15min")
        d = np.arange(len(t)) / 96.0
        g = 130 + 20 * np.sin(np.arange(len(t)) / 7) + rng.normal(0, 2, len(t))
        df = pd.DataFrame({"patient_id": f"p{p}", "period_id": f"p{p}_0", "t": t,
                           "days_since_start": d, "cgm_now": g,
                           "cgm_delta_30": np.r_[0, 0, np.diff(g, 2)],
                           "tod_sin": np.sin(2 * np.pi * d), "tod_cos": np.cos(2 * np.pi * d)})  # fmt: skip
        for h in (30, 60, 120):
            df[f"y_{h}"] = np.r_[g[h // 15 :], [np.nan] * (h // 15)]
        df["pred_zs_60"] = df["y_60"] - bias  # population forecast too low by `bias`
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_feature_sets_use_only_shared_columns() -> None:
    a = [
        "cgm_now",
        "cgm_lag_5",
        "cgm_lag_15",
        "tod_sin",
        "tod_cos",
        "static_age_years",
        "static_bmi",
    ]
    b = ["cgm_now", "cgm_lag_15", "tod_sin", "tod_cos", "static_age_years", "static_bmi", "meal_x"]
    sets, dropped = core.feature_sets(a, b, CFG)
    assert sets["cgm_tod"] == ["cgm_lag_15", "cgm_now", "tod_sin", "tod_cos"]
    assert sets["cgm_tod_static"][-2:] == ["static_age_years", "static_bmi"]
    assert dropped == ["cgm_lag_5"]
    with pytest.raises(ValueError):
        core.feature_sets(a, b[:-3], CFG)  # static_bmi missing in b


def test_eval_table_keeps_first_period_and_enough_anchors() -> None:
    tab = table(n_pat=2)
    short = tab[tab.patient_id == "p1"].head(150)  # too few anchors after day 3
    later = tab[tab.patient_id == "p0"].copy()
    later["period_id"], later["t"] = "p0_1", later["t"] + pd.Timedelta(days=30)
    ev = core.eval_table(pd.concat([tab[tab.patient_id == "p0"], short, later]), 3, 96)
    assert set(ev.patient_id) == {"p0"} and set(ev.period_id) == {"p0_0"}


def test_offset_correction_removes_a_constant_bias() -> None:
    tab = table(bias=20.0)
    out = core.fewshot(tab, "pred_zs_60", 60, 3.0, "offset", FS)
    before = (tab["y_60"] - tab["pred_zs_60"]).abs().mean()
    after = (tab["y_60"] - out).abs().mean()
    assert before == pytest.approx(20.0) and after < 0.7 * before  # shrunk, still much lower


def test_ridge_correction_runs_and_reduces_error() -> None:
    tab = table(bias=20.0)
    out = core.fewshot(tab, "pred_zs_60", 60, 3.0, "ridge", FS)
    assert (tab["y_60"] - out).abs().mean() < (tab["y_60"] - tab["pred_zs_60"]).abs().mean()


def test_correction_never_uses_targets_after_day_k() -> None:
    tab = table()
    base = core.fewshot(tab, "pred_zs_60", 60, 1.0, "offset", FS)
    changed = tab.copy()
    late = changed["days_since_start"] + 60 / 1440 >= 1.0  # target time at or after day k
    changed.loc[late, "y_60"] += 500.0  # corrupt every target outside the adaptation pairs
    again = core.fewshot(changed, "pred_zs_60", 60, 1.0, "offset", FS)
    early = changed["days_since_start"] + 60 / 1440 < 1.0
    # the correction at rows after day k must be identical (the same few adaptation pairs)
    shift_base = base - tab["pred_zs_60"]
    shift_again = again - changed["pred_zs_60"]
    assert np.allclose(shift_base[~early], shift_again[~early], equal_nan=True)


def test_too_few_pairs_means_no_correction() -> None:
    tab = table()
    out = core.fewshot(tab, "pred_zs_60", 60, 0.1, "offset", FS)  # 0.1 day = 9 pairs < 24
    assert np.allclose(out, tab["pred_zs_60"], equal_nan=True)


def test_mae_and_diff_tables() -> None:
    tab = table()
    tab["pred_a_60"], tab["pred_b_60"] = tab["y_60"] + 10, tab["y_60"] + 20
    boot = {"n_boot": 50, "ci": 0.95}
    r = core.mae_rows(tab, ["a", "b"], [60], boot)
    assert [round(x["mae"]) for x in r] == [10, 20]
    d = core.diff_rows(tab, [("a", "b")], [60], boot)
    assert d[0]["diff_mae"] == pytest.approx(-10.0)
