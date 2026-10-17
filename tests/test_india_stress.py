"""Part 8b helpers (D-062): 15-min grid, feature selection, synthetic anchor table, bias."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.india import cohort as ic
from src.india import stress
from src.utils.config import PROJECT_ROOT, load_config

SCFG = {"feature_groups": ["cgm", "tod", "wearable", "meals", "static"], "static": ["age_years", "bmi"]}


def test_feature_columns_shared_only_and_no_targets():
    real = ["cgm_now", "tod_sin", "tod_cos", "hr_mean_1h", "meal_minutes_since_last", "static_age_years",
            "static_fasting_insulin", "y_60", "days_since_start", "hyper180_60"]  # fmt: skip
    syn = [c for c in real if c != "static_fasting_insulin"]
    got = stress.feature_columns(real, syn, SCFG)
    assert got == ["cgm_now", "tod_sin", "tod_cos", "hr_mean_1h", "meal_minutes_since_last", "static_age_years"]


def test_feature_columns_respects_groups():
    cols = ["cgm_now", "tod_sin", "hr_mean_1h", "meal_minutes_since_last", "static_age_years"]
    got = stress.feature_columns(cols, cols, {**SCFG, "feature_groups": ["cgm", "tod"]})
    assert got == ["cgm_now", "tod_sin"]


def test_constant_columns():
    t = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3], "c": [np.nan] * 3})
    assert stress.constant_columns(t, ["a", "b", "c"]) == ["a", "c"]


def test_bias_rows_sign():
    df = pd.DataFrame({"pred_m_30": [110.0, 120.0], "y_30": [100.0, 100.0]})
    r = stress.bias_rows(df, ["m"], [30])[0]
    assert r["bias"] == pytest.approx(15.0) and r["n_rows"] == 2


def _need_inputs():
    if not (PROJECT_ROOT / "reports" / "metrics" / "mech_fitted_params.csv").exists():
        pytest.skip("mechanistic fits not available")


def test_to_grid_is_15_min_and_causal():
    _need_inputs()
    icfg, mc = load_config("india"), load_config("mech")
    c = ic.generate(icfg, mc, 1.0, "t", 1, n_patients=2, days=2)
    g = ic.to_grid(c, 15)
    assert (g.groupby("patient_id")["minutes_since_start"].diff().dropna() == 15).all()
    # trailing wearable mean: the first row equals the raw first row (no future values used)
    p0 = c["grid"][c["grid"].patient_id == c["grid"].patient_id.iloc[0]].reset_index(drop=True)
    g0 = g[g.patient_id == p0.patient_id.iloc[0]].reset_index(drop=True)
    assert g0["hr"].iloc[0] == pytest.approx(p0["hr"].iloc[0])
    assert g0["hr"].iloc[1] == pytest.approx(p0["hr"].iloc[1:4].mean())


def test_synthetic_table_builds_and_has_static():
    _need_inputs()
    icfg, mc, fcfg = load_config("india"), load_config("mech"), load_config("features")
    scfg = {**icfg["stress"], "n_hours": 6}
    c = ic.generate(icfg, mc, 1.0, "t", 1, n_patients=2, days=6)
    tab = stress.synthetic_table(c, icfg, fcfg, scfg)
    assert {"cgm_now", "y_60", "static_hba1c_pct", "tod_sin"} <= set(tab.columns)
    assert tab["static_age_years"].notna().all() and len(tab) > 100
