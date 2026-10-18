"""Part 10: candidate list, ensemble averaging, lock rule (no LightGBM needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.improve import run as imp
from src.models.tuned import OBJECTIVES, TunedLGBM, lgb_params, trial_params
from src.utils.config import load_config


def cfg() -> dict:
    return load_config("improve")["tuning"]


def test_candidates_are_deterministic_and_start_with_status_quo():
    a, b = trial_params(cfg()), trial_params(cfg())
    assert a == b
    assert len(a) == cfg()["n_trials"]
    assert a[0] == cfg()["status_quo"]


def test_candidates_stay_inside_the_space():
    space = cfg()["space"]
    for t in trial_params(cfg())[1:]:
        assert all(t[k] in space[k] for k in space)
        assert t["objective"] in OBJECTIVES


def test_lgb_params_use_mae_metric_and_objective_mapping():
    t = {**cfg()["status_quo"], "objective": "huber25"}
    p = lgb_params(t, cfg(), seed=7, n_jobs=2)
    assert p["metric"] == "l1" and p["objective"] == "huber" and p["alpha"] == 25.0
    assert p["random_state"] == 7 and "n_estimators" in p


class _Const:
    def __init__(self, v: float):
        self.v = v

    def predict(self, x):
        return np.full(len(x), self.v)


def test_prediction_is_member_average_and_decodes_delta():
    m = TunedLGBM(["a"], {}, base="cgm_now", target="delta", tcfg=cfg())
    m.members = {60: [_Const(2.0), _Const(4.0), _Const(6.0)]}
    tab = pd.DataFrame({"a": [0.0, 1.0], "cgm_now": [100.0, 150.0]})
    out = m.predict(tab, horizons=[60])[60]
    assert out.tolist() == [104.0, 154.0]


def test_evaluate_pairs_models_on_identical_rows():
    rng = np.random.default_rng(0)
    pid = np.repeat([f"p{i}" for i in range(8)], 20)
    t = np.tile(np.arange(20), 8)
    tab = pd.DataFrame({"patient_id": pid, "period_id": 0, "t": t, "cgm_now": 120.0})
    tab["y_60"] = 120.0 + rng.normal(0, 10, len(tab))
    keys = tab[["patient_id", "period_id", "t"]]
    sq = keys.assign(pred_60=tab["cgm_now"] + 5.0)
    tu = keys.assign(pred_60=tab["cgm_now"])
    boot = {"n_boot": 50, "ci": 0.95}
    r = imp.evaluate(sq, tu, tab, [60], boot)
    d = r["diff"][0]
    assert d["model_1"] == "tuned" and d["model_2"] == "status_quo"
    assert d["diff_mae"] < 0  # the unbiased predictor is better
    assert r["n_patients"] == 8


def test_holdout_refuses_second_run(tmp_path, monkeypatch):
    (tmp_path / "improve_holdout.json").write_text("{}")
    monkeypatch.setattr(imp, "METRICS", tmp_path)
    with pytest.raises(SystemExit):
        imp.stage_holdout()
