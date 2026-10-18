"""D-072: the tuned population ensemble inside the Part 6 / Part 7 / app machinery."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.appdata import rt
from src.explain import shap_values as shap
from src.personalize.data import PopEns, fit_fixed, fit_pop_tuned
from src.uncertainty import models as um
from src.utils.config import load_config

COLS = ["cgm_now", "a", "b"]


def frame(n: int = 1500, seed: int = 0, n_pat: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(
        {
            "patient_id": [f"p{i % n_pat}" for i in range(n)],
            "cgm_now": rng.normal(140, 30, n),
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "days_since_start": np.tile(np.linspace(0, 8, n // n_pat + 1), n_pat)[:n],
        }
    )
    for h in (30, 60, 120):
        d[f"y_{h}"] = d["cgm_now"] + 8 * d["a"] * h / 60 - 3 * d["b"] + rng.normal(0, 5, n)
    return d


def small_tcfg(**extra) -> dict:
    t = load_config("improve")["tuning"]
    return {
        **t,
        "n_trials": 3,
        "top_k": 2,
        "fixed": {**t["fixed"], "n_estimators": 60},
        "early_stopping_rounds": 10,
        **extra,
    }


def bc() -> dict:
    return load_config("baselines")


@pytest.fixture(scope="module")
def pop() -> dict[int, PopEns]:
    d = frame()
    out, _ = fit_pop_tuned(d.iloc[:1000], d.iloc[1000:], COLS, bc(), small_tcfg(), 1)
    return out


def test_ensemble_forecast_is_base_plus_member_mean(pop):
    d = frame(200, seed=1)
    p = pop[60]
    assert isinstance(p, PopEns) and len(p.members) == 2
    want = d["cgm_now"].to_numpy() + np.mean([m.raw(d) for m in p.members], axis=0)
    assert np.allclose(p.predict(d), want)


def test_fixed_refit_keeps_each_members_trees_and_leaves(pop):
    ref = fit_fixed(frame(800, seed=2), pop, bc(), 1)
    for h in (30, 60, 120):
        assert isinstance(ref[h], PopEns)
        for a, b in zip(ref[h].members, pop[h].members, strict=True):
            assert a.n_iter == max(b.n_iter, 1) and a.leaves == b.leaves
            assert a.est.booster_.num_trees() == a.n_iter


def test_runtime_text_reproduces_the_ensemble(pop):
    d = frame(300, seed=3)
    forest = rt.parse_model_text(pop[120].model_str())
    got = d["cgm_now"].to_numpy() + forest.predict(d[COLS].to_numpy(float))
    assert np.max(np.abs(got - pop[120].predict(d))) < 1e-6


def test_ensemble_finetune_is_member_mean(pop):
    d = frame(600, seed=4)
    pc = load_config("personalize")
    mp = pop[60]
    a = d["days_since_start"].to_numpy() < 3
    y = d["y_60"].to_numpy() - d["cgm_now"].to_numpy()
    prm = um.pm.finetune_params(bc()["lightgbm"]["params"], pc["finetune"], 0.05)
    bs = um.ft_fit(mp, d.loc[a], y[a], prm, 5)
    assert len(bs) == len(mp.members)
    e = d.loc[~a]
    single = [
        b.predict(e[m.cols], start_iteration=m.n_iter, num_iteration=5)
        for b, m in zip(bs, mp.members, strict=True)
    ]
    assert np.allclose(um.ft_extra(mp, bs, e, 5), np.mean(single, axis=0))


def test_shap_of_ensemble_adds_up_to_the_predicted_delta(pop):
    d = frame(100, seed=5)
    phi, base = shap.contributions(pop[30], d)
    assert np.allclose(phi.sum(axis=1).to_numpy() + base, pop[30].raw(d), atol=1e-6)
    assert set(shap.used_features(pop[30])) <= set(COLS)


def test_monotone_constraints_reach_every_member():
    d = frame()
    b = bc()
    b = {**b, "lightgbm": {**b["lightgbm"], "params": {**b["lightgbm"]["params"]}}}
    b["lightgbm"]["params"]["monotone_constraints"] = [0, 1, 0]
    out, _ = fit_pop_tuned(d.iloc[:1000], d.iloc[1000:], COLS, b, small_tcfg(), 1)
    for m in out[60].members:
        assert m.est.get_params()["monotone_constraints"] == [0, 1, 0]
    x = frame(50, seed=6)
    lo, hi = x.assign(a=-1.0), x.assign(a=1.0)
    assert np.all(out[60].predict(hi) >= out[60].predict(lo) - 1e-9)


def test_forced_rerun_keeps_the_lock_log(tmp_path, monkeypatch):
    from src.whatif import run as wr

    lock = tmp_path / "FINAL_DONE"
    lock.write_text(json.dumps({"runs": 2, "forced_reruns": 1, "forced_rerun_log": ["1: x"]}))
    monkeypatch.setattr(wr, "OUT", tmp_path)
    monkeypatch.setattr(wr, "LOCK", lock)
    wr.write_lock(2, True, "D-072")
    got = json.loads(lock.read_text())
    assert got["runs"] == 3 and got["forced_reruns"] == 2
    assert got["forced_rerun_log"] == ["1: x", "2: final stage re-run: D-072"]
