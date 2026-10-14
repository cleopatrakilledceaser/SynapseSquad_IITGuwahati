"""Part 5 runner (``make hybrid``): twin features, hybrids A0 / A / B / C, neural onset fix.

Stages (cached in data/processed/hybrid/, skipped if present):
  features  causal twin features per fold and k (both cohorts; src.hybrid.build)
  lgbm      Hybrid A0 / A / A-minus-activity / B (CGMacros, k = 0/1/3)
  pinn      Hybrid C (physics-informed GRU, CGMacros, k = 3, 5 seeds)
  onset     Part 3 GRU + FiLM with an onset-weighted loss (both cohorts, last)
Usage: python -m src.hybrid.run_hybrid [features|lgbm|pinn|onset|all]
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from joblib import Parallel, delayed

from src.hybrid.build import OUT, fold_features
from src.mech.run_mech import make_ctx
from src.models.fusion_lgbm import folds
from src.utils.config import load_config
from src.utils.seed import set_global_seed

COHORTS = ["cgmacros", "shanghai_t2dm"]


def _feature_job(cohort: str, fold: int, train: list[str], test: list[str]) -> None:
    path = OUT / f"twinfeat_{cohort}_fold{fold}.parquet"
    if path.exists():
        return
    t0 = time.time()
    hc, mc, fc = load_config("hybrid"), load_config("mech"), load_config("fusion")
    c = make_ctx(cohort, mc, fc)
    train = [p for p in train if p in c.by_pat]
    test = [p for p in test if p in c.by_pat]
    f, tr, fits, ch = fold_features(c, fold, train, test, hc)
    num = [x for x in f.columns if x.startswith("tw_")]
    f[num] = f[num].astype(np.float32)
    if len(tr):
        tc = [x for x in tr.columns if x.startswith("tr_")]
        tr[tc] = tr[tc].astype(np.float32)
        tr.to_parquet(OUT / f"traj_{cohort}_fold{fold}.parquet", index=False)
    fits.to_parquet(OUT / f"twinfeatfits_{cohort}_fold{fold}.parquet", index=False)
    ch["seconds"] = round(time.time() - t0, 1)
    (OUT / f"twinfeat_{cohort}_fold{fold}.json").write_text(json.dumps(ch, indent=1))
    f.to_parquet(path, index=False)  # written last: marks the fold as complete
    print(f"  {path.name}: {len(f)} rows, {time.time() - t0:.0f}s", flush=True)


def stage_features(n_jobs: int = 5) -> None:
    """Twin features for both cohorts, folds in parallel processes."""
    hc = load_config("hybrid")
    jobs = [(c, f, tr, te) for c in COHORTS for _, f, tr, te in folds(c, [hc["cv_repeat"]])]
    Parallel(n_jobs=n_jobs)(delayed(_feature_job)(*j) for j in jobs)


def main() -> None:
    """Run the requested stage(s)."""
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    set_global_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    todo = ["features", "lgbm", "pinn", "onset"] if stage == "all" else [stage]
    for s in todo:
        t0 = time.time()
        print(f"[{s}]", flush=True)
        if s == "features":
            stage_features()
        elif s == "lgbm":
            from src.hybrid.models import stage_lgbm

            stage_lgbm()
        elif s == "pinn":
            from src.hybrid.pinn import stage_pinn

            stage_pinn()
        elif s == "onset":
            from src.hybrid.onset import stage_onset

            stage_onset()
        print(f"[{s}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
