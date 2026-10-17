"""Part 8c, D-063a: EXPLORATORY privacy-utility sweep (added after the first federated run).

Same layouts, splits, seeds, clip and FedAvg reference as ``src.federated.run``; only the noise
multiplier of the client-level DP runs is swept. Writes reports/metrics/federated_privacy.json.
Run: ``python -W ignore -u -m src.federated.privacy_sweep`` (about 10 minutes).
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from src.federated import core as fc
from src.federated.run import OUT, PROC, load_data, one_split
from src.transfer import core as tc
from src.utils.config import load_config
from src.utils.provenance import file_hash, write_metrics


def main() -> None:
    """Run the sweep for every layout and write the metrics file."""
    cfg, tcfg, bc = load_config("federated"), load_config("transfer"), load_config("baselines")
    rows, ev, pool, cols = load_data(cfg, tcfg)
    zs = list(cfg["sweep_noise_multipliers"])
    hz = cfg["horizons_min"]
    models = ["persistence", "fedavg", *[f"fedavg_dp_z{z:g}" for z in zs]]
    pairs = [(f"fedavg_dp_z{z:g}", "fedavg") for z in zs]
    metrics: dict[str, Any] = {
        "EXPLORATORY": "D-063a: noise levels chosen after the first run; a curve, not a tuned model",
        "SIMULATION": "clinics are groups of open-cohort patients",
        "epsilon": {f"z{z:g}": fc.epsilon_gaussian(cfg["train"]["rounds"], z, cfg["dp"]["delta"])
                    for z in zs},
        "delta": cfg["dp"]["delta"], "clip": cfg["dp"]["clip"], "rounds": cfg["train"]["rounds"],
    }  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    for name, layout in cfg["layouts"].items():
        t0 = time.time()
        parts = [one_split(layout, i, pool, rows, ev, cols, cfg, sweep_z=zs) for i in range(cfg["n_splits"])]
        res: pd.DataFrame = pd.concat(parts, ignore_index=True)
        res.to_parquet(OUT / f"sweep_{name}.parquet", index=False)
        metrics[name] = {"clinics": layout["clinics"],
                         "mae": tc.mae_rows(res, models, hz, bc["bootstrap"]),
                         "diff": tc.diff_rows(res, pairs, hz, bc["bootstrap"])}  # fmt: skip
        print(f"  layout {name}: {time.time() - t0:.0f}s", flush=True)
    files = [PROC / "static.parquet"] + [
        PROC / "features" / f"{n}_N{tcfg['n_hours']}.parquet" for n in tcfg["tables"].values()
    ]
    write_metrics("federated_privacy", metrics, cfg, cfg["seed"], file_hash(files))
    print("wrote reports/metrics/federated_privacy.json")


if __name__ == "__main__":
    main()
