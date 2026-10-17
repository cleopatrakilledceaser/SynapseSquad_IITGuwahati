"""Part 9 demo artifact builder (D-058): writes ``app/demo`` (the offline app bundle).

No test-patient outcome label (``y_*`` / ``*_any_*``) is ever scored, read for display, or stored:
every stored model output is an out-of-fold forecast of a Part 6 / Part 7 fold model that never saw
the demo patient, and calibration / alert thresholds are fitted on the fold's TRAINING patients
only. The one read of a test patient's rows projects to the ``patient_id`` column alone.

Written (all under ``app/demo``):
  models/           both forms' boosters + OOF residuals (export.export_models; read by appdata.rt)
  calib.npz/json    event calibration maps + alert thresholds (training OOF only)
  stats.json        stored Part 3/5/6/7 cohort numbers for the reference pages
  global_shap.json  mean |SHAP| per feature and horizon over the DISPLAYED windows
  patients.json     labels, status, profile, twin Gb, per-patient metadata
  manifest.json     build provenance, display config echo, guards
  figures/          the evidence figures
  patients/<pid>/   replay.parquet (anchors: forecasts, CV+ band, event probs, alerts, quality,
                    confidence, model inputs), explain.parquet (template sentences + reasons),
                    shap.npz, grid.parquet + meals.parquet (causal feature window, truncated at
                    the last replay anchor = the app's live what-if state), twin.npz (what-if
                    twin trajectories on the app's carb / slow-ratio grid) + whatif.json

Guards run on every build (recorded in manifest.json):
  rtform_max_abs_diff          shipped text models reproduce in-memory predictions and bands
  recompute_full_max_abs_diff  the truncated live window reproduces the stored feature rows
  interp_max_abs_diff          linear interpolation of the twin grid vs the engine
  labels_in_outputs            no outcome column ever appears in an output frame

Usage:  python -m src.appdata.build [--check]
  --check  guards and numbers only, nothing is written
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.appdata import events as ev
from src.appdata import export as ex
from src.appdata import replay as rp
from src.appdata import rt
from src.appdata import synthetic as sy
from src.appdata import whatif_grid as wg
from src.explain import shap_values as shap
from src.explain import templates as tp
from src.hybrid.twin_features import HORIZONS
from src.uncertainty import alarm_eval as ae
from src.uncertainty.common import LABEL_COLS, groups
from src.utils.config import PROJECT_ROOT, load_config
from src.whatif import context as cx
from src.whatif import edits as ed
from src.whatif.engine import Period, PROC, twin_specs

APP = PROJECT_ROOT / "app" / "demo"
GEN = PROJECT_ROOT / "logs"
COHORT = "cgmacros"
ALPHA80 = 0.2
CLIP = (40.0, 405.0)
KEYS = ["patient_id", "period_id", "t"]
VAR_ORDER = ("twin_learned", "twin_assumed")  # rows of every precomputed twin grid
LABEL_MARKERS = ("_any_",)  # outcome-column name markers forbidden in every output frame
EVENTS_ALL = ("hyper180", "hyper140", "hypo70")  # event probabilities shown in the app


# --------------------------------------------------------------------------- config

def load_cfg() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    fcfg = load_yaml(PROJECT_ROOT / "configs" / "features.yaml")
    whatif = load_yaml(PROJECT_ROOT / "configs" / "whatif.yaml")
    unc = load_yaml(PROJECT_ROOT / "configs" / "uncertainty.yaml")
    app = load_yaml(PROJECT_ROOT / "configs" / "app.yaml")


# --------------------------------------------------------------------------- io helpers

def sync_dir(dst: Path, src: Path) -> None:
    """Copy a directory tree, deleting files in dst that are not in src (deterministic)."""
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(dst.rglob("*")):
        if p.is_file():
            p.unlink()
    for p in sorted(src.rglob("*")):
        if p.is_file():
            (dst / p.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / p.relative_to(src))


def gzip_json(obj: dict[str, Any], path: Path) -> None:
    buf = gzip.compress(json.dumps(obj, ensure_ascii=False).encode("utf-8"), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf)


def clean_pandas(obj: Any) -> Any:
    """JSON-serialisable mirror of a small numpy value (None/float/int/str)."""
    if isinstance(obj, dict):
        return {str(k): clean_pandas(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_pandas(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None or isinstance(obj, bool) or isinstance(obj, (int, float, str)):
        return obj
    return str(obj)

    return fcfg, whatif, unc, app
