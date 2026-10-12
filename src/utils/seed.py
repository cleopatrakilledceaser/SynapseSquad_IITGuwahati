"""Global seeding for reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and the hash seed of child processes.

    Args:
        seed: Seed value (project default 42, from ``configs/base.yaml``).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Legacy global state is seeded on purpose for third-party code that still uses it.
    # Project code must use np.random.default_rng(seed) instead.
    np.random.seed(seed)  # noqa: NPY002
