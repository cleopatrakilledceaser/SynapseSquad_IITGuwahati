"""India meal library (D-054): assumed recipes -> per-portion macros from IFCT 2017.

The repo stores only our assumed recipes (IFCT codes + raw grams) and the resulting per-portion
macros (``src/india/meals.json``). The IFCT 2017 table is fetched at run time into the gitignored
``data/external/`` and never committed (ifct2017 package is AGPL-3.0; values are NIN's).
Usage: python -m src.india.library build | check
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import PROJECT_ROOT

LIB = Path(__file__).with_name("meals.json")
CACHE = PROJECT_ROOT / "data" / "external" / "ifct2017_compositions.csv"
URL = "https://unpkg.com/@ifct2017/compositions/index.csv"
COLS = {"choavldf": "carbs_g", "protcnt": "protein_g", "fatce": "fat_g", "fibtg": "fiber_g"}
MACROS = ("carbs_g", "protein_g", "fat_g", "fiber_g", "kcal")
KJ_PER_KCAL = 4.184
CHECK_TOL = 0.05  # g / kcal rounding tolerance of the stored per-portion values
SRC_IFCT = "IFCT2017-derived from an assumed recipe"
SRC_APPROX = "approximate assumption (partly IFCT2017-derived)"
SOURCES = (SRC_IFCT, SRC_APPROX)


def load_library(path: Path = LIB) -> dict[str, Any]:
    """Library JSON, schema-checked."""
    lib = json.loads(path.read_text(encoding="utf-8"))
    validate(lib)
    return lib


def validate(lib: dict[str, Any]) -> None:
    """Raise ValueError if the library does not follow the D-054 schema."""
    for k in ("disclaimer", "source", "presets"):
        if k not in lib:
            raise ValueError(f"library missing '{k}'")
    if "not clinical" not in lib["disclaimer"].lower():
        raise ValueError("disclaimer must say the values are not clinical")
    for name, p in lib["presets"].items():
        for k in ("label", "portion", "ingredients"):
            if k not in p:
                raise ValueError(f"{name}: missing '{k}'")
        for ing in p["ingredients"]:
            if not (2 <= len(ing) <= 3 and isinstance(ing[0], str) and float(ing[1]) > 0):
                raise ValueError(f"{name}: bad ingredient {ing}")
            if len(ing) == 3 and ing[2] not in lib.get("approximations", {}):
                raise ValueError(f"{name}: undocumented approximation '{ing[2]}'")
        if "macros" in p:
            m = p["macros"]
            if set(m) != set(MACROS) or any(float(m[k]) < 0 for k in MACROS):
                raise ValueError(f"{name}: macros must be non-negative {MACROS}")
            if p.get("source") not in SOURCES:
                raise ValueError(f"{name}: bad source label")
        if any(k.startswith(("per_100", "per100")) for k in p):
            raise ValueError(f"{name}: per-100 g values must not be stored (D-054)")


def fetch_ifct(cache: Path = CACHE) -> pd.DataFrame:
    """IFCT 2017 compositions (code-indexed), fetched once into data/external/ (gitignored)."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, cache)  # noqa: S310 (fixed https URL)
    d = pd.read_csv(cache)
    d.columns = [c.split("; ")[-1] for c in d.columns]
    return d.set_index("code")


def portion_macros(ingredients: list, ifct: pd.DataFrame) -> dict[str, float]:
    """Per-portion carbs / protein / fat / fiber (g) and kcal for raw-ingredient grams."""
    out = dict.fromkeys(MACROS, 0.0)
    for ing in ingredients:
        code, grams = ing[0], float(ing[1])
        if code not in ifct.index:
            raise KeyError(f"IFCT code {code} not found")
        r = ifct.loc[code]
        f = grams / 100.0
        for src, dst in COLS.items():
            out[dst] += f * float(pd.to_numeric(r[src], errors="coerce") or 0.0)
        out["kcal"] += f * float(r["enerc"]) / KJ_PER_KCAL
    return {k: round(v, 1) for k, v in out.items()}


def build(path: Path = LIB) -> dict[str, Any]:
    """Compute per-portion macros + source labels and write them into the library JSON."""
    lib = json.loads(path.read_text(encoding="utf-8"))
    ifct = fetch_ifct()
    for p in lib["presets"].values():
        p["macros"] = portion_macros(p["ingredients"], ifct)
        approx = any(len(i) == 3 for i in p["ingredients"])
        p["source"] = SRC_APPROX if approx else SRC_IFCT
        p["ifct_codes"] = sorted({i[0] for i in p["ingredients"]})
    validate(lib)
    path.write_text(json.dumps(lib, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return lib


def check(path: Path = LIB) -> float:
    """Max |stored - recomputed| over presets / macros (raises above CHECK_TOL)."""
    lib = load_library(path)
    ifct = fetch_ifct()
    worst = 0.0
    for name, p in lib["presets"].items():
        m = portion_macros(p["ingredients"], ifct)
        d = max(abs(m[k] - float(p["macros"][k])) for k in MACROS)
        if d > CHECK_TOL:
            raise RuntimeError(f"{name}: stored macros differ from IFCT by {d}")
        worst = max(worst, d)
    return worst


def main() -> None:
    """CLI: build | check."""
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "build":
        lib = build()
        for n, p in lib["presets"].items():
            print(f"{n:26s} {p['macros']}")
    else:
        print(f"max |stored - IFCT| = {check():.3f}")


if __name__ == "__main__":
    main()
