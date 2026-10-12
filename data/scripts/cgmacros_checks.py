"""Part 1B verification checks on the raw CGMacros zip (read-only; prints short summaries).

Reads ``data/raw/cgmacros/CGMacros_dateshifted365.zip`` in place (no extraction, no copies) and
prints compact answers to the five Part 1B checks. Results are summarised in docs/data_card.md.

Run from the repo root: ``python data/scripts/cgmacros_checks.py``
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ZIP_PATH = ROOT / "data" / "raw" / "cgmacros" / "CGMacros_dateshifted365.zip"
NIGHT_HOURS = (1, 5)  # hours (inclusive) treated as night for the day/night HR test
DAY_HOURS = (10, 20)
LB_TO_KG = 0.45359237
IN_TO_M = 0.0254


def participant_files(z: zipfile.ZipFile) -> dict[str, str]:
    """Map participant id ('001') to its per-participant CSV inside the zip."""
    out: dict[str, str] = {}
    for n in z.namelist():
        m = re.search(r"CGMacros-(\d{3})\.csv$", n)
        if m and "__MACOSX" not in n:
            out[m.group(1)] = n
    return dict(sorted(out.items()))


def summarise_participant(pid: str, df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.Series]:
    """Return (summary record, meal rows, mean HR by hour) for one participant."""
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.assign(ts=ts)
    step = ts.diff().dt.total_seconds().div(60)
    rec: dict = {
        "pid": pid,
        "n_rows": len(df),
        "days": round((ts.max() - ts.min()).total_seconds() / 86400, 2),
        "row_step_min": float(step.mode().iloc[0]),
        "dup_ts": int(ts.duplicated().sum()),
        "nonmono": int((step < 0).sum()),
        "sleep_cols": [c for c in df.columns if "sleep" in c.lower()],
    }
    for key, col in (("libre", "Libre GL"), ("dexcom", "Dexcom GL"), ("hr", "HR")):
        s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
        rec[f"{key}_n"] = int(s.notna().sum())
        if key != "hr" and s.notna().any():
            rec[f"{key}_step"] = float(
                ts[s.notna()].diff().dt.total_seconds().div(60).mode().iloc[0]
            )
            rec[f"{key}_min"], rec[f"{key}_max"] = float(s.min()), float(s.max())
    rec["hr_cov"] = round(rec["hr_n"] / max(len(df), 1), 3)
    mets_col = next((c for c in df.columns if c.lower() == "mets"), None)
    rec["mets_n"] = int(df[mets_col].notna().sum()) if mets_col else 0
    meals = df.loc[df["Meal Type"].notna()].assign(pid=pid) if "Meal Type" in df else df.iloc[0:0]
    rec["meals"] = len(meals)
    hr = pd.DataFrame({"hour": ts.dt.hour, "hr": pd.to_numeric(df.get("HR"), errors="coerce")})
    by_hour = hr.dropna().groupby("hour")["hr"].mean().rename(pid)
    return rec, meals, by_hour


def check_time_of_day(meals: pd.DataFrame, hr_hour: pd.DataFrame) -> None:
    """Check 1: meal-start clustering by meal type, and day vs night heart rate."""
    print("\n[check 1: time of day]")
    hour = meals["ts"].dt.hour + meals["ts"].dt.minute / 60
    q = hour.groupby(meals["Meal Type"].str.strip().str.title()).describe(
        percentiles=[0.1, 0.5, 0.9]
    )
    print(q[["count", "10%", "50%", "90%"]].round(1).to_string())
    night = hr_hour.loc[NIGHT_HOURS[0] : NIGHT_HOURS[1]].mean()
    day = hr_hour.loc[DAY_HOURS[0] : DAY_HOURS[1]].mean()
    diff = (day - night).dropna()
    print(
        f"HR day{DAY_HOURS} minus night{NIGHT_HOURS}: median {diff.median():.1f} bpm; "
        f"day>night in {(diff > 0).sum()}/{len(diff)} participants"
    )
    print("cohort mean HR by clock hour:", hr_hour.mean(axis=1).round(0).astype(int).to_dict())


def check_meals(meals: pd.DataFrame) -> None:
    """Check 5: are macros present for every logged meal; meals per day."""
    print("\n[check 5: meals]")
    print("meal type values:", meals["Meal Type"].value_counts().to_dict())
    macro = [c for c in ("Calories", "Carbs", "Protein", "Fat", "Fiber") if c in meals.columns]
    print("macro NaN per field:", meals[macro].isna().sum().to_dict(), "of", len(meals))
    per_day = meals.groupby(["pid", meals["ts"].dt.date]).size()
    print(
        "meals/participant-day:",
        per_day.describe()[["mean", "min", "50%", "max"]].round(2).to_dict(),
    )
    if "Amount Consumed" in meals.columns:
        amt = pd.to_numeric(meals["Amount Consumed"], errors="coerce")
        print("Amount Consumed:", amt.describe()[["count", "min", "50%", "max"]].round(1).to_dict())


def check_units(z: zipfile.ZipFile) -> None:
    """Check 2: HbA1c units and lb/in conversions using bio.csv."""
    print("\n[check 2: units (bio.csv)]")
    name = next(n for n in z.namelist() if n.lower().endswith("bio.csv") and "__MACOSX" not in n)
    bio = pd.read_csv(io.BytesIO(z.read(name)))
    bio.columns = [c.strip() for c in bio.columns]
    print("bio shape", bio.shape, "| cols:", list(bio.columns))
    a1c_col = next(c for c in bio.columns if "A1c" in c)
    a1c = pd.to_numeric(bio[a1c_col], errors="coerce")
    print(f"{a1c_col!r}: min={a1c.min()} max={a1c.max()} n={a1c.notna().sum()}")
    groups = pd.cut(a1c, [0, 5.699, 6.4, 99], labels=["<5.7", "5.7-6.4", ">6.4"])
    print("groups using HbA1c as %:", groups.value_counts().sort_index().to_dict())
    w = pd.to_numeric(bio["Body weight"], errors="coerce")
    h = pd.to_numeric(bio["Height"], errors="coerce")
    print(f"weight lb {w.min()}-{w.max()} | height in {h.min()}-{h.max()}")
    bmi_calc = (w * LB_TO_KG) / (h * IN_TO_M) ** 2
    err = (bmi_calc - pd.to_numeric(bio["BMI"], errors="coerce")).abs()
    print(f"BMI from lb/in vs reported: median |diff| {err.median():.2f}, max {err.max():.2f}")


def main() -> None:
    """Run all five checks and print compact results."""
    z = zipfile.ZipFile(ZIP_PATH)
    names = [n for n in z.namelist() if "__MACOSX" not in n]
    print(
        "zip entries:",
        len(names),
        "| non-participant files:",
        [n for n in names if "CGMacros-0" not in n and not n.endswith("/")][:10],
    )
    files = participant_files(z)
    print(
        "participant CSVs:",
        len(files),
        "| ids 1..49 absent:",
        [f"{i:03d}" for i in range(1, 50) if f"{i:03d}" not in files],
    )
    recs, meal_parts, hr_parts = [], [], []
    for pid, name in files.items():
        df = pd.read_csv(io.BytesIO(z.read(name)), low_memory=False)
        if not recs:
            print("columns:", list(df.columns))
        rec, meals, hr = summarise_participant(pid, df)
        recs.append(rec)
        meal_parts.append(meals)
        hr_parts.append(hr)
    per = pd.DataFrame(recs).set_index("pid")
    print("\n[check 3: coverage per participant]")
    cols = ["days", "row_step_min", "libre_n", "dexcom_n", "hr_cov", "meals"]
    print(per[cols].describe().loc[["min", "50%", "max"]].round(2).to_string())
    for label, mask in (
        ("no Libre", per.libre_n == 0),
        ("no Dexcom", per.dexcom_n == 0),
        ("no HR", per.hr_n == 0),
        ("HR cov<50%", (per.hr_n > 0) & (per.hr_cov < 0.5)),
        ("no METs", per.mets_n == 0),
        ("no meals", per.meals == 0),
    ):
        print(f"{label}: {per.index[mask].tolist()}")
    print("dexcom step modes:", per["dexcom_step"].value_counts().to_dict())
    print("libre step modes:", per["libre_step"].value_counts().to_dict())
    print(
        "glucose range dexcom",
        per.dexcom_min.min(),
        per.dexcom_max.max(),
        "| libre",
        per.libre_min.min(),
        per.libre_max.max(),
    )
    print("dup timestamps:", int(per.dup_ts.sum()), "| backwards steps:", int(per.nonmono.sum()))
    print("\n[check 4: sleep] sleep-like columns:", sorted({c for s in per.sleep_cols for c in s}))
    meals = pd.concat(meal_parts, ignore_index=True)
    check_time_of_day(meals, pd.concat(hr_parts, axis=1))
    check_meals(meals)
    check_units(z)
    per.drop(columns=["sleep_cols"]).to_csv(ROOT / "data" / "interim" / "cgmacros_raw_coverage.csv")


if __name__ == "__main__":
    main()
