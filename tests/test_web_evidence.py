"""The console's evidence payload: copied tables stay in step with the README, and empty stats are safe."""

from pathlib import Path

from src.web import evidence as ev

README = (Path(__file__).resolve().parents[1] / "README.md").read_text("utf-8")


def test_copied_rows_appear_in_readme():
    for t in [*ev.INDIA.values(), ev.SLEEP]:
        for row in t["rows"]:
            assert all(cell in README for cell in row[1:] if cell), row


def test_empty_stats_gives_not_available_not_numbers():
    d = ev.build({})
    assert d["tiers"]["rows"][1][2] == "not available"
    assert d["personalization"]["horizons"]["60"]["series"] == {}
    assert d["alarms"]["rows"] == []


def test_figures_exist():
    assert all((ev.FIG_SRC / Path(f["file"]).name).exists() for f in ev.build({})["figures"])
