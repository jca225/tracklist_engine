from __future__ import annotations

import json
from pathlib import Path

from sensors.lfs.canary_gold import load_canary_stem_gold
from sensors.lfs.claimed_stem import run_claimed_stem_lf, write_vertical_slice
from sensors.lfs.title_stem import SOURCE_NAME as TITLE_LF
from sensors.lfs.title_stem import guess_stem_from_title, run_title_stem_heuristic_lf

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "gold"


def test_claimed_stem_lf_emits_and_abstains() -> None:
    units, emissions = run_claimed_stem_lf(FIXTURES)
    assert len(units) == len(emissions)
    assert len(units) >= 1
    assert all(e.abstained or e.value is not None for e in emissions)
    assert any(not e.abstained for e in emissions)


def test_independent_canary_bundle(tmp_path: Path) -> None:
    out = write_vertical_slice(FIXTURES, tmp_path)
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert bundle["canary"]["source_name"] == TITLE_LF
    assert bundle["canaries"][0]["source_name"] == TITLE_LF
    assert bundle["canary"]["source_name"] != "claimed_stem_lf"
    gold = load_canary_stem_gold(FIXTURES)
    assert bundle["gold_by_unit"] == gold
    assert "Do NOT regenerate" in (FIXTURES / "canary_stem_gold.json").read_text()
    names = {e["source_name"] for e in bundle["emissions"]}
    assert "claimed_stem_lf" in names
    assert TITLE_LF in names


def test_title_heuristic_keywords() -> None:
    assert guess_stem_from_title("Official Instrumental") == "instrumental"
    assert guess_stem_from_title("Studio Acapella") == "acappella"
    assert guess_stem_from_title("THE FRAY · Over My Head") is None


def test_title_lf_vote_abstain_counts() -> None:
    _units, ems = run_title_stem_heuristic_lf(FIXTURES)
    assert sum(1 for e in ems if not e.abstained) == 4
    assert sum(1 for e in ems if e.abstained) == 2
