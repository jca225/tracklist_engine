"""Round-trip laws: write locality, parse∘print=id, reparse stability.

Three fixture tiers:
- synthetic sessions (always run, machine-independent),
- committed real-Live fixtures in fixtures/als/ (auto-discovered — drop a new
  `.als` there and it's tested with zero code changes; see its README),
- Mac-local golden sessions (BB11/BB12/seed template; skip where absent).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labeling.als.cst import load_als_xml
from labeling.als.read import parse_layer_clips
from labeling.als.roundtrip import (
    check_locator_write,
    check_reparse_stable,
    check_tempo_write,
)
from labeling.als.validate import has_errors, validate_session
from tests.labeling.synth_session import session_als_file, session_root

# PROVENANCE MATTERS: root-level "<SET> align.als" files in ~/aligning are
# SEEDER OUTPUT (machine predictions rendered to .als) — never GT. The human
# hand sessions live in the Ableton "<name> Project/" folders. bb12_seeded is
# kept deliberately: seeder output is a real dialect the codec must parse.
GOLDEN = {
    "seed_template": (Path.home() / "aligning/_seed_template.als", None),
    "bb12_canonical": (
        Path.home()
        / "aligning/_backups/20260616_150150/big bootie 12 labeling Project/"
        / "big bootie 12 labeling_fast.als",
        301,
    ),
    "bb11_hand": (
        Path.home()
        / "aligning/2nvzlh2k__Two Friends - Big Bootie Mix Episode 11/"
        / "BB11 align Project/BB11 align.als",
        None,  # live session — clip count changes as John labels
    ),
}
# (the seeded-BB12 machine-dialect session is a committed fixture now:
# fixtures/als/bb12_seeded.als — auto-discovered, no golden entry needed)


def test_tempo_write_roundtrip_flat():
    assert check_tempo_write(session_root(), [(0.0, 120.0)]) == []


def test_tempo_write_roundtrip_ramps_steps_and_sentinels():
    curve = [
        (-8.0, 100.0),  # negative beat — writer clamps to 0
        (0.0, 100.0),
        (64.0, 128.0),  # linear ramp
        (64.0, 90.0),  # zero-width step (duplicate Time)
        (256.0, 90.0),
    ]
    assert check_tempo_write(session_root(), curve) == []


def test_locator_write_roundtrip():
    markers = [(16.0, "001 first song"), (-3.0, 42), (128.5, "002 & <escaped>")]
    assert check_locator_write(session_root(), markers) == []


def test_reparse_stable_synthetic(tmp_path):
    assert check_reparse_stable(session_als_file(tmp_path)) == []


def _session_laws(path: Path, name: str, expected_clips: int | None) -> None:
    assert check_reparse_stable(path) == []
    root = load_als_xml(path)
    diags = validate_session(root)
    assert not has_errors(diags), [d.render() for d in diags]
    if expected_clips is not None:
        n = len(parse_layer_clips(root))
        assert n == expected_clips, (
            f"{name} raw layer-clip count changed: {n} != {expected_clips} — "
            "session edited, or extraction behavior drifted"
        )
    # write laws must hold on real sessions, not just synthetic ones
    assert check_tempo_write(root, [(0.0, 120.0), (64.0, 126.0)]) == []
    assert check_locator_write(root, [(8.0, "test-marker")]) == []


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_session_laws(name):
    path, expected_clips = GOLDEN[name]
    if not path.exists():
        pytest.skip(f"golden session not on this machine: {path}")
    _session_laws(path, name, expected_clips)


# Committed real-Live fixtures (see fixtures/als/README.md for the authoring
# matrix). Auto-discovered: every *.als runs the full law set; an optional
# <name>.expect.json sidecar pins extraction facts.
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "als"
_FIXTURES = sorted(FIXTURE_DIR.glob("*.als"))


@pytest.mark.parametrize("path", _FIXTURES, ids=lambda p: p.stem)
def test_committed_fixture_laws(path):
    expect_path = path.with_suffix(".expect.json")
    expect = json.loads(expect_path.read_text()) if expect_path.exists() else {}
    _session_laws(path, path.stem, expect.get("layer_clips"))
    if "tempo_points" in expect:
        from labeling.als.semantics import parse_master_tempo

        assert len(parse_master_tempo(load_als_xml(path))) == expect["tempo_points"]


def test_fixture_dir_wired():
    # the discovery glob must point at a real directory even before any
    # fixture lands, so a typo'd path can't silently skip everything
    assert FIXTURE_DIR.is_dir()
