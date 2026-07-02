"""Round-trip laws: write locality, parse∘print=id, reparse stability.

Synthetic tests always run; golden tests pin real Mac-local sessions and skip
where those are absent (CI, other machines).
"""

from __future__ import annotations

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

GOLDEN = {
    "seed_template": (Path.home() / "aligning/_seed_template.als", None),
    "bb12": (
        Path.home()
        / "aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12/BB12 align.als",
        799,
    ),
    "bb11": (
        Path.home()
        / "aligning/2nvzlh2k__Two Friends - Big Bootie Mix Episode 11/BB11 align.als",
        149,
    ),
}


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


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_session_laws(name):
    path, expected_clips = GOLDEN[name]
    if not path.exists():
        pytest.skip(f"golden session not on this machine: {path}")
    assert check_reparse_stable(path) == []
    root = load_als_xml(path)
    assert not has_errors(validate_session(root))
    if expected_clips is not None:
        n = len(parse_layer_clips(root))
        assert n == expected_clips, (
            f"{name} raw layer-clip count changed: {n} != {expected_clips} — "
            "session edited, or extraction behavior drifted"
        )
    # write laws must hold on real sessions, not just synthetic ones
    assert check_tempo_write(root, [(0.0, 120.0), (64.0, 126.0)]) == []
    assert check_locator_write(root, [(8.0, "test-marker")]) == []
