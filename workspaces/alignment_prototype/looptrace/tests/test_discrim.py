#!/usr/bin/env python3
"""Phase-4 tests: the discriminability mask zeroes identical (clone) repeat
bodies and keeps unique content at 1; instance re-selection switches a
segment to the repeat image with clearly better weighted support and holds
the incumbent on ties."""

from __future__ import annotations

import numpy as np
import pytest

from workspaces.alignment_prototype.looptrace import discrim, fixtures
from workspaces.alignment_prototype.looptrace.selfsim import SR


def test_mask_zero_on_clones_one_on_unique(tmp_path):
    import soundfile as sf

    song = fixtures.make_song(40.0, seed=5)
    song[25 * SR : 31 * SR] = song[10 * SR : 16 * SR]  # planted clone
    f = tmp_path / "ref.wav"
    sf.write(f, song, SR)
    pairs = [{"a0": 10.0, "a1": 16.0, "lag_s": 15.0, "class": "clone"}]
    mask, hz = discrim.discriminability_mask(str(f), pairs)
    inside = mask[int(11 * hz) : int(15 * hz)]
    outside = mask[int(2 * hz) : int(8 * hz)]
    assert float(inside.mean()) < 0.1, f"clone body should be ~0, got {inside.mean()}"
    assert float(outside.min()) == 1.0


def test_reselect_switches_to_supported_image(monkeypatch):
    # two repeat images 20 s apart; the decoded segment sits on the WRONG
    # one; discriminative points support the right one
    pairs = [{"a0": 10.0, "a1": 16.0, "lag_s": 20.0, "class": "distinct"}]
    hz = 10.0
    mask = np.full(600, 1.0, dtype=np.float32)
    monkeypatch.setattr(discrim, "discriminability_mask", lambda *_a, **_k: (mask, hz))
    # points collinear on the CORRECT diagonal (song = mix + 30 -> image 2)
    tm = np.linspace(0.0, 6.0, 40)
    pts = np.stack([tm, tm + 31.0], axis=1).astype(np.float32)
    segs = [(0.0, 11.0, 17.0)]  # decoded on image 1 (song 11..17)
    out = discrim.reselect_instances(segs, pts, 1.0, pairs, "unused", 6.0)
    assert out[0][1] == pytest.approx(31.0, abs=0.01)  # switched +20 s


def test_reselect_holds_incumbent_on_tie(monkeypatch):
    pairs = [{"a0": 10.0, "a1": 16.0, "lag_s": 20.0, "class": "distinct"}]
    mask = np.full(600, 1.0, dtype=np.float32)
    monkeypatch.setattr(
        discrim, "discriminability_mask", lambda *_a, **_k: (mask, 10.0)
    )
    tm = np.linspace(0.0, 6.0, 40)
    pts = np.concatenate(
        [np.stack([tm, tm + 11.0], axis=1), np.stack([tm, tm + 31.0], axis=1)]
    ).astype(np.float32)
    segs = [(0.0, 11.0, 17.0)]
    out = discrim.reselect_instances(segs, pts, 1.0, pairs, "unused", 6.0)
    assert out[0][1] == pytest.approx(11.0, abs=0.01)  # tie -> incumbent
