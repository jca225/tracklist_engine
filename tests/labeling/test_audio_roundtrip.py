"""Denotational audio round-trip — catches distant-clip GT merges."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from labeling.als.render_offline import RenderClip, sum_render_clips
from labeling.audio_roundtrip import assert_audio_equivalent


SR = 22050


def _write_tone(path: Path, *, freq: float, dur_s: float, sr: int = SR) -> None:
    t = np.arange(int(dur_s * sr), dtype=np.float64) / sr
    y = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), y, sr)


def test_identical_renders_pass(tmp_path: Path):
    wav = tmp_path / "a.wav"
    _write_tone(wav, freq=440.0, dur_s=2.0)
    clips = [
        RenderClip(wav, 0.0, 1.5, 0.0, 1.0),
        RenderClip(wav, 2.0, 3.0, 0.2, 1.0),
    ]
    a = sum_render_clips(clips, sr=SR)
    b = sum_render_clips(list(clips), sr=SR)
    r = assert_audio_equivalent(a, b, sr=SR)
    assert r.ok, r.detail


def test_distant_reprise_merge_fails_audio_law(tmp_path: Path):
    """BB12-099 class bug: two clips of the same file separated by ~12s.

    Arrangement plays A, silence, A again. A wrongly-merged GT plays A
    continuously across the gap — denotation must fail.
    """
    wav = tmp_path / "reprise.wav"
    _write_tone(wav, freq=330.0, dur_s=20.0)

    # Arrangement: 0–2s and 14–16s (12s gap).
    arrangement = [
        RenderClip(wav, 0.0, 2.0, 0.0, 1.0),
        RenderClip(wav, 14.0, 16.0, 0.0, 1.0),
    ]
    # Correct GT: same two spans.
    gt_ok = [
        RenderClip(wav, 0.0, 2.0, 0.0, 1.0),
        RenderClip(wav, 14.0, 16.0, 0.0, 1.0),
    ]
    # Bad merge: one clip 0–16s reading the file continuously.
    gt_merged = [
        RenderClip(wav, 0.0, 16.0, 0.0, 1.0),
    ]

    arr = sum_render_clips(arrangement, sr=SR)
    ok_buf = sum_render_clips(gt_ok, sr=SR)
    bad_buf = sum_render_clips(gt_merged, sr=SR)

    good = assert_audio_equivalent(arr, ok_buf, sr=SR)
    assert good.ok, good.detail

    bad = assert_audio_equivalent(arr, bad_buf, sr=SR)
    assert not bad.ok, f"merged reprise should fail audio law, got {bad.detail}"


def test_assert_rejects_empty_vs_signal():
    a = np.zeros(0, dtype=np.float32)
    b = np.ones(1000, dtype=np.float32)
    r = assert_audio_equivalent(a, b, sr=SR)
    assert not r.ok
