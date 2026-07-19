"""FingerprintProbe's mix_fp injection: reuse a precomputed full-mix fingerprint.

The probe already injects its ``loader``; the mix constellation+hashes, however,
live inside ``fp_offset`` and were recomputed per candidate. A ``mix_fp`` provider
lets the ACCEPT harness fingerprint the mix ONCE per span and reuse it across the
positive + all decoys — the full-mix load must not happen when the provider is set,
and the result must equal the un-cached probe on identical audio.
"""

from pathlib import Path

import numpy as np

from workspaces.alignment_prototype.harness.contract import (
    CandidatePool,
    MixContext,
    RefContext,
)
from workspaces.alignment_prototype.harness.fingerprint_probe import FingerprintProbe
from workspaces.alignment_prototype.landmark_fp import SR, fingerprint_from_audio

MIX_PATH = Path("/fake/mix.flac")
REF_PATH = Path("/fake/ref.flac")


def _tone_complex(dur_s: float, freqs) -> np.ndarray:
    t = np.arange(int(dur_s * SR)) / SR
    y = np.zeros_like(t)
    for f in freqs:
        y += np.sin(2 * np.pi * f * t)
    env = ((t * 4.0) % 1.0 < 0.5).astype(np.float64)
    y = y * env
    rng = np.random.default_rng(7)
    for idx in np.cumsum(rng.integers(4000, 16000, size=50)):
        if idx < len(t):
            y[idx] += 3.0
    return y.astype(np.float32)


def _fixtures():
    ref = _tone_complex(20.0, [430, 880, 1750, 3200])
    lead = int(3.0 * SR)
    mix = np.zeros(lead + len(ref) + int(2.0 * SR), dtype=np.float32)
    mix[lead : lead + len(ref)] += ref
    return mix, ref


def _ctxs():
    mix_ctx = MixContext(audio_path=MIX_PATH, span_start_s=0.0, span_end_s=25.0)
    ref_ctx = RefContext(recording_id="r1", audio_path=REF_PATH)
    return mix_ctx, ref_ctx, CandidatePool()


def test_mix_fp_provider_skips_mix_load_and_matches_uncached():
    mix, ref = _fixtures()
    audio = {MIX_PATH: mix, REF_PATH: ref}

    load_calls: list[Path] = []

    def counting_loader(path: Path):
        load_calls.append(path)
        return audio[path]

    mix_ctx, ref_ctx, pool = _ctxs()

    uncached = FingerprintProbe(loader=counting_loader).run(mix_ctx, ref_ctx, pool)

    load_calls.clear()
    mfp = fingerprint_from_audio(mix)
    cached_probe = FingerprintProbe(loader=counting_loader, mix_fp=lambda m: mfp)
    cached = cached_probe.run(mix_ctx, ref_ctx, pool)
    # ran the probe twice-worth of candidates; the mix must never be loaded
    cached_probe.run(mix_ctx, ref_ctx, pool)

    assert MIX_PATH not in load_calls  # provider supplied the mix fingerprint
    assert load_calls.count(REF_PATH) == 2  # ref still loaded per candidate
    assert cached.offset_s == uncached.offset_s
    assert cached.confidence == uncached.confidence
    assert cached.recording_id == uncached.recording_id
