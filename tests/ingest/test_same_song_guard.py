from __future__ import annotations

import numpy as np

from ingest.adapters.fingerprint import Fingerprint
from ingest.same_song_guard import GuardVerdict, same_song_guard


def _fp(seed: int, n: int = 400, dur: float = 200.0) -> Fingerprint:
    rng = np.random.default_rng(seed)
    return Fingerprint(
        duration_s=dur, raw=rng.integers(0, 2**32, size=n, dtype=np.uint32)
    )


def test_title_disjoint_refuses_on_title_channel():
    # the 20911 case: acquired song != target recording title
    v = same_song_guard("Come On Over Baby", "Good Time", "acappella", None, None)
    assert v.accept is False
    assert v.channel == "title"


def test_title_overlap_and_no_fp_accepts():
    v = same_song_guard(
        "Good Time (Studio Acapella)", "Good Time", "acappella", None, None
    )
    assert v.accept is True
    assert v.channel is None


def test_no_signal_accepts():
    # no acquired title AND no fingerprints -> cannot verify -> accept (not fail on absence)
    v = same_song_guard("", "Good Time", "acappella", None, None)
    assert v.accept is True


def test_content_wrong_song_refuses_when_title_passes():
    # title overlaps but instrumental content similarity is far too low -> WRONG_SONG
    a, b = _fp(1), _fp(999)  # unrelated fingerprints -> low similarity
    v = same_song_guard("Good Time", "Good Time Instrumental", "instrumental", a, b)
    assert v.accept is False
    assert v.channel == "content"


def test_duration_mismatch_refuses():
    long_ref = _fp(1, dur=200.0)
    short_cand = _fp(1, dur=20.0)  # ratio 0.1 -> DURATION_MISMATCH
    v = same_song_guard("Good Time", "Good Time", "acappella", long_ref, short_cand)
    assert v.accept is False
    assert v.channel == "content"


def test_fallback_to_original_accepts():
    # identical fingerprints, equal duration -> sim ~1.0, dur_ratio 1.0 ->
    # classify() returns FALLBACK_TO_ORIGINAL, which is NOT in _CONTENT_REFUSE.
    # Core invariant: a wrong-STEM-but-same-song signal must not refuse.
    fp = _fp(1)
    v = same_song_guard("", "", "instrumental", fp, fp)
    assert v.accept is True
    assert v.channel is None


def test_weak_signal_acappella_accepts():
    # different-seed fingerprints, equal duration -> classify() always returns
    # WEAK_SIGNAL for acappella once duration passes (chroma is weak by design
    # for vocals-only content), which is NOT in _CONTENT_REFUSE.
    fp_a, fp_b = _fp(1), _fp(2)
    v = same_song_guard("", "", "acappella", fp_a, fp_b)
    assert v.accept is True
