"""End-to-end: real_probe_scorer fingerprints the mix ONCE per span.

Guards the wiring the unit tests don't: that ``real_probe_scorer`` builds its
probes from a shared ``MixFeatureCache`` so a span's positive + decoy candidates
reuse one full-mix fingerprint instead of recomputing it per candidate (the
~1 min/case perf fix). Uses tiny on-disk audio + a call-counting mix-fp compute.
"""

from pathlib import Path

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")
# The fp/chroma probes lazy-import librosa and ABSTAIN when it is missing, so
# without it this test collects and runs but counts zero fingerprints. Skip
# rather than assert on work that never happened (CI omits the audio stack).
pytest.importorskip("librosa")

from alignment.landmark_fp import SR
from pws_aligner.corpus import mix_feature_cache
from pws_aligner.corpus.cotrain_seam import MixSpan, RefCandidate, real_probe_scorer


def _tone(dur_s, freqs, seed):
    t = np.arange(int(dur_s * SR)) / SR
    y = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    rng = np.random.default_rng(seed)
    for idx in np.cumsum(rng.integers(4000, 16000, size=40)):
        if idx < len(t):
            y[idx] += 3.0
    return y.astype(np.float32)


def test_real_scorer_fingerprints_mix_once_per_span(tmp_path, monkeypatch):
    ref_a = _tone(15.0, [430, 880, 1750], seed=1)
    ref_b = _tone(15.0, [500, 1000, 2000], seed=2)
    lead = int(3.0 * SR)
    mix = np.zeros(lead + len(ref_a) + int(2.0 * SR), dtype=np.float32)
    mix[lead : lead + len(ref_a)] += ref_a

    aligning = tmp_path / "aligning"
    aligning.mkdir()
    soundfile.write(str(aligning / "mix.wav"), mix, SR)
    a_path = tmp_path / "ref_a.wav"
    b_path = tmp_path / "ref_b.wav"
    soundfile.write(str(a_path), ref_a, SR)
    soundfile.write(str(b_path), ref_b, SR)

    # count full-mix fingerprint computations across the whole scorer
    calls = {"n": 0}
    real_fp = mix_feature_cache._default_mix_fp

    def counting_mix_fp(mix_ctx):
        calls["n"] += 1
        return real_fp(mix_ctx)

    monkeypatch.setattr(mix_feature_cache, "_default_mix_fp", counting_mix_fp)

    scorer = real_probe_scorer(aligning)
    span = MixSpan(set_id="t", slot_label="001", set_start_s=0.0, span_dur_s=20.0)
    pos = RefCandidate(recording_id="A", source_url="x", source_path=str(a_path))
    decoy = RefCandidate(recording_id="B", source_url="y", source_path=str(b_path))

    res_pos = scorer(pos, span)
    res_decoy = scorer(decoy, span)

    assert calls["n"] == 1  # mix fingerprinted once, reused across both candidates
    # both candidates got an fp channel back through the cached scorer path
    assert any(r.source == "fp" for r in res_pos)
    assert any(r.source == "fp" for r in res_decoy)
