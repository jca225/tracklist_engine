"""Corpus-probe plumbing: run the seam's probes on pi-storage corpus stems.

The ACCEPT harness scores against the ``~/aligning/<set>/`` layout, which only
exists for the 2 GT sets. To harvest the ~1,011 downloaded corpus sets the same
probes must read the pi-storage layout (``/mnt/storage/stems/set/<aid>/`` for the
mix, the full mix at ``set_audio.path``). This tests that mix-audio resolution is
injectable and that ``corpus_mix_resolver`` maps stems to the pi-storage layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from workspaces.pws_aligner.cotrain_seam import (
    MixSpan,
    RefCandidate,
    corpus_mix_resolver,
    real_probe_scorer,
)

soundfile = pytest.importorskip("soundfile")
from workspaces.alignment_prototype.landmark_fp import SR  # noqa: E402


def _tone(dur_s, freqs, seed):
    t = np.arange(int(dur_s * SR)) / SR
    y = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    rng = np.random.default_rng(seed)
    for idx in np.cumsum(rng.integers(4000, 16000, size=40)):
        if idx < len(t):
            y[idx] += 3.0
    return y.astype(np.float32)


# ── corpus_mix_resolver: pi-storage layout routing (pure) ─────────────────────


def test_corpus_resolver_routes_stems_to_pistorage_layout(tmp_path):
    stem_dir = tmp_path / "stems" / "set" / "42"
    stem_dir.mkdir(parents=True)
    (stem_dir / "vocals.flac").write_bytes(b"x")
    (stem_dir / "instrumental.flac").write_bytes(b"x")
    full_mix = tmp_path / "objects" / "mix.m4a"
    full_mix.parent.mkdir(parents=True)
    full_mix.write_bytes(b"x")

    resolve = corpus_mix_resolver(full_mix, stem_dir)
    assert resolve("acappella") == stem_dir / "vocals.flac"
    assert resolve("instrumental") == stem_dir / "instrumental.flac"
    assert resolve("regular") == full_mix


def test_corpus_resolver_returns_none_when_absent(tmp_path):
    stem_dir = tmp_path / "stems" / "set" / "7"  # does not exist
    full_mix = tmp_path / "gone.m4a"  # does not exist
    resolve = corpus_mix_resolver(full_mix, stem_dir)
    assert resolve("instrumental") is None
    assert resolve("regular") is None


# ── real_probe_scorer honors an injected mix_resolver ─────────────────────────


def _corpus_fixture(tmp_path):
    ref = _tone(15.0, [430, 880, 1750], seed=1)
    lead = int(3.0 * SR)
    mix = np.zeros(lead + len(ref) + int(2.0 * SR), dtype=np.float32)
    mix[lead : lead + len(ref)] += ref
    stem_dir = tmp_path / "stems" / "set" / "99"
    stem_dir.mkdir(parents=True)
    full_mix = tmp_path / "mix.wav"
    soundfile.write(str(full_mix), mix, SR)
    ref_path = tmp_path / "ref.wav"
    soundfile.write(str(ref_path), ref, SR)
    return full_mix, stem_dir, ref_path


def test_scorer_consults_injected_resolver_with_routed_stem(tmp_path):
    full_mix, stem_dir, ref_path = _corpus_fixture(tmp_path)
    calls: list[str] = []

    base = corpus_mix_resolver(full_mix, stem_dir)

    def spy(stem):
        calls.append(stem)
        return base(stem)

    scorer = real_probe_scorer(mix_resolver=spy)
    span = MixSpan(set_id="corpus", slot_label="001", set_start_s=0.0, span_dur_s=20.0)
    cand = RefCandidate(
        recording_id="A", source_url="x", source_path=str(ref_path), stem="regular"
    )
    results = scorer(cand, span)
    # the scorer consulted the injected corpus resolver with the routed stem …
    assert calls == ["regular"]
    # … and ran the routed probes (fp+chroma) against the resolved mix.
    assert {r.source for r in results} == {"fp", "chroma"}


def test_scorer_abstains_when_resolver_returns_none(tmp_path):
    # a corpus set whose mix stems haven't been rendered yet → resolver returns
    # None → every routed probe abstains (the safe mix-absent no-op).
    _, _, ref_path = _corpus_fixture(tmp_path)
    scorer = real_probe_scorer(mix_resolver=lambda stem: None)
    span = MixSpan(set_id="corpus", slot_label="001", set_start_s=0.0, span_dur_s=20.0)
    cand = RefCandidate(
        recording_id="A", source_url="x", source_path=str(ref_path), stem="regular"
    )
    assert all(r.abstain for r in scorer(cand, span))
