"""MixFeatureCache: compute each mix/ref feature once, reuse across candidates.

The ACCEPT-precision harness scores a span's GT-correct ref plus N decoys against
the SAME mix window. The mix-side features (full-mix fingerprint, whole-mix chroma,
windowed mix chroma/HuBERT) depend only on the mix path (+ window) — not the
candidate — so they must be computed once and memoized. This is the fix for the
~1 min/case perf bug that blocked the rigorous per-axis run.

Tested with counting stub compute-fns (no audio deps): the contract is the
memoization keying, not the DSP (which the probe tests cover).
"""

from pathlib import Path

from workspaces.alignment_prototype.harness.contract import MixContext, RefContext
from workspaces.pws_aligner.mix_feature_cache import MixFeatureCache


class _Counter:
    def __init__(self, tag):
        self.tag = tag
        self.calls = []

    def __call__(self, ctx):
        self.calls.append(ctx)
        return f"{self.tag}:{id(ctx)}"

    @property
    def n(self):
        return len(self.calls)


def _mix(path="/m/mix.flac", start=10.0, end=50.0):
    return MixContext(audio_path=Path(path), span_start_s=start, span_end_s=end)


def _ref(path="/r/ref.flac", rid="r1"):
    return RefContext(recording_id=rid, audio_path=Path(path))


def _cache():
    return MixFeatureCache(
        compute_mix_fp=_Counter("fp"),
        compute_mix_chroma_whole=_Counter("cw"),
        compute_mix_chroma_window=_Counter("cwin"),
        compute_mix_hubert_window=_Counter("hwin"),
        compute_ref_chroma=_Counter("rc"),
        compute_ref_hubert=_Counter("rh"),
    )


def test_mix_fp_keyed_by_path_only():
    c = _cache()
    a, b = c.mix_fp(_mix(start=0.0)), c.mix_fp(_mix(start=999.0))
    assert a == b  # span ignored — full-mix fp is span-independent
    assert c._compute_mix_fp.n == 1
    c.mix_fp(_mix(path="/m/other.flac"))
    assert c._compute_mix_fp.n == 2  # different path recomputes


def test_mix_chroma_whole_keyed_by_path_only():
    c = _cache()
    c.mix_chroma_whole(_mix(start=0.0))
    c.mix_chroma_whole(_mix(start=123.0))
    assert c._compute_mix_chroma_whole.n == 1


def test_mix_chroma_window_keyed_by_path_and_window():
    c = _cache()
    c.mix_chroma_window(_mix(start=10.0, end=50.0))
    c.mix_chroma_window(_mix(start=10.0, end=50.0))  # same window -> cached
    assert c._compute_mix_chroma_window.n == 1
    c.mix_chroma_window(_mix(start=60.0, end=90.0))  # new window -> recompute
    assert c._compute_mix_chroma_window.n == 2


def test_mix_hubert_window_keyed_by_path_and_window():
    c = _cache()
    c.mix_hubert_window(_mix(start=10.0, end=50.0))
    c.mix_hubert_window(_mix(start=10.0, end=50.0))
    assert c._compute_mix_hubert_window.n == 1


def test_ref_features_keyed_by_ref_path():
    c = _cache()
    c.ref_chroma(_ref(path="/r/a.flac"))
    c.ref_chroma(_ref(path="/r/a.flac", rid="different-rid"))  # path is the key
    assert c._compute_ref_chroma.n == 1
    c.ref_chroma(_ref(path="/r/b.flac"))
    assert c._compute_ref_chroma.n == 2
    c.ref_hubert(_ref(path="/r/a.flac"))
    assert c._compute_ref_hubert.n == 1


def test_returned_value_is_stable_across_calls():
    c = _cache()
    first = c.mix_fp(_mix())
    second = c.mix_fp(_mix(start=1.0))
    assert first == second  # same cached object returned, not recomputed
