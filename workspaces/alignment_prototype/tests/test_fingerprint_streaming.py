from __future__ import annotations

import numpy as np
import soundfile as sf

from workspaces.alignment_prototype.landmark_fp import (
    SR,
    fingerprint_from_audio,
    fingerprint_from_file_streaming,
)


def _synth(seconds: float = 40.0) -> np.ndarray:
    # A few stepping tones so constellation finds many landmarks across time.
    t = np.arange(int(seconds * SR)) / SR
    y = np.zeros_like(t)
    for i, f in enumerate([220.0, 440.0, 660.0, 880.0, 330.0, 550.0]):
        seg = (t >= i * seconds / 6) & (t < (i + 1) * seconds / 6)
        y[seg] += np.sin(2 * np.pi * f * t[seg])
    y += 0.01 * np.sin(2 * np.pi * 1000.0 * t)  # broadband-ish content
    return y.astype(np.float32)


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def test_streaming_matches_full_signal_fingerprint(tmp_path):
    y = _synth(40.0)
    wav = tmp_path / "m.wav"
    sf.write(
        wav, y, SR, subtype="FLOAT"
    )  # 32-bit lossless; 16-bit PCM clips & shifts STFT grids
    full = fingerprint_from_audio(y)
    # chunk_s=15 + overlap 3 → ~3 chunks over the 40s signal
    streamed = fingerprint_from_file_streaming(wav, chunk_s=15.0, overlap_s=3.0)
    # near-lossless: the interior of each chunk reproduces full-signal hashes; only
    # a few STFT-edge frames per boundary differ. Require high key overlap.
    j = _jaccard(set(full.hashes), set(streamed.hashes))
    assert j > 0.9, f"streaming keys Jaccard {j:.3f} too low"
    assert abs(streamed.duration_s - full.duration_s) < 0.5


def test_overlap_recovers_boundary_pairs(tmp_path):
    y = _synth(40.0)
    wav = tmp_path / "m.wav"
    sf.write(wav, y, SR, subtype="FLOAT")
    full_keys = set(fingerprint_from_audio(y).hashes)
    with_ovl = set(
        fingerprint_from_file_streaming(wav, chunk_s=15.0, overlap_s=3.0).hashes
    )
    no_ovl = set(
        fingerprint_from_file_streaming(wav, chunk_s=15.0, overlap_s=0.0).hashes
    )
    # overlap must recover at least as many true keys as no-overlap
    assert _jaccard(full_keys, with_ovl) >= _jaccard(full_keys, no_ovl)


def test_streaming_never_loads_whole_signal(monkeypatch, tmp_path):
    y = _synth(40.0)
    wav = tmp_path / "m.wav"
    sf.write(wav, y, SR, subtype="FLOAT")
    import librosa

    seen = []
    real_load = librosa.load

    def spy_load(*args, **kwargs):
        seen.append(kwargs.get("duration"))
        return real_load(*args, **kwargs)

    # `librosa` is imported function-locally inside landmark_fp, so landmark_fp has
    # no `librosa` attribute at module level.  Patch the module directly.
    monkeypatch.setattr(librosa, "load", spy_load)
    fingerprint_from_file_streaming(wav, chunk_s=10.0, overlap_s=3.0)
    # every load bounded a single chunk window; NONE loaded the whole 40s
    assert seen, "expected chunked loads"
    # hop-snapping rounds chunk_s+overlap_s up by at most 2 hops (FHOP/SR ≈ 0.023 s each)
    assert all(d is not None and d <= 10.0 + 3.0 + 2 * (512 / 22050) for d in seen)


def test_hop_alignment_is_load_bearing(tmp_path):
    """Regression: chunk STFT offsets MUST be integer multiples of FHOP=512.

    The broken implementation steps in seconds (``t0 += chunk_s``) and computes
    ``frame_off = round(t0 * FPS)``.  For any ``chunk_s`` whose sample count is
    not a multiple of FHOP — e.g. chunk_s=5.7 → 5.7*22050=125685 samples, and
    125685 % 512 = 245 ≠ 0 — every subsequent chunk's STFT frame grid is
    shifted by that residual relative to the full-signal grid.  This shifts
    every spectral peak's (time_frame, freq_bin) position, diverging the hash
    keys and collapsing the Jaccard against the full-signal reference (~0.16
    vs ~0.93 for the signal and chunk_s used here).

    The correct implementation snaps chunk_s to an integer hop count:
        step_hops = round(chunk_s * SR / FHOP)   # 245 for chunk_s=5.7
        step_samp = step_hops * FHOP              # 125440, a multiple of FHOP
    so ``off_samp // FHOP`` is always exact.

    This test proves the property end-to-end: run both the correct
    ``fingerprint_from_file_streaming`` and a local broken re-implementation
    against the same file, compare both to ``fingerprint_from_audio(y)``.
    Reverting to the broken step drops the Jaccard by > 0.4 (typical: ~0.76).
    """
    import librosa

    from workspaces.alignment_prototype.landmark_fp import (
        FHOP,
        FPS,
        constellation,
        hashes,
    )

    # Use the shared _synth signal — same pure-tone content as the other tests.
    # A 50 s span gives ≥8 chunks at chunk_s=5.7, enough to accumulate the
    # hop-grid drift across multiple boundaries.
    y = _synth(50.0)
    wav = tmp_path / "hop_align.wav"
    sf.write(wav, y, SR, subtype="FLOAT")  # 32-bit lossless, no quantisation noise

    # chunk_s=5.7 has a large residual: 5.7 * 22050 = 125685 samples,
    # 125685 % 512 = 245 samples off the hop grid.  The correct implementation
    # snaps to step_hops=245 → step_samp=125440 (a multiple of FHOP).
    # The broken implementation keeps t0 at 5.7, 11.4, … seconds — each one
    # 245, 490, … samples displaced from the nearest hop boundary, so the STFT
    # grid shifts by a different fractional hop every chunk.
    chunk_s = 5.7

    # ---- Reference: full-signal fingerprint (no chunking) -------------------
    full_keys: set[tuple[int, int, int]] = set(fingerprint_from_audio(y).hashes)

    # ---- Aligned: correct hop-snapped implementation ------------------------
    aligned_keys: set[tuple[int, int, int]] = set(
        fingerprint_from_file_streaming(wav, chunk_s=chunk_s, overlap_s=3.0).hashes
    )
    aligned_jaccard = _jaccard(full_keys, aligned_keys)

    # ---- Misaligned: broken seconds-based implementation --------------------
    # Re-implements the broken logic inline so the test is self-contained and
    # fails only when hop-alignment is absent — not when other behaviour changes.
    def _fingerprint_broken(path: str) -> set[tuple[int, int, int]]:
        """Broken streaming: steps in seconds, rounds frame_off — NOT hop-aligned."""
        win_s = chunk_s + 3.0  # overlap_s=3.0 (same as aligned call above)
        merged: dict[tuple[int, int, int], set[int]] = {}
        t0 = 0.0
        total_dur = 0.0
        while True:
            y_chunk, _ = librosa.load(path, sr=SR, mono=True, offset=t0, duration=win_s)
            if y_chunk.size == 0:
                break
            total_dur = max(total_dur, t0 + y_chunk.size / SR)
            tf, fb = constellation(y_chunk)
            if tf.size:
                frame_off = round(t0 * FPS)  # BROKEN: fractional hop not snapped
                for key, times in hashes(tf, fb).items():
                    bucket = merged.setdefault(key, set())
                    for tt in times:
                        bucket.add(int(tt) + frame_off)
            if y_chunk.size < round(win_s * SR):
                break
            t0 += chunk_s  # BROKEN: step in seconds, not hop-aligned samples
        return set({k: tuple(sorted(v)) for k, v in merged.items()})

    misaligned_keys = _fingerprint_broken(str(wav))
    misaligned_jaccard = _jaccard(full_keys, misaligned_keys)

    # Hop-alignment invariant: aligned must reproduce the full-signal hashes
    # well (>0.9); the broken version must diverge materially (<0.5).
    # Observed values for this signal + chunk_s: aligned≈0.926, broken≈0.161.
    assert aligned_jaccard > 0.9, (
        f"aligned Jaccard {aligned_jaccard:.3f} < 0.9 — hop-snapping may be broken"
    )
    assert misaligned_jaccard < 0.5, (
        f"broken Jaccard {misaligned_jaccard:.3f} ≥ 0.5 — "
        "hop-misalignment is not diverging peaks as expected"
    )
    # Large gap confirms the invariant is load-bearing, not just marginal noise.
    assert aligned_jaccard > misaligned_jaccard + 0.4, (
        f"gap too small: aligned={aligned_jaccard:.3f} broken={misaligned_jaccard:.3f}; "
        "hop-alignment must produce a large Jaccard improvement (expected >0.4)"
    )
