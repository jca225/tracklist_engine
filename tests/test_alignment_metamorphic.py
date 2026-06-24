"""Phase 3.5 (Tier-3): metamorphic properties of the alignment harness.

You can't prove an aligner "correct", but you CAN assert structural invariants on
synthetic audio with KNOWN ground truth, end-to-end through the contract:

  * recovery   — a mix built by embedding a reference at a known point decodes,
                 confidently (not abstaining), to that placement.
  * equivariance — shifting the embed point by Δ shifts the recovered offset by Δ
                 (convention-independent; the load-bearing metamorphic claim).

Driven through FingerprintProbe -> AlignmentResult, so it exercises the real DSP
and the contract together. Synthetic signal is a fixed tone "melody" (time-varying
spectrum) so landmarks localize sharply.
"""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("librosa")
pytest.importorskip("scipy")

from workspaces.alignment_prototype.harness import CandidatePool, MixContext, RefContext
from workspaces.alignment_prototype.harness.fingerprint_probe import FingerprintProbe

SR = 22050
_FREQS = [440, 660, 550, 880, 330, 770, 495, 620, 392, 587, 466, 698]  # fixed melody


def _note(f: float, dur: float) -> "np.ndarray":
    # Harmonics + a broadband onset click => many time-frequency landmarks and
    # sharp time localization (a bare sine yields too few peaks to fingerprint).
    t = np.arange(int(SR * dur)) / SR
    y = sum(np.sin(2 * np.pi * f * h * t) / h for h in (1, 2, 3, 4))
    click = np.zeros_like(y)
    click[:96] = np.linspace(1.0, 0.0, 96)  # onset transient (broadband)
    return (y + click).astype(np.float32)


def _ref_signal() -> "np.ndarray":
    rng = np.random.default_rng(0)  # deterministic; identical in ref and mix
    melody = np.concatenate([_note(f, 0.4) for f in _FREQS])  # ~4.8s, time-varying
    noise = (0.02 * rng.standard_normal(melody.shape[0])).astype(np.float32)
    return melody + noise


def _mix_with_ref_at(ref: "np.ndarray", pad_s: float) -> "np.ndarray":
    pad = np.zeros(int(SR * pad_s), dtype=np.float32)
    tail = np.zeros(SR, dtype=np.float32)
    return np.concatenate([pad, ref, tail])


def _probe_offset(pad_s: float):
    ref = _ref_signal()
    mix = _mix_with_ref_at(ref, pad_s)
    signals = {"mix": mix, "ref": ref}
    probe = FingerprintProbe(stretches=(1.0,), loader=lambda p: signals[str(p)])
    out = probe(
        MixContext(audio_path=Path("mix")),
        RefContext(recording_id="ref", audio_path=Path("ref")),
        CandidatePool(),
    )
    return out


def test_recovers_a_real_match_confidently():
    out = _probe_offset(1.0)
    assert not out.abstain, "a true embed must not abstain"
    assert out.confidence > 0.5
    assert out.recording_id == "ref"


def test_time_shift_equivariance():
    # Shifting the embed by Δ shifts the recovered offset by ~Δ (within a few
    # STFT hops). This is the metamorphic invariant, independent of sign/zero.
    o1 = _probe_offset(1.0)
    o2 = _probe_offset(2.0)
    assert not o1.abstain and not o2.abstain
    delta = abs(o2.offset_s - o1.offset_s)
    assert delta == pytest.approx(1.0, abs=0.1), (
        f"expected ~1.0s shift, got {delta:.3f}"
    )


def test_offset_is_linear_in_shift():
    # Strongest form: recovered offset tracks the embed delay with slope -1 (the
    # fingerprint lag convention). Linear recovery over the whole sweep, not just
    # a pairwise shift.
    for pad in (0.5, 1.0, 1.5, 2.0):
        out = _probe_offset(pad)
        assert not out.abstain
        assert out.offset_s == pytest.approx(-pad, abs=0.05)
