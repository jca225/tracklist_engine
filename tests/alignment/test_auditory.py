"""Synthetic-signal tests for the auditory-neuroscience probes + belief gates.

Pure DSP verified on constructed signals — no real audio, no model calls — so
the deterministic harness is provably correct before any GT calibration.
"""

from __future__ import annotations

import numpy as np

from alignment.agentic.auditory import (
    ENV_FPS,
    harmonic_salience,
    harmonic_sieve_match,
    modulation_ratio,
    modulation_spectrum,
    onset_align,
    onset_asynchrony,
    onset_envelope,
)
from alignment.agentic.belief import (
    Observation,
    masking_gate,
    surprise_prior,
)


def test_onset_envelope_peaks_at_energy_jump():
    # 40 frames, 8 freq bins; energy jumps up at frame 20
    spec = np.ones((8, 40)) * 0.1
    spec[:, 20:] = 1.0
    env = onset_envelope(spec)
    assert env.argmax() == 20
    assert 0.0 <= env.min() and env.max() == 1.0


def test_onset_align_recovers_known_lag():
    rng = np.random.default_rng(0)
    ref = np.abs(rng.standard_normal(200))
    lag_frames = 137
    mix = np.zeros(600)
    mix[lag_frames : lag_frames + 200] += ref  # embed ref at a known lag
    mix += 0.05 * np.abs(rng.standard_normal(600))  # light noise
    res = onset_align(mix, ref)
    assert res is not None
    assert abs(res.lag_s - lag_frames / ENV_FPS) < 2.0 / ENV_FPS
    assert res.peak > 0.7
    assert res.sharpness > 1.5  # a distinct alignment, not noise


def test_onset_asynchrony_low_when_synchronous_high_when_shifted():
    base = np.zeros(400)
    for p in (50, 150, 250, 350):
        base[p] = 1.0
    synchronous = onset_asynchrony(base, base.copy())
    shifted = base.copy()
    shift = int(0.1 * ENV_FPS)  # 100 ms >> 30 ms tolerance
    shifted = np.roll(shifted, shift)
    async_score = onset_asynchrony(base, shifted)
    assert synchronous < 0.1
    assert async_score > 0.8


def test_harmonic_salience_strong_on_comb_selfmatch_unity():
    # harmonic comb: energy at bins 4, 8, 12, ... (f0 at bin 4)
    spec = np.full((80, 30), 0.01)
    for h in range(1, 8):
        spec[4 * h] = 1.0
    sal = harmonic_salience(spec, sr=22050)
    noise = harmonic_salience(
        np.abs(np.random.default_rng(1).standard_normal((80, 30))), sr=22050
    )
    assert sal.mean() > noise.mean()
    assert abs(harmonic_sieve_match(sal, sal) - 1.0) < 1e-9


def test_modulation_spectrum_finds_beat_rate():
    t = np.arange(0, 8, 1.0 / ENV_FPS)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)  # 2 Hz modulation (120 BPM)
    rates, mag = modulation_spectrum(env)
    peak_rate = rates[mag.argmax()]
    assert abs(peak_rate - 2.0) < 0.3


def test_modulation_ratio_reflects_tempo_change():
    t = np.arange(0, 8, 1.0 / ENV_FPS)
    mix = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)  # mix sped up: 4 Hz
    ref = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)  # ref: 2 Hz
    ratio = modulation_ratio(mix, ref)
    assert ratio is not None
    assert abs(ratio - 0.5) < 0.1  # ref_rate / mix_rate = 2/4


def test_masking_gate_downweights_in_shadow_only():
    o = Observation("onset_align", set_start_s=100.0, confidence=0.8, precision=0.6)
    masked = masking_gate(o, ((90.0, 110.0),), floor=0.5)
    assert masked.precision == 0.3 and "masked" in masked.detail
    clear = masking_gate(o, ((200.0, 210.0),))
    assert clear.precision == 0.6  # outside every shadow, unchanged
    ab = masking_gate(Observation("x", None, 0.0, 0.6), ((90.0, 110.0),))
    assert ab.precision == 0.6  # abstentions never masked


def test_surprise_prior_emits_candidate_at_novelty_peak():
    curve = tuple((float(i), 0.1) for i in range(20))
    curve = curve[:10] + ((10.0, 5.0),) + curve[11:]  # sharp peak at t=10
    obs = surprise_prior(curve, min_z=1.0)
    assert any(o.set_start_s == 10.0 and o.probe == "surprise" for o in obs)
    # a flat curve emits nothing
    assert surprise_prior(tuple((float(i), 1.0) for i in range(20))) == ()
