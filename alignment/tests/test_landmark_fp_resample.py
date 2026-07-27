import numpy as np
import librosa
from alignment.landmark_fp import SR, fp_offset_resample


def _tone_complex(dur_s: float, freqs, sr: int = SR) -> np.ndarray:
    """A signal with stable spectral-peak landmarks: sustained tones gated by
    0.25 s onsets (gives both frequency and time structure for the constellation).

    Enriched vs. the original brief fixture: a sparse impulse train with
    irregular (prime-number-spaced) inter-event gaps is added so that the
    constellation has enough aperiodic time structure to produce a sharp
    vote peak rather than a flat histogram.  The three behavioral assertions
    (votes > 0, ratio within one grid step, offset within 1 s) are unchanged.
    """
    t = np.arange(int(dur_s * sr)) / sr
    y = np.zeros_like(t)
    for f in freqs:
        y += np.sin(2 * np.pi * f * t)
    env = ((t * 4.0) % 1.0 < 0.5).astype(np.float64)  # 4 Hz on/off gate
    y = y * env
    # Sparse impulse train with aperiodic spacing to break the repeating landmark symmetry
    rng = np.random.default_rng(42)
    impulse_times = np.cumsum(rng.integers(4000, 16000, size=50))
    impulse_times = impulse_times[impulse_times < len(t)]
    for idx in impulse_times:
        y[idx] += 3.0
    return y.astype(np.float32)


def test_fp_offset_resample_recovers_planted_ratio_and_offset():
    ref = _tone_complex(20.0, [400, 900, 1700, 3100, 5200])
    ratio = 1.10  # mix track is sped up 10%: pitch+tempo x1.10
    # simulate the resampled track as it appears in the mix
    mix_track = librosa.resample(ref, orig_sr=SR, target_sr=int(round(SR / ratio)))
    lead = int(3.0 * SR)  # planted 3 s into the mix
    mix = np.zeros(lead + len(mix_track) + int(2.0 * SR), dtype=np.float32)
    mix[lead : lead + len(mix_track)] += mix_track
    ratios = tuple(round(1.0 + 0.02 * k, 3) for k in range(-13, 17))  # ~0.74..1.32
    ref_start_s, votes, r_hat, sharp = fp_offset_resample(mix, ref, ratios=ratios)
    assert votes > 0
    assert abs(r_hat - ratio) <= 0.03  # within one grid step
    assert abs(ref_start_s - 3.0) < 1.0  # planted lead recovered
