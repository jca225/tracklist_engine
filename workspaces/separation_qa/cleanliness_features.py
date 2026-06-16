"""No-reference cleanliness features for a vocal/acappella or instrumental signal.

The "quality" of an acappella sourced online (or a stem we produced) is, for the
common no-reference case, really *cleanliness*: how free it is of separation
bleed, musical-noise artifacts, and band-limiting. These are physical properties
of the waveform — content-agnostic — which is why a ranker over them can
generalise to arbitrary acappellas it has never seen, despite tiny training n.

This module emits a small, interpretable feature vector. All features are
peak-normalised so they are gain-invariant (an online acap and a Demucs stem
differ in loudness; we do not want to reward the louder one).

Orientation (which direction means *cleaner*), per feature:
  floor_db            lower (more negative)  — deeper silence floor in vocal gaps
  dynamic_range_db    higher                 — bigger gap between signal and floor
  hf16k_ratio_db      higher                 — full-band (studio tell; MP3-sep is band-limited)
  rolloff95_hz        higher                 — same bandwidth tell
  gap_flatness        lower                  — true silence in gaps, not broadband musical noise
  hpss_perc_ratio     lower (for vocals)     — less drum/percussive bleed
  lowend_ratio_db     lower (for vocals)     — a real acappella has ~no <120 Hz (no kick/bass);
                                               a separated vocal leaks low-end instrumental bleed

NOTE on the silence floor: on the BB12 online-vs-separator pairs, floor_db and
gap_flatness *invert* — a separator hard-masks gaps to near-digital-silence,
while a real studio acappella keeps breaths/reverb/room tone. So "deep floor"
measures gating aggressiveness, not quality. The hand-set ORIENTATION below is a
prior only; the pairwise ranker re-learns the weights (and will down-weight /
flip floor) from the verified pairs. lowend_ratio_db is the separator-free bleed
proxy that should track contamination the right way regardless of gating.

The two heavy / external signals (DNSMOS, separator-bleed-residual) are exposed
behind explicit calls so the cheap path runs with only librosa+numpy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np

SR = 44100
_FRAME = 2048
_HOP = 1024
_GAP_DROP_DB = 35.0  # a frame >35 dB below the active level counts as a vocal gap
_HF_CUT_HZ = 16_000  # band above which studio acappellas keep energy, MP3-sep does not
_LOW_CUT_HZ = 120  # below this a clean vocal is ~silent; bleed leaks kick/bass here
_EPS = 1e-10


@dataclass(frozen=True)
class Cleanliness:
    duration_s: float
    gap_frac: float
    floor_db: float
    dynamic_range_db: float
    hf16k_ratio_db: float
    rolloff95_hz: float
    gap_flatness: float
    hpss_perc_ratio: float
    lowend_ratio_db: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


# Feature name -> +1 if higher is cleaner, -1 if lower is cleaner. Used by the
# pairwise eval to vote and (later) to fit a Bradley-Terry weight per feature.
ORIENTATION: dict[str, int] = {
    "floor_db": -1,
    "dynamic_range_db": +1,
    "hf16k_ratio_db": +1,
    "rolloff95_hz": +1,
    "gap_flatness": -1,
    "hpss_perc_ratio": -1,
    "lowend_ratio_db": -1,
}


def load_mono(path: str | Path, sr: int = SR) -> np.ndarray:
    """Load to mono float32, peak-normalised to ~1.0 (gain-invariant)."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > _EPS:
        y = y / peak
    return y.astype(np.float32)


def _frame_rms_db(y: np.ndarray) -> np.ndarray:
    rms = librosa.feature.rms(y=y, frame_length=_FRAME, hop_length=_HOP)[0]
    return 20.0 * np.log10(rms + _EPS)


def _gap_mask(rms_db: np.ndarray) -> np.ndarray:
    """Frames that are vocal-silent: >_GAP_DROP_DB below the active (p95) level."""
    if rms_db.size == 0:
        return np.zeros(0, dtype=bool)
    active = float(np.percentile(rms_db, 95))
    return rms_db < (active - _GAP_DROP_DB)


def _bandwidth(y: np.ndarray, sr: int) -> tuple[float, float, float]:
    S = np.abs(librosa.stft(y, n_fft=_FRAME, hop_length=_HOP)) ** 2
    power = S.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_FRAME)
    total = float(power.sum()) + _EPS
    hf = float(power[freqs >= _HF_CUT_HZ].sum())
    hf_ratio_db = 10.0 * np.log10((hf + _EPS) / total)
    low = float(power[freqs < _LOW_CUT_HZ].sum())
    lowend_ratio_db = 10.0 * np.log10((low + _EPS) / total)
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, roll_percent=0.95, n_fft=_FRAME, hop_length=_HOP
    )[0]
    return hf_ratio_db, float(np.median(rolloff)), lowend_ratio_db


def extract(path: str | Path, sr: int = SR) -> Cleanliness:
    """Compute the cheap (librosa-only) no-reference cleanliness vector."""
    y = load_mono(path, sr)
    dur = float(len(y) / sr)
    rms_db = _frame_rms_db(y)
    gap = _gap_mask(rms_db)
    gap_frac = float(gap.mean()) if gap.size else 0.0

    if gap.any():
        floor_db = float(np.median(rms_db[gap]))
    else:
        floor_db = float(np.min(rms_db)) if rms_db.size else 0.0
    p95 = float(np.percentile(rms_db, 95)) if rms_db.size else 0.0
    dynamic_range_db = p95 - floor_db

    hf_ratio_db, rolloff95, lowend_ratio_db = _bandwidth(y, sr)

    flat = librosa.feature.spectral_flatness(y=y, n_fft=_FRAME, hop_length=_HOP)[0]
    # flatness restricted to the gap frames = "is the silence truly silent, or
    # broadband musical noise?". Falls back to global if no gaps detected.
    gap_flatness = float(np.mean(flat[gap])) if gap.any() else float(np.mean(flat))

    harm, perc = librosa.effects.hpss(y)
    hp = float(np.sum(perc**2))
    hh = float(np.sum(harm**2))
    hpss_perc_ratio = hp / (hp + hh + _EPS)

    return Cleanliness(
        duration_s=dur,
        gap_frac=gap_frac,
        floor_db=floor_db,
        dynamic_range_db=dynamic_range_db,
        hf16k_ratio_db=hf_ratio_db,
        rolloff95_hz=rolloff95,
        gap_flatness=gap_flatness,
        hpss_perc_ratio=hpss_perc_ratio,
        lowend_ratio_db=lowend_ratio_db,
    )


if __name__ == "__main__":
    import json
    import sys

    for arg in sys.argv[1:]:
        print(arg)
        print(json.dumps(extract(arg).as_dict(), indent=2))
