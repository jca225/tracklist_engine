"""Stretch-tolerant landmark fingerprints for mix↔ref localization.

Constellation hashing (Shazam-style) used by fp_probe / refine_ref_offsets.
Serialized into ``track_fingerprints.fingerprint`` blobs (kind=landmark).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

SR = 22050
FHOP = 512
FPS = SR / FHOP
NFFT = 2048


@dataclass(frozen=True)
class LandmarkFingerprint:
    fps: float
    duration_s: float
    hashes: dict[tuple[int, int, int], tuple[int, ...]]

    def to_blob(self) -> bytes:
        entries = [[list(k), list(v)] for k, v in self.hashes.items()]
        payload = {
            "v": 1,
            "kind": "landmark",
            "fps": self.fps,
            "duration_s": self.duration_s,
            "entries": entries,
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_blob(cls, blob: bytes) -> LandmarkFingerprint:
        payload = json.loads(blob.decode("utf-8"))
        if payload.get("kind") != "landmark":
            raise ValueError(f"unsupported fingerprint kind: {payload.get('kind')!r}")
        entries = payload["entries"]
        hashes: dict[tuple[int, int, int], tuple[int, ...]] = {}
        for key, times in entries:
            hashes[(int(key[0]), int(key[1]), int(key[2]))] = tuple(
                int(t) for t in times
            )
        return cls(
            fps=float(payload["fps"]),
            duration_s=float(payload["duration_s"]),
            hashes=hashes,
        )


def constellation(y: np.ndarray, *, peak_size: int = 19, db_floor: float = 60.0):
    """(time_frames, freq_bins) of spectral-peak landmarks."""
    import librosa
    from scipy.ndimage import maximum_filter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=NFFT, hop_length=FHOP))
        )
    mx = maximum_filter(s, size=(peak_size, peak_size))
    pk = (s == mx) & (s > s.max() - db_floor)
    fb, tf = np.where(pk)
    return tf.astype(np.int32), fb.astype(np.int32)


def hashes(tf: np.ndarray, fb: np.ndarray, *, fan: int = 8, dt_max: int = 80) -> dict:
    """{(f1, f2, dt): (anchor_time_frames, ...)}."""
    order = np.argsort(tf)
    tf, fb = tf[order], fb[order]
    out: dict[tuple[int, int, int], list[int]] = {}
    n = len(tf)
    for i in range(n):
        for j in range(i + 1, min(i + 1 + fan, n)):
            dt = int(tf[j] - tf[i])
            if 1 <= dt <= dt_max:
                out.setdefault((int(fb[i]) // 2, int(fb[j]) // 2, dt), []).append(
                    int(tf[i])
                )
    return {k: tuple(v) for k, v in out.items()}


def fingerprint_from_audio(y: np.ndarray, *, sr: int = SR) -> LandmarkFingerprint:
    if sr != SR:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    tf, fb = constellation(y)
    return LandmarkFingerprint(
        fps=FPS,
        duration_s=float(len(y) / SR),
        hashes=hashes(tf, fb),
    )


def _vote_histogram(
    mix_hashes: dict[tuple[int, int, int], tuple[int, ...]],
    ref_hashes: dict[tuple[int, int, int], tuple[int, ...]],
) -> dict[int, int]:
    votes: dict[int, int] = {}
    for key, mts in mix_hashes.items():
        rts = ref_hashes.get(key)
        if not rts:
            continue
        for mt in mts:
            for rt in rts:
                off = rt - mt
                votes[off] = votes.get(off, 0) + 1
    return votes


def vote_sharpness(votes: dict[int, int]) -> float:
    """Peak / second-peak ratio; 0 when empty."""
    if not votes:
        return 0.0
    ranked = sorted(votes.values(), reverse=True)
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else 0
    return float(top / max(second, 1))


def fp_offset(
    mix_y: np.ndarray | None,
    ref_y: np.ndarray | None = None,
    *,
    ref_fp: LandmarkFingerprint | None = None,
    mix_fp: LandmarkFingerprint | None = None,
    stretches: tuple[float, ...] = (0.98, 1.0, 1.02),
) -> tuple[float, int, float, float]:
    """(ref_start_s, votes, stretch, sharpness).

    ``mix_fp`` is a precomputed full-mix fingerprint — symmetric with ``ref_fp``.
    When given, the mix constellation+hashes are reused instead of recomputed
    (``mix_y`` may then be None). Purely a speed knob: the result is identical to
    passing ``mix_y``. This is what lets the ACCEPT-precision harness fingerprint
    the mix once per span and reuse it across the positive + all decoys.
    """
    import librosa

    if ref_fp is None:
        if ref_y is None:
            raise ValueError("ref_y or ref_fp required")
        ref_fp = fingerprint_from_audio(ref_y)

    if mix_fp is not None:
        hm = mix_fp.hashes
    else:
        if mix_y is None:
            raise ValueError("mix_y or mix_fp required")
        tfm, fbm = constellation(mix_y)
        hm = hashes(tfm, fbm)
    best = (0.0, 0, 1.0, 0.0)
    for st in stretches:
        if ref_y is not None and abs(st - 1.0) > 1e-3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ry = librosa.effects.time_stretch(ref_y, rate=1.0 / st)
            hr = hashes(*constellation(ry))
        else:
            hr = ref_fp.hashes
        votes = _vote_histogram(hm, hr)
        if not votes:
            continue
        off, v = max(votes.items(), key=lambda kv: kv[1])
        sharp = vote_sharpness(votes)
        if v > best[1]:
            best = (off * FHOP / SR * st, v, st, sharp)
    return best


def fp_offset_resample(
    mix_y: np.ndarray,
    ref_y: np.ndarray,
    *,
    ratios: tuple[float, ...] = tuple(round(1.0 + 0.02 * k, 3) for k in range(-13, 17)),
) -> tuple[float, int, float, float]:
    """Resample-ratio fingerprint offset — for the RESAMPLE transform (pitch AND
    tempo coupled by one ratio). Unlike `fp_offset` (time-stretch, pitch-preserving),
    this resamples the ref by each ratio `r` with fast interpolation (not a
    phase-vocoder), so a pitch-shifted alignment diagonal re-registers.

    Returns (set_start_s, votes, ratio, sharpness). A mix track resampled to speed r
    has pitch+tempo x r; resampling the ref to target_sr = SR / r reproduces that
    speed-r playback when read back at SR.

    SIGN CONVENTION (differs from `fp_offset`): the vote offset is `rt - mt`
    (negative when the ref sits earlier than the mix), so this returns
    `-off * FHOP / SR * r` = the POSITIVE mix set_start directly. `fp_offset`
    returns the un-negated value and leaves the negation to its caller
    (`max(0, -off)`); here the negation is already applied, so a consumer uses
    `max(0.0, set_start_s)` WITHOUT re-negating.
    """
    import librosa

    hm = hashes(*constellation(mix_y))
    best = (0.0, 0, 1.0, 0.0)
    for r in ratios:
        if abs(r - 1.0) < 1e-3:
            ry = ref_y
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ry = librosa.resample(ref_y, orig_sr=SR, target_sr=int(round(SR / r)))
        votes = _vote_histogram(hm, hashes(*constellation(ry)))
        if not votes:
            continue
        off, v = max(votes.items(), key=lambda kv: kv[1])
        if v > best[1]:
            best = (-off * FHOP / SR * r, v, r, vote_sharpness(votes))
    return best
