"""Chromaprint fingerprinting for variant identity sanity-checks.

Used when we acquire an acappella/instrumental variant: compare it against the
track's 'original' to catch the two failure modes seen during manual aligning —
a wrong song/edit, or a *silent fallback* where the search returned the regular
master instead of the variant.

The core tension (see analysis design notes): chromaprint matches the *same
recording*, but a real variant is deliberately a *different* recording, so
similarity is a graded signal that behaves differently per variant type:
  - instrumental vs original → shares chord progression + timing, so similarity
    lands in a MIDDLE band; too-high (~1.0) means the original was returned
    (fallback), too-low means wrong song.
  - acappella vs original → vocals-only chroma barely resembles the full song,
    so chromaprint is weak; lean on duration + a later vocal-stem check.

Thresholds below are first-pass and want empirical calibration on a handful of
known-good / known-bad variants before any hard gating. Today this is advisory.

Fingerprinting shells out to `fpcalc` (Chromaprint) with `-raw -json`; the
`fpcalc` binary and `pyacoustid` are already present in venvs/audio. Lives in
ingest/ because acquisition is its consumer; promote to an analysis/identity
module when `track_fingerprints` population is built.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

from core.result import Err, Ok, Result


@dataclass(frozen=True)
class FingerprintError:
    kind: str   # 'tool_missing' | 'fingerprint' | 'parse'
    detail: str


# Min overlapping frames before a slide offset is trusted (avoids a spuriously
# high score from a 2-frame overlap).
_MIN_OVERLAP = 16
# Slide window: variant and original start near-aligned (same song), so the best
# offset is near 0; ~256 frames ≈ ~32 s of intro slack is plenty.
_MAX_OFFSET = 256


@dataclass(frozen=True)
class Fingerprint:
    duration_s: float
    raw: np.ndarray   # uint32 sub-fingerprints, ~7.8 frames/sec


def fingerprint_file(path: str, *, timeout_s: float = 120.0) -> Result[Fingerprint, FingerprintError]:
    """Compute a raw Chromaprint fingerprint for `path` via fpcalc."""
    fpcalc = shutil.which("fpcalc")
    if fpcalc is None:
        return Err(FingerprintError(kind="tool_missing", detail="fpcalc not on PATH (install chromaprint)"))
    try:
        proc = subprocess.run(
            [fpcalc, "-raw", "-json", "-length", "0", path],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return Err(FingerprintError(kind="fingerprint", detail=str(e)))
    if proc.returncode != 0:
        return Err(FingerprintError(kind="fingerprint", detail=proc.stderr.strip()[:200]))
    try:
        data = json.loads(proc.stdout)
        raw = np.asarray(data["fingerprint"], dtype=np.uint32)
        return Ok(Fingerprint(duration_s=float(data["duration"]), raw=raw))
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return Err(FingerprintError(kind="parse", detail=f"parse fpcalc output: {e}"))


def similarity(a: np.ndarray, b: np.ndarray, *, max_offset: int = _MAX_OFFSET) -> float:
    """Best bit-agreement of two raw fingerprints over a bounded slide, in [0,1].

    For each offset, the mean per-frame bit-error-rate over the overlap is
    `popcount(a_i XOR b_i)/32`; similarity = 1 - min BER. Same recording ≈ 0.9+,
    unrelated ≈ 0.5 (random 32-bit words agree on ~half their bits).
    """
    a = np.asarray(a, dtype=np.uint32)
    b = np.asarray(b, dtype=np.uint32)
    if a.size == 0 or b.size == 0:
        return 0.0
    best = 0.0
    lo = -min(b.size - 1, max_offset)
    hi = min(a.size - 1, max_offset)
    for off in range(lo, hi + 1):
        a0 = max(0, off)
        a1 = min(a.size, b.size + off)
        if a1 - a0 < _MIN_OVERLAP:
            continue
        xor = a[a0:a1] ^ b[a0 - off:a1 - off]
        # popcount each uint32: view as bytes, unpack to bits, sum per word.
        bits = np.unpackbits(xor.view(np.uint8)).reshape(xor.size, 32).sum(axis=1)
        ber = float(bits.mean()) / 32.0
        if 1.0 - ber > best:
            best = 1.0 - ber
    return best


# Verdict codes (advisory). Calibrate thresholds before hard-gating.
_DUR_LO, _DUR_HI = 0.90, 1.15
_FALLBACK_SIM = 0.95          # >= this vs original ⇒ probably the original itself
_INSTR_MIN_SIM = 0.55         # instrumental should still share the harmony/timing


def classify(role: str, sim: float, dur_ratio: float) -> tuple[str, str]:
    """Return (verdict_code, human_detail) for an acquired variant.

    `role` is 'instrumental' | 'acappella'; `sim` is similarity-to-original;
    `dur_ratio` is variant_duration / original_duration.
    """
    if not (_DUR_LO <= dur_ratio <= _DUR_HI):
        return ("DURATION_MISMATCH",
                f"duration ratio {dur_ratio:.2f} outside [{_DUR_LO}, {_DUR_HI}] — likely a different edit or song")
    if sim >= _FALLBACK_SIM:
        return ("FALLBACK_TO_ORIGINAL",
                f"similarity {sim:.2f} ≈ identical to the original — the search likely returned the full master, not a {role}")
    if role == "instrumental":
        if sim >= _INSTR_MIN_SIM:
            return ("OK", f"similarity {sim:.2f}: shares the original's harmony/timing with vocals removed")
        return ("WRONG_SONG", f"similarity {sim:.2f} too low — probably a different song/arrangement")
    # acappella: chromaprint is weak (vocals-only chroma differs by design)
    return ("WEAK_SIGNAL",
            f"similarity {sim:.2f}: acappella chroma differs from the full song by design; duration ok — confirm by ear / vocal-stem check")
