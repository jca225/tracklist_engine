"""Chromaprint adapter: computes one fingerprint per audio file.

Wraps `fpcalc` (installed via `brew install chromaprint`) through the
`pyacoustid` Python binding. Raw fingerprint is persisted as a BLOB in
`track_fingerprints`; decoded, it's a sequence of int32 hashes at ~8 Hz.

Decoded representation chosen over pyacoustid's base64 string to keep
similarity math cheap at query time — XOR-popcount over two decoded
arrays is direct, whereas base64 needs a round-trip to get there.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import DbError
from ..result import Err, Ok, Result


@dataclass(frozen=True)
class FingerprintError:
    kind: str           # 'fpcalc_missing' | 'fpcalc_failed' | 'decode' | 'io'
    detail: str


@dataclass(frozen=True)
class Fingerprint:
    """One track's chromaprint fingerprint + its duration.

    `hashes` is the decoded int32 sequence — dtype uint32 so bitwise
    ops (XOR, popcount) stay zero-copy. The raw base64 bytes are also
    retained for round-trip storage in the DB BLOB column (cheaper
    than re-encoding from hashes at write time).
    """
    duration_s: float
    hashes: np.ndarray        # uint32, shape (n_hashes,)
    raw: bytes                # compressed form suitable for BLOB storage


def compute(
    audio_path: Path, *, maxlength_s: int = 7200,
) -> Result[Fingerprint, FingerprintError]:
    """Compute a chromaprint fingerprint for one audio file.

    Uses `pyacoustid.fingerprint_file`, which shells out to `fpcalc`.
    Accepts any format ffmpeg can decode (mp3/wav/flac/m4a/webm/etc).

    `maxlength_s` overrides pyacoustid's default 120-second cap.
    DJ sets are 30–120 minutes long; keeping the default would give
    you a 2-minute fingerprint for a 60-min mix and every scan would
    miss the last 58 minutes. 7200s = 2 hours, enough slack for any
    full mix we're likely to see.
    """
    try:
        import acoustid
    except ImportError as e:
        return Err(FingerprintError(kind="fpcalc_missing", detail=f"pyacoustid import: {e}"))

    try:
        duration, fp_b64 = acoustid.fingerprint_file(
            str(audio_path), maxlength=int(maxlength_s),
        )
    except (acoustid.FingerprintGenerationError, subprocess.SubprocessError) as e:
        return Err(FingerprintError(kind="fpcalc_failed", detail=str(e)))
    except FileNotFoundError as e:
        return Err(FingerprintError(kind="io", detail=str(e)))

    try:
        hashes = decode_hashes(fp_b64)
    except Exception as e:  # noqa: BLE001 — boundary catch, decode failures are fatal
        return Err(FingerprintError(kind="decode", detail=str(e)))

    return Ok(Fingerprint(
        duration_s=float(duration),
        hashes=hashes,
        raw=fp_b64 if isinstance(fp_b64, bytes) else str(fp_b64).encode("ascii"),
    ))


def decode_hashes(fp_b64: bytes | str) -> np.ndarray:
    """Decode chromaprint's base64-encoded fingerprint into a uint32 array.

    pyacoustid returns the standard chromaprint base64 encoding
    (URL-safe, no padding). We use chromaprint's own C decoder by
    proxy through pyacoustid's public helper if present; otherwise
    fall back to a manual decode matching the spec (header byte +
    sequence of little-endian int32s).
    """
    try:
        import chromaprint as _cp  # optional — pyacoustid doesn't expose a decoder
        raw = _cp.decode_fingerprint(fp_b64, base64=True)[0]
        return np.asarray(raw, dtype=np.uint32)
    except ImportError:
        pass

    # Manual path: pyacoustid uses urlsafe base64 without padding and
    # the decoded bytes are raw chromaprint bytes (1-byte header
    # `algorithm`, then 4-byte little-endian int hashes).
    import base64
    s = fp_b64.decode("ascii") if isinstance(fp_b64, bytes) else fp_b64
    pad = (-len(s)) % 4
    decoded = base64.urlsafe_b64decode(s + "=" * pad)
    if len(decoded) < 5:
        return np.zeros(0, dtype=np.uint32)
    header = decoded[0]
    body = decoded[1:]
    # The chromaprint format above version 1 uses a compressed layout;
    # we only handle the uncompressed variant here, sufficient for
    # fingerprints fpcalc emits by default. Unknown algorithm byte →
    # empty result, caller treats as "fingerprint present but
    # undecodable for direct compare" and still stores the raw form.
    if header not in (0, 1):
        return np.zeros(0, dtype=np.uint32)
    usable = (len(body) // 4) * 4
    return np.frombuffer(body[:usable], dtype="<u4").copy()


def similarity(a: np.ndarray, b: np.ndarray, max_hashes: int = 120) -> float:
    """Chromaprint similarity over the first `max_hashes` aligned hashes.

    Each hash is 32 bits; a "bit-error rate" compares them XOR-popcount
    style. Similarity = 1 − (bits differing) / (bits compared). 120
    hashes ≈ 15 s of audio, enough signal to distinguish tracks.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(a.size, b.size, max_hashes)
    xor = np.bitwise_xor(a[:n], b[:n])
    bits = np.unpackbits(xor.view(np.uint8)).sum()
    total_bits = n * 32
    return 1.0 - (bits / total_bits)
