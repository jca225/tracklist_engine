"""Content-addressed audio identity primitives (stdlib only — runs on pi's bare python3).

Two keys bind a clip to a recording:
  * file_sha256  — full-file sha256, IDENTICAL to track_audio.sha256
                   (ingest/adapters/downloader.py::_sha256, 1 MiB chunks).
  * mdat_sha256  — sha256 of the mp4 top-level `mdat` box payload. iTunes tag
                   injection (tag_aligning_folder.py) rewrites moov/udta atoms but
                   never the `mdat` audio payload, so this is tag-invariant: a
                   locally-tagged master hashes to the same value as pi's canonical
                   file. Validated 2026-07-21 (Chainsmokers "Honest" mdat fe374e…
                   == pi 2vmxu50p).
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

_CHUNK = 1 << 20


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def mdat_sha256(path: str | Path) -> str | None:
    """sha256 of the first top-level `mdat` box payload, or None if there is none."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return None
            size = struct.unpack(">I", hdr[:4])[0]
            typ = hdr[4:8]
            if size == 1:  # 64-bit extended size
                size = struct.unpack(">Q", f.read(8))[0]
                hdrlen = 16
            else:
                hdrlen = 8
            payload = size - hdrlen
            if typ == b"mdat":
                remaining = payload
                while remaining > 0:
                    chunk = f.read(min(_CHUNK, remaining))
                    if not chunk:
                        break
                    h.update(chunk)
                    remaining -= len(chunk)
                return h.hexdigest()
            if size == 0:
                return None
            f.seek(payload, 1)
