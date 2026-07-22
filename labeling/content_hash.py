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
    """sha256 of the first top-level `mdat` box payload, or None if there is none.

    Fails closed on a malformed box: a non-zero declared `size` smaller than
    the box's own header (< 8, or < 16 for a 64-bit extended-size box) cannot
    hold a real payload, so it returns None rather than seeking to a bogus
    (possibly backward) offset. A `struct.error` from a truncated/corrupt
    header (e.g. a 64-bit extended-size marker with no size bytes left to
    read) is likewise treated as "malformed" -> None, never propagated.
    """
    h = hashlib.sha256()
    try:
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
                if size != 0 and size < hdrlen:
                    # Malformed: declared size can't even cover its own header.
                    return None
                if typ == b"mdat":
                    # Found mdat: hash its payload
                    if size == 0:
                        # ISO-BMFF size==0 means "box extends to end of file"
                        while True:
                            chunk = f.read(_CHUNK)
                            if not chunk:
                                break
                            h.update(chunk)
                    else:
                        # Normal bounded box
                        payload = size - hdrlen
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
                payload = size - hdrlen
                f.seek(payload, 1)
    except struct.error:
        return None
