#!/usr/bin/env python3
"""Optional vocal restoration in front of the lyrics-ASR channel.

The lyrics channel's ceiling is *candidate coverage* — Whisper failing on
marginal, artifact-heavy acappella stems (Roformer/Demucs leave metallic
bleed and HF rolloff). A restoration pass (VoiceFixer) before transcription
can only help here, because the output feeds a *transcriber*, not a matcher:
if resynthesis "hallucinates" HF harmonics to make a gargly stem intelligible,
that's a coverage win and nothing fake ever reaches an identity/placement
feature. (Do NOT reuse this on chroma/HuBERT/fp channels — there hallucination
is a liability. See the "perceptual quality != alignment fidelity" note.)

Isolation: VoiceFixer pins old torch/librosa that would break the
Whisper/transformers stack in ``venvs/audio``. So it runs file->file as a
subprocess under its own venv — the same pattern this repo uses for Essentia
(``venvs/essentia`` invoked from ``venvs/audio``). If that venv is absent the
functions fall back to the raw audio with a warning; the channel never breaks.

Enhanced files are cached under ``.cache/enhanced_vocals/`` keyed by source
(path+mtime+size), so the expensive restore runs once per stem and the eval
path is a pure lookup.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CACHE = _HERE / ".cache" / "enhanced_vocals"
_ENHANCER = _HERE / "enhance_vocal.py"
# repo root is two levels up from alignment/
_DEFAULT_VENV_PY = _HERE.parents[1] / "venvs" / "voicefixer" / "bin" / "python"

_WARNED = False


def _enhancer_python() -> str | None:
    """Path to the isolated venv's python, or None if unavailable.

    Override with ``VOICEFIXER_PY`` (e.g. point at a resemble-enhance venv).
    """
    override = os.environ.get("VOICEFIXER_PY")
    if override and Path(override).is_file():
        return override
    return str(_DEFAULT_VENV_PY) if _DEFAULT_VENV_PY.is_file() else None


def enhanced_path(src: str | Path) -> Path:
    """Deterministic cache path for the restored version of ``src`` (pure)."""
    src = Path(src)
    st = src.stat()
    key = hashlib.md5(f"{src}:{st.st_mtime}:{st.st_size}".encode()).hexdigest()[:16]
    return _CACHE / f"{key}.wav"


def resolve_for_cache(src: str | Path, enabled: bool) -> str | Path:
    """Path the transcription cache should key on — no side effects.

    Used by the cache-only (eval) read path: returns the enhanced file only if
    it already exists, else the raw source (so a missing restore reads as a
    plain cache miss, never a subprocess run).
    """
    if not enabled:
        return src
    out = enhanced_path(src)
    return out if (out.is_file() and out.stat().st_size) else src


def maybe_enhance(src: str | Path, enabled: bool) -> str | Path:
    """Return a restored copy of ``src``, creating it if needed (may spawn).

    Falls back to the raw ``src`` (with a one-time warning) when disabled or
    when the isolated enhancer venv / subprocess is unavailable or fails.
    """
    global _WARNED
    if not enabled:
        return src
    out = enhanced_path(src)
    if out.is_file() and out.stat().st_size:
        return out

    py = _enhancer_python()
    if py is None:
        if not _WARNED:
            _WARNED = True
            warnings.warn(
                "vocal enhancement requested but no enhancer venv found "
                f"(looked at $VOICEFIXER_PY and {_DEFAULT_VENV_PY}); "
                "transcribing raw audio. See vocal_enhance.py header to set it up.",
                stacklevel=2,
            )
        return src

    _CACHE.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.wav")
    proc = subprocess.run(
        [py, str(_ENHANCER), "--in", str(src), "--out", str(tmp)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not tmp.is_file():
        warnings.warn(
            f"vocal enhancement failed for {src} (rc={proc.returncode}); "
            f"transcribing raw audio.\n{(proc.stderr or '')[-800:]}",
            stacklevel=2,
        )
        tmp.unlink(missing_ok=True)
        return src
    tmp.replace(out)
    print(f"enhanced {Path(src).name} -> {out.name}", file=sys.stderr)
    return out
