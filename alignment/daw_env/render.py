"""Offline listen render — crop mix/stem bus to WAV for sensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alignment.daw_env.session import DawSession
from alignment.refine_ref_offsets import SR

_REPO = Path(__file__).resolve().parents[2]
ALIGNING = Path.home() / "aligning"


def resolve_mix_path(
    set_id: str, bus: str, *, set_dir: Path | None = None
) -> Path | None:
    """Locate mix / stem bus audio for a set."""
    roots: list[Path] = []
    if set_dir is not None:
        roots.append(set_dir)
    # Common aligning folder layout: <set_id>__* or BB* folders.
    if ALIGNING.is_dir():
        for p in sorted(ALIGNING.glob(f"{set_id}*")):
            if p.is_dir():
                roots.append(p)
        # Prefer dirs that already have mix audio on disk.
        for p in sorted(ALIGNING.iterdir()):
            if p.is_dir() and any(
                (p / n).is_file() for n in ("mix.wav", "mix.flac", "mix.m4a")
            ):
                roots.append(p)
    names = {
        "mix": ("mix.wav", "mix.flac", "mix.m4a"),
        "mix_instrumental": (
            "mix_instrumental.wav",
            "mix_instrumental.flac",
            "mix_instrumental.m4a",
        ),
        "mix_vocals": ("mix_vocals.wav", "mix_vocals.flac", "mix_vocals.m4a"),
    }.get(bus, ())
    if not names:
        # slot bus: not resolved here — caller falls back to mix
        names = ("mix.wav", "mix.flac", "mix.m4a")
    for root in roots:
        for n in names:
            cand = root / n
            if cand.is_file():
                return cand
    return None


def _load_mono(path: Path, offset_s: float, duration_s: float) -> np.ndarray:
    import librosa

    y, _ = librosa.load(
        str(path),
        sr=SR,
        mono=True,
        offset=max(0.0, offset_s),
        duration=max(0.01, duration_s),
    )
    return y.astype(np.float32)


def _write_wav(path: Path, y: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(str(path), y, sr)
    except Exception:
        import wave

        pcm = np.clip(y, -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm_i16.tobytes())


def render_listen(
    session: DawSession,
    t0_s: float,
    t1_s: float,
    *,
    set_dir: Path | None = None,
    mix_path: Path | None = None,
) -> Path:
    """Crop the solo bus to ``out_dir/listen_{t0}_{t1}.wav``.

    If audio is missing, writes a short silence placeholder so the loop can still
    unit-test without a pulled set (sensors will abstain).
    """
    out_root = session.out_dir or (
        _REPO / "alignment/out/daw_env" / session.set_id
    )
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = float(min(t0_s, t1_s))
    t1 = float(max(t0_s, t1_s))
    dur = max(0.05, t1 - t0)
    stem = f"listen_{t0:.2f}_{t1:.2f}.wav"
    out = out_root / stem

    path = mix_path or resolve_mix_path(
        session.set_id, session.solo_bus, set_dir=set_dir
    )
    if path is None or not path.is_file():
        y = np.zeros(int(dur * SR), dtype=np.float32)
    else:
        try:
            y = _load_mono(path, t0, dur)
        except Exception:
            y = np.zeros(int(dur * SR), dtype=np.float32)
    _write_wav(out, y)
    return out
