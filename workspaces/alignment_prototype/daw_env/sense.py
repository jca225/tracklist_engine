"""Sense after listen — emit agentic Observations from the rendered window."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from workspaces.alignment_prototype.agentic.belief import Observation
from workspaces.alignment_prototype.daw_env.session import DawSession, SpanGeom
from workspaces.alignment_prototype.refine_ref_offsets import SR


def _load_wav(path: Path) -> np.ndarray:
    try:
        import soundfile as sf

        y, sr = sf.read(str(path), always_2d=False)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        return y
    except Exception:
        import librosa

        y, _ = librosa.load(str(path), sr=SR, mono=True)
        return y.astype(np.float32)


def onset_strength(y: np.ndarray) -> np.ndarray:
    """Cheap spectral-flux onset envelope (no new sensor channel)."""
    if y.size < 1024:
        return np.zeros(1, dtype=np.float64)
    # Frame energy difference as a stand-in for auditory onset_align.
    hop = 512
    n = 1 + max(0, (y.size - 1024) // hop)
    e = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sl = y[i * hop : i * hop + 1024]
        e[i] = float(np.mean(sl * sl))
    d = np.diff(e, prepend=e[0])
    d = np.maximum(d, 0.0)
    if d.max() > 0:
        d = d / d.max()
    return d


def sense_listen(
    session: DawSession,
    geom: SpanGeom,
    *,
    listen_wav: Path | None = None,
) -> list[Observation]:
    """Return Observations from the last listen window for ``geom``.

    Primary v1 probe: ``daw_onset`` — confidence from onset energy in the
    rendered mix window near the clip's set_start. Precision floor 0.55
    (below solo auto-commit) until GT-calibrated.
    """
    wav = listen_wav or session.last_listen_wav
    obs: list[Observation] = []
    if wav is None or not wav.is_file():
        obs.append(
            Observation(
                probe="daw_onset",
                set_start_s=None,
                confidence=0.0,
                precision=0.55,
                cost=0.2,
                detail="no listen wav",
            )
        )
        return obs

    y = _load_wav(wav)
    env = onset_strength(y)
    peak = float(env.max()) if env.size else 0.0
    # If the window is silent / flat, abstain.
    if peak < 0.05:
        obs.append(
            Observation(
                probe="daw_onset",
                set_start_s=None,
                confidence=0.0,
                precision=0.55,
                cost=0.2,
                ref_start_s=geom.ref_start_s,
                detail="flat onset envelope",
            )
        )
        return obs

    # Propose the session geometry (agent already placed); confidence scales
    # with onset presence — listen confirms "something is happening here".
    conf = float(np.clip(0.4 + 0.5 * peak, 0.0, 0.95))
    obs.append(
        Observation(
            probe="daw_onset",
            set_start_s=geom.set_start_s,
            confidence=conf,
            precision=0.55,
            cost=0.2,
            ref_start_s=geom.ref_start_s,
            detail=f"onset_peak={peak:.3f}",
        )
    )
    return obs
