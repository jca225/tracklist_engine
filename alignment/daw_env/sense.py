"""Sense after listen — onset + live fp / HuBERT / lyrics Observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from alignment.agentic.actions import REGISTRY
from alignment.agentic.belief import Observation
from alignment.daw_env.session import DawSession, SpanGeom
from alignment.refine_ref_offsets import SR

# Content probes that can disagree with cue and drive nudges.
CONTENT_PROBES = frozenset({"fp", "stem_hubert", "lyrics"})


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
    """Cheap spectral-flux onset envelope."""
    if y.size < 1024:
        return np.zeros(1, dtype=np.float64)
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


def _onset_obs(session: DawSession, geom: SpanGeom, wav: Path | None) -> Observation:
    """Map peak onset inside the listen window to an absolute set_start proposal."""
    if wav is None or not wav.is_file():
        return Observation(
            probe="daw_onset",
            set_start_s=None,
            confidence=0.0,
            precision=0.55,
            cost=0.2,
            detail="no listen wav",
        )
    y = _load_wav(wav)
    env = onset_strength(y)
    peak = float(env.max()) if env.size else 0.0
    if peak < 0.05:
        return Observation(
            probe="daw_onset",
            set_start_s=None,
            confidence=0.0,
            precision=0.55,
            cost=0.2,
            ref_start_s=geom.ref_start_s,
            detail="flat onset envelope",
        )
    # Peak frame → time within listen window → absolute mix time.
    hop = 512
    peak_i = int(np.argmax(env))
    t0, t1 = session.last_listen_window or (geom.set_start_s, geom.set_end_s)
    win_dur = max(1e-3, t1 - t0)
    rel_s = (peak_i * hop / SR) if y.size else 0.0
    # Clamp to window; prefer onset near clip start (first 40% of window).
    propose = t0 + min(rel_s, win_dur)
    conf = float(np.clip(0.35 + 0.5 * peak, 0.0, 0.9))
    return Observation(
        probe="daw_onset",
        set_start_s=float(propose),
        confidence=conf,
        precision=0.55,
        cost=0.2,
        ref_start_s=geom.ref_start_s,
        detail=f"onset_peak={peak:.3f} propose={propose:.1f}s",
    )


def _live_sensor_obs(live_ctx: Any, geom: SpanGeom) -> list[Observation]:
    """fp / lyrics / stem_hubert via agentic LiveContext runners (abstain-safe)."""
    if live_ctx is None:
        return []
    from alignment.agentic.live_runners import (
        fp_runner,
        lyrics_runner,
        stem_hubert_runner,
    )

    data = {
        "slot_label": geom.slot,
        "recording_id": geom.recording_id,
        "claimed_stem": geom.claimed_stem,
        "set_start_s": geom.set_start_s,
        "set_end_s": geom.set_end_s,
        "cue_anchor_s": geom.cue_anchor_s,
    }
    out: list[Observation] = []
    loaded = getattr(live_ctx, "loaded", set()) or set()
    stem = geom.claimed_stem or "regular"
    for factory, probe, stems in (
        (fp_runner, "fp", frozenset({"instrumental"})),
        (lyrics_runner, "lyrics", frozenset({"acappella"})),
        (stem_hubert_runner, "stem_hubert", frozenset({"acappella"})),
    ):
        # Skip channels that never loaded (e.g. lyrics when AGENTIC_LIVE_SKIP_LYRICS)
        # or that cannot apply to this stem (avoids abstain spam in the event log).
        if loaded and probe not in loaded:
            continue
        if stem not in stems:
            continue
        try:
            obs = factory(live_ctx)(data)
            if obs is not None:
                out.append(obs)
        except Exception as e:  # noqa: BLE001
            prec = REGISTRY[probe].precision if probe in REGISTRY else 0.5
            out.append(
                Observation(
                    probe=probe,
                    set_start_s=None,
                    confidence=0.0,
                    precision=prec,
                    cost=0.0,
                    detail=f"sensor error: {type(e).__name__}",
                )
            )
    return out


def sense_listen(
    session: DawSession,
    geom: SpanGeom,
    *,
    listen_wav: Path | None = None,
    live_ctx: Any | None = None,
) -> list[Observation]:
    """Onset from listen WAV + optional LiveContext content sensors."""
    wav = listen_wav or session.last_listen_wav
    ctx = live_ctx if live_ctx is not None else getattr(session, "live_ctx", None)
    obs = [_onset_obs(session, geom, wav)]
    obs.extend(_live_sensor_obs(ctx, geom))
    return obs


def content_target_set_start(belief) -> float | None:
    """Weight-averaged set_start from non-abstaining content probes, if any."""
    live = [
        o
        for o in belief.observations
        if (not o.abstained) and o.probe in CONTENT_PROBES and o.set_start_s is not None
    ]
    if not live:
        return None
    w = sum(o.weight for o in live)
    if w <= 0:
        return None
    return float(sum(o.set_start_s * o.weight for o in live) / w)
