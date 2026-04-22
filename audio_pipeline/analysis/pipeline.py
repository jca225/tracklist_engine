"""Per-track analysis composition.

`analyze_track` runs: demucs → beat_this → cue-detr → loudness → MERT
per cue-delimited section, and returns a `TrackAnalysisResult` holding
everything the DB adapter needs to persist.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import AudioAsset
from ..result import Err, Ok, Result
from .adapters import audio_io, beat_this_adapter, cue_detr_adapter, demucs_adapter, loudness, mert_adapter
from .errors import AnalysisError
from .models import (
    BeatGrid,
    CuePoints,
    LoudnessReading,
    SectionEmbedding,
    TrackAnalysisResult,
)


@dataclass(frozen=True)
class Analyzers:
    """All model handles bundled so they load once per process."""
    demucs: demucs_adapter.DemucsHandle
    beats: beat_this_adapter.BeatThisHandle
    cues: cue_detr_adapter.CueDetrHandle
    mert: mert_adapter.MertHandle


def load_analyzers(device: str = "auto") -> Result[Analyzers, AnalysisError]:
    """Load every model once. Fails fast on the first load error."""
    d = demucs_adapter.load(device=device)
    if not d.is_ok():
        return d
    b = beat_this_adapter.load(device=device)
    if not b.is_ok():
        return b
    c = cue_detr_adapter.load()
    if not c.is_ok():
        return c
    m = mert_adapter.load(device=device)
    if not m.is_ok():
        return m
    return Ok(Analyzers(demucs=d.value, beats=b.value, cues=c.value, mert=m.value))


def _section_bounds(
    cue_times: tuple[float, ...], total_duration_s: float
) -> tuple[tuple[float, float], ...]:
    """Convert cue timestamps into contiguous (start, end) section intervals.

    If cue-detr returned no cues we return a single whole-track section so
    downstream MERT still runs once (useful for short promo mixes / intros).
    """
    if not cue_times:
        return ((0.0, total_duration_s),)
    cuts = sorted({0.0, *cue_times, total_duration_s})
    return tuple((cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i])


def _slice(samples: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    start = max(0, int(start_s * sr))
    end = min(samples.size, int(end_s * sr))
    return samples[start:end]


def analyze_track(
    a: Analyzers,
    asset: AudioAsset,
    stems_dir: Path,
) -> Result[TrackAnalysisResult, AnalysisError]:
    """Run the full per-track analysis, streaming resources linearly."""
    assert asset.track_audio_id is not None, "AudioAsset must be persisted before analysis"
    audio_path = Path(asset.path)

    # Demucs first (uses its own torchaudio loader).
    stems_r = demucs_adapter.separate(a.demucs, audio_path, stems_dir, asset.track_audio_id)
    if not stems_r.is_ok():
        return stems_r

    # beat_this reads the file directly too.
    beats_r = beat_this_adapter.predict(a.beats, audio_path)
    if not beats_r.is_ok():
        return beats_r
    beat_times, downbeat_times = beats_r.value
    bpm = beat_this_adapter.estimate_bpm(beat_times)
    measures = beat_this_adapter.measure_times(downbeat_times)

    # cue-detr on the same file.
    cues_r = cue_detr_adapter.predict(a.cues, audio_path)
    if not cues_r.is_ok():
        return cues_r
    cue_times = cues_r.value

    # Load audio as mono@24k once for loudness + MERT.
    wf_r = audio_io.load_mono(audio_path, target_sr=mert_adapter.MERT_SR)
    if not wf_r.is_ok():
        return wf_r
    wf = wf_r.value
    total_s = wf.samples.size / wf.sample_rate

    lufs_r = loudness.integrated_lufs(wf.samples, wf.sample_rate)
    if not lufs_r.is_ok():
        return lufs_r

    bounds = _section_bounds(cue_times, total_s)
    sections: list[SectionEmbedding] = []
    for idx, (s, e) in enumerate(bounds):
        chunk = _slice(wf.samples, wf.sample_rate, s, e)
        emb_r = mert_adapter.embed_section(
            a.mert, chunk, asset.track_audio_id, idx, s, e,
        )
        match emb_r:
            case Err(err) if err.kind == "empty_section":
                continue              # skip ultra-short cue gaps, keep others
            case Err(_):
                return emb_r
            case Ok(emb):
                sections.append(emb)

    versions = {
        "demucs": a.demucs.version,
        "beat_this": a.beats.version,
        "cue_detr": a.cues.checkpoint,
        "mert": a.mert.version,
    }
    return Ok(TrackAnalysisResult(
        track_audio_id=asset.track_audio_id,
        stems=stems_r.value,
        beats=BeatGrid(
            track_audio_id=asset.track_audio_id,
            beat_times=beat_times,
            downbeat_times=downbeat_times,
            measure_times=measures,
            bpm=bpm,
        ),
        cues=CuePoints(
            track_audio_id=asset.track_audio_id,
            cue_times=cue_times,
            model_version=a.cues.checkpoint,
        ),
        loudness=LoudnessReading(
            track_audio_id=asset.track_audio_id,
            integrated_lufs=lufs_r.value,
        ),
        sections=tuple(sections),
        analyzer_versions=versions,
    ))
