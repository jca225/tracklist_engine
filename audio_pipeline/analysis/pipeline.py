"""Per-track analysis composition.

`analyze_track` runs: demucs → beat_this → cue-detr → loudness → MERT
per cue-delimited section, and returns a `TrackAnalysisResult` holding
everything the DB adapter needs to persist.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import logging

from .adapters import (
    audio_io, beat_this_adapter, cue_detr_adapter, demucs_adapter,
    essentia_adapter, loudness,
)

from core.models import AudioAsset
from core.result import Err, Ok, Result
from .adapters import mert_adapter
from .errors import AnalysisError
from .models import (
    BeatGrid,
    CuePoints,
    EssentiaFeatures,
    LoudnessReading,
    MeasureEmbedding,
    TrackAnalysisResult,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Analyzers:
    """All model handles bundled so they load once per process.

    `with_essentia` flips automatically based on whether the
    venvs/essentia/ sandbox exists on disk — Essentia is best-effort,
    not required.
    """
    demucs: demucs_adapter.DemucsHandle
    beats: beat_this_adapter.BeatThisHandle
    cues: cue_detr_adapter.CueDetrHandle
    mert: mert_adapter.MertHandle
    with_essentia: bool = False


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
    return Ok(Analyzers(
        demucs=d.value, beats=b.value, cues=c.value, mert=m.value,
        with_essentia=essentia_adapter.is_available(),
    ))


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
    """Run the full per-track analysis, streaming resources linearly.

    All analysis (BPM, beats, downbeats, measure grid, cue points, MERT,
    Essentia, loudness) runs against the **original full-mix audio** at
    asset.path — never the demucs-separated vocal/instrumental stems.
    Stems are produced as a side output for downstream alignment use only.
    """
    assert asset.track_audio_id is not None, "AudioAsset must be persisted before analysis"
    audio_path = Path(asset.path)   # original full audio — single source of truth for all analysis

    # Demucs writes stems to disk; its output is not fed back into the analyzers below.
    stems_r = demucs_adapter.separate(a.demucs, audio_path, stems_dir, asset.track_audio_id)
    if not stems_r.is_ok():
        return stems_r

    # beat_this on the original audio.
    beats_r = beat_this_adapter.predict(a.beats, audio_path)
    if not beats_r.is_ok():
        return beats_r
    beat_times, downbeat_times = beats_r.value
    bpm = beat_this_adapter.estimate_bpm(beat_times)
    measure_times = beat_this_adapter.measure_times(downbeat_times)

    # cue-detr on the original audio.
    cues_r = cue_detr_adapter.predict(a.cues, audio_path)
    if not cues_r.is_ok():
        return cues_r
    cue_times = cues_r.value

    # Load audio as mono@24k once for loudness + MERT.
    wf_r = audio_io.load_mono(audio_path, target_sr=mert_adapter.MERT_SR)
    if not wf_r.is_ok():
        return wf_r
    wf = wf_r.value

    lufs_r = loudness.integrated_lufs(wf.samples, wf.sample_rate)
    if not lufs_r.is_ok():
        return lufs_r

    # Per-measure MERT: one embedding per beat_this-derived measure.
    # Single forward pass over the full track; mean-pool frames per measure.
    # The BPE cue-point optimizer (Phase 8b) re-aggregates these into
    # post-BPE section embeddings without rerunning MERT.
    measures_r = mert_adapter.embed_track_per_measure(
        a.mert, wf.samples, asset.track_audio_id, measure_times,
    )
    if not measures_r.is_ok():
        return measures_r
    measure_embeddings = measures_r.value

    versions = {
        "demucs": a.demucs.version,
        "beat_this": a.beats.version,
        "cue_detr": a.cues.checkpoint,
        "mert": a.mert.version,
    }

    # Essentia: best-effort enrichment. The sandbox lives in venvs/essentia/
    # (Py 3.13) and is invoked via subprocess. If the venv is missing or the
    # worker fails, we log and continue — stems/beats/MERT are the contract,
    # Essentia features are a bonus layer.
    essentia_features: EssentiaFeatures | None = None
    if a.with_essentia:
        ess_r = essentia_adapter.analyze(audio_path, asset.track_audio_id)
        match ess_r:
            case Ok(feat):
                essentia_features = feat
                versions["essentia"] = feat.version
            case Err(err):
                _log.warning(
                    "essentia worker failed for track_audio_id=%s: %s — %s",
                    asset.track_audio_id, err.kind, err.detail,
                )

    return Ok(TrackAnalysisResult(
        track_audio_id=asset.track_audio_id,
        stems=stems_r.value,
        beats=BeatGrid(
            track_audio_id=asset.track_audio_id,
            beat_times=beat_times,
            downbeat_times=downbeat_times,
            measure_times=measure_times,
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
        measures=measure_embeddings,
        analyzer_versions=versions,
        essentia=essentia_features,
    ))
