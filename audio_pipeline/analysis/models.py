"""Frozen records produced by the analysis adapters.

Each adapter owns one of these; the pipeline composition in `pipeline.py`
assembles them into `TrackAnalysisResult` which is what gets persisted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


STEM_NAMES: Final[tuple[str, ...]] = ("vocals", "drums", "bass", "other")

# Derived stems — not produced by demucs directly, computed as sums of
# demucs outputs. `instrumental` = drums + bass + other is what a DJ
# plays when the text says "(Instrumental)"; pre-summing once at
# ingestion turns the 3-stem instrumental hypothesis into a single-file
# alignment target downstream. Kept separate from STEM_NAMES so the
# demucs loop doesn't try to find it in the model output.
DERIVED_STEM_NAMES: Final[tuple[str, ...]] = ("instrumental",)


@dataclass(frozen=True)
class AudioSignal:
    """Mono PCM ready for analysis. Owned by whoever loaded it."""
    samples_f32_path: str          # on-disk mono .wav (not kept in memory between adapters)
    sample_rate: int
    duration_s: float


@dataclass(frozen=True)
class StemAsset:
    """One Demucs-separated stem on disk."""
    track_audio_id: int
    stem_name: str                 # 'vocals' | 'drums' | 'bass' | 'other'
    path: str
    codec: str                     # 'wav' | 'flac' | 'mp3'


@dataclass(frozen=True)
class StemSet:
    track_audio_id: int
    stems: tuple[StemAsset, ...]


@dataclass(frozen=True)
class BeatGrid:
    """Output of beat_this: timestamps in seconds."""
    track_audio_id: int
    beat_times: tuple[float, ...]
    downbeat_times: tuple[float, ...]
    measure_times: tuple[float, ...]   # derived: every N downbeats per time-sig
    bpm: float


@dataclass(frozen=True)
class CuePoints:
    """cue-detr output: EDM cue-point timestamps in seconds."""
    track_audio_id: int
    cue_times: tuple[float, ...]
    model_version: str


@dataclass(frozen=True)
class LoudnessReading:
    track_audio_id: int
    integrated_lufs: float


@dataclass(frozen=True)
class SectionEmbedding:
    """MERT embedding for one cue-delimited section of a track.

    `embedding` is serialized as raw bytes at persistence time to keep this
    dataclass hashable; shape is (n_frames, dim).
    """
    track_audio_id: int
    section_idx: int
    start_s: float
    end_s: float
    n_frames: int
    dim: int
    dtype: str                     # 'float16'
    embedding_bytes: bytes


@dataclass(frozen=True)
class EssentiaFeatures:
    """Features extracted by the Essentia subprocess worker.

    Sandbox lives in venvs/essentia/ (Py 3.13) because Essentia has no Py 3.14
    wheels. Two source layers:
      1. Signal-processing algorithms — always populated (key, bpm, danceability_sp).
      2. TF classifier heads — populated when their .pb model file is present
         under data/essentia_models/. Otherwise None.

    All `*_prob` fields are in [0, 1]. `valence` / `arousal` are rescaled
    from emoMusic's 1..9 scale into [0, 1]. `valence_raw` / `arousal_raw`
    keep the original scale for traceability.
    """
    track_audio_id: int
    version: str                   # 'essentia_v2'
    models_present: tuple[str, ...]
    # Signal processing — always populated.
    key_tonic: str                 # 'C', 'C#', 'D', ...
    key_mode: str                  # 'major' | 'minor'
    key_strength: float            # 0..1, key-detection confidence
    key_profile: str               # 'edma' (purpose-built for EDM)
    bpm: float                     # cross-check vs beat_this; differ on complex EDM
    n_beats: int
    danceability_sp: float         # Essentia signal-processing Danceability (DFA)
    # TF heads — None when the model file is missing.
    mood_happy: float | None       # P(happy) — proxy for valence
    mood_acoustic: float | None    # P(acoustic)  → acousticness
    mood_aggressive: float | None  # P(aggressive) → energy proxy
    voice_prob: float | None       # P(voice) → instrumentalness = 1 - voice_prob
    danceability_tf: float | None  # P(danceable)
    valence: float | None          # rescaled to [0, 1]
    arousal: float | None          # rescaled to [0, 1]
    valence_raw: float | None      # raw emoMusic 1..9
    arousal_raw: float | None      # raw emoMusic 1..9
    # YAMNet-derived. None when YAMNet model is absent.
    speechiness: float | None      # max P(speech) | P(conversation), excludes singing
    liveness: float | None         # max-over-frames P(applause | cheering | crowd)
    yamnet_raw: dict[str, float] | None   # full YAMNet aggregations (debugging)


@dataclass(frozen=True)
class TrackAnalysisResult:
    """Everything produced for a single downloaded track.

    `essentia` is None when the venvs/essentia/ sandbox isn't installed
    on this machine, or when the worker subprocess failed (logged but
    non-fatal — stems / beats / MERT still persist).
    """
    track_audio_id: int
    stems: StemSet
    beats: BeatGrid
    cues: CuePoints
    loudness: LoudnessReading
    sections: tuple[SectionEmbedding, ...]
    analyzer_versions: dict[str, str]
    essentia: EssentiaFeatures | None = None
