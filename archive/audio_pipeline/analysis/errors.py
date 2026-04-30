from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioIoError:
    kind: str  # 'not_found' | 'decode' | 'unsupported'
    path: str
    detail: str


@dataclass(frozen=True)
class StemError:
    kind: str  # 'model_load' | 'inference' | 'disk'
    detail: str


@dataclass(frozen=True)
class BeatError:
    kind: str  # 'model_load' | 'inference'
    detail: str


@dataclass(frozen=True)
class CueError:
    kind: str  # 'model_load' | 'inference' | 'not_edm'
    detail: str


@dataclass(frozen=True)
class MertError:
    kind: str  # 'model_load' | 'inference' | 'empty_section'
    detail: str


@dataclass(frozen=True)
class LoudnessError:
    kind: str  # 'signal_too_short' | 'nan'
    detail: str


type AnalysisError = (
    AudioIoError | StemError | BeatError | CueError | MertError | LoudnessError
)
