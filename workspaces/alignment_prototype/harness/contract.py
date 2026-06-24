"""The alignment contract: one normalized result type + one probe interface.

Today every probe (chroma matched-filter, landmark fingerprint, path decode,
continuity stack, MERT head) emits a bespoke shape with an incomparable
confidence scale (votes vs peak∈[0,1] vs Viterbi log-prob vs logit vs z-score).
That makes them impossible to compose or to drive uniformly. This module fixes
the shape:

    Probe.run(MixContext, RefContext, CandidatePool) -> AlignmentResult

`confidence` is contracted to [0,1] (calibration is a separate step) so a single
abstention threshold and cross-probe agreement are meaningful. A driver — LLM
agent, deterministic pipeline, or trained model — is anything that produces
AlignmentResults against this surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..records import SlotCandidate


@dataclass(frozen=True)
class RefSegment:
    """One contiguous (mix-time -> ref-time) mapping inside a span.

    A simple placement is one segment; loops / section-cuts produce several.
    """

    mix_start_s: float
    ref_start_s: float
    ref_end_s: float


@dataclass(frozen=True)
class AlignmentResult:
    """Normalized output for one (mix span, candidate) placement decision.

    The single shape all probes and drivers emit. ``offset_s`` is the primary
    ref-time placement of the span start; ``segments`` carries the full piecewise
    map when it's not a single line. ``confidence`` is calibrated to [0,1] so
    ``abstain`` and cross-probe comparison are well-defined; ``source`` records
    which probe/driver produced it (for ablation + agreement merging).
    """

    recording_id: str | None
    offset_s: float
    ref_end_s: float | None = None
    segments: tuple[RefSegment, ...] = ()
    tempo_ratio: float | None = None
    confidence: float = 0.0
    abstain: bool = False
    source: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be calibrated to [0,1], got {self.confidence!r} "
                f"(source={self.source!r})"
            )

    @classmethod
    def abstained(
        cls, *, source: str, recording_id: str | None = None
    ) -> "AlignmentResult":
        """A no-decision result — the probe declines rather than guesses."""
        return cls(
            recording_id=recording_id,
            offset_s=0.0,
            confidence=0.0,
            abstain=True,
            source=source,
        )


@dataclass(frozen=True)
class MixContext:
    """The set mix a span is being placed into. Probes load features lazily from
    ``audio_path`` (the shared feature registry is a separate step)."""

    audio_path: Path
    set_id: str = ""
    span_start_s: float | None = None
    span_end_s: float | None = None


@dataclass(frozen=True)
class RefContext:
    """A candidate reference recording's audio."""

    recording_id: str
    audio_path: Path
    stem: str = "regular"


@dataclass(frozen=True)
class CandidatePool:
    """The library candidates a probe may select among for one slot."""

    candidates: tuple[SlotCandidate, ...] = ()

    def __iter__(self):
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)


class Probe(ABC):
    """A unit of alignment evidence. Concrete probes wrap the existing DSP/ML
    routines and emit a normalized AlignmentResult — they do not reimplement the
    math, only adapt its interface and confidence."""

    #: short stable identifier, e.g. "chroma" / "fp" / "path_decode" / "mert".
    name: str = "probe"

    @abstractmethod
    def run(
        self, mix: MixContext, ref: RefContext, candidates: CandidatePool
    ) -> AlignmentResult:
        """Produce a normalized placement decision (or AlignmentResult.abstained)."""

    def __call__(
        self, mix: MixContext, ref: RefContext, candidates: CandidatePool
    ) -> AlignmentResult:
        return self.run(mix, ref, candidates)
