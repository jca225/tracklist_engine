"""Provenance-first core types (§1/§2/§4 of the engine contract).

Faithful but grounded subset of docs/engine/dj_engine_pseudocode.md — the types
the whole migration hangs off. Deliberately minimal: model/environment provenance
(§8 ProcessSpec/EnvironmentSpec) is deferred to Phase 3; a Run here carries the
producing code + params inline (partial Law 14) until that lands. Claims/Evidence/
AxisBelief (§5/§9) arrive with the identity slice (Brick 2).

The load-bearing invariants these encode:
- An ``Artifact`` is addressed by ``content_sha256`` — bytes are identity, paths
  are not (fundamental rule #1/#4).
- A ``Run`` is an immutable record of one execution (#2); ``Derivation`` edges
  make the lineage graph (Laws 1/2).
- An ``Observation`` records what a source *said*, never canonical truth (#3);
  it carries a ``status`` so an unparseable row is a persisted diagnostic, not a
  silent drop (Law 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping, Optional

Json = dict[str, Any]


class Axis(StrEnum):
    IDENTITY = "identity"
    PLACEMENT = "placement"
    STRUCTURE = "structure"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class ArtifactKind(StrEnum):
    HTML_PAGE = "html_page"
    HTML_ROW = "html_row"
    AUDIO = "audio"
    ABLETON_SESSION = "ableton_session"
    LABEL_MANIFEST = "label_manifest"
    FEATURE_BLOB = "feature_blob"
    MODEL_CHECKPOINT = "model_checkpoint"
    DIAGNOSTICS = "diagnostics"
    PREDICTED_TIMELINE = "predicted_timeline"


class ObservationStatus(StrEnum):
    OBSERVED = "observed"
    EXPLICIT_NULL = "explicit_null"
    ABSTAINED = "abstained"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class Interval:
    """A mix/ref-time span. ``None`` means unknown — NEVER 0.0 (Law 11)."""

    start_s: Optional[float]
    end_s: Optional[float]


@dataclass(frozen=True)
class SubjectRef:
    """A stable referent an observation/claim is *about* (a slot, an asset).

    ``subject_id`` is a locator into the domain (e.g. a slot_label + set_id key),
    explicitly NOT a musical identity — that is only ever a decided belief.
    """

    subject_type: str
    subject_id: str


@dataclass(frozen=True)
class Artifact:
    """Immutable, content-addressed bytes. ``content_sha256`` is the identity."""

    content_sha256: str
    kind: ArtifactKind
    media_type: str
    byte_size: int
    object_uri: str
    created_at: datetime
    source_uri: Optional[str] = None
    source_system: Optional[str] = None
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True)
class Run:
    """An immutable record of one execution of a versioned process."""

    run_id: str
    process: str
    process_version: str
    code_commit: str
    params_hash: str
    status: RunStatus
    started_at: datetime
    finished_at: Optional[datetime] = None
    random_seed: Optional[int] = None
    error: Optional[Json] = None


@dataclass(frozen=True)
class Derivation:
    """A lineage edge: ``child`` was produced from ``parent`` by ``run``."""

    child_sha256: str
    parent_sha256: str
    run_id: str
    relation: str


@dataclass(frozen=True)
class Observation:
    """What a source said about a subject — a contestable record, not truth.

    ``status`` distinguishes a real value from an explicit null, an abstention,
    or a malformed parse (so nothing is ever silently dropped — Law 6).
    ``diagnostic_code`` is set for abstained/malformed.
    """

    observation_id: str
    subject: SubjectRef
    predicate: str
    value: Any
    source_sha256: str
    producer_run_id: str
    observed_at: datetime
    status: ObservationStatus = ObservationStatus.OBSERVED
    source_confidence: Optional[float] = None
    diagnostic_code: Optional[str] = None


# --- §5/§9 claims, evidence, beliefs --------------------------------------
# Identity/placement/structure each get their OWN claim → evidence → belief →
# decision chain (Law 10 — never fused into one score). A tokenizer "claim" is
# just one piece of Evidence with source_family="tokenizer_claim"; audio
# perception is more Evidence — the belief weighs them, it does not inherit the
# tracklist's word as identity (Law 4/5).


class Decision(StrEnum):
    ACCEPTED = "accepted"
    UNRESOLVED = "unresolved"  # abstain — below posterior floor / above entropy ceiling
    REJECTED = "rejected"


class EvidenceDirection(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"
    ABSTAINS = "abstains"


@dataclass(frozen=True)
class Claim:
    """A contestable proposition on one axis about one subject.

    e.g. axis=IDENTITY, subject=(set-slot), predicate="is_recording",
    candidate=(recording). The candidate is a *hypothesis*, never a settled fact
    until a belief decides it.
    """

    claim_id: str
    axis: Axis
    subject: SubjectRef
    predicate: str
    candidate: Optional[SubjectRef]
    context: Json
    created_by_run_id: str


@dataclass(frozen=True)
class Evidence:
    """One source's signal for/against a claim, with its own uncertainty."""

    evidence_id: str
    claim_id: str
    axis: Axis
    source_family: str  # "tokenizer_claim" | "mert_identity_head" | ...
    producer_run_id: str
    direction: EvidenceDirection
    native_score: Json
    uncertainty: Json


@dataclass(frozen=True)
class AxisBelief:
    """The decided posterior over candidates for one subject on one axis.

    ``chosen`` is the accepted candidate id, or ``None`` when the decision
    abstains (UNRESOLVED) — an abstention is a first-class, persisted outcome,
    never a silent guess.
    """

    belief_id: str
    subject: SubjectRef
    axis: Axis
    inference_run_id: str
    posterior: Mapping[str, float]
    entropy: float
    decision: Decision
    chosen: Optional[str]
    contributing_evidence_ids: tuple[str, ...]
    supersedes_belief_id: Optional[str] = None


@dataclass(frozen=True)
class DecisionRule:
    """Margin/entropy gate turning a posterior into a decision. Tuned on ONE set,
    validated on the OTHER (never fit on both — Law 19)."""

    decision_rule_id: str
    axis: Axis
    version: str
    minimum_posterior: float
    maximum_entropy: float
    calibration_status: str = "development_only"  # | corpus_supported
