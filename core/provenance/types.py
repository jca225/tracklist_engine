"""Provenance-first core types (§1/§2/§4/§8 of the engine contract).

Faithful but grounded subset of docs/engine/dj_engine_pseudocode.md — the types
the whole migration hangs off. Claims/Evidence/AxisBelief (§5/§9) arrived with
the identity slice (Brick 2); the §2/§8 versioning backbone (ParameterSet /
EnvironmentSpec / ProcessSpec / TrainingSnapshot / FittedModel) arrived with the
feature-producer foundation (Brick 7). A ``Run`` still carries process/version/
code inline for writers that predate ProcessSpec; ``begin_run_from_spec`` stamps
those fields *from* a spec so registry-produced runs are fully versioned.

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
    """Immutable, content-addressed bytes. ``content_sha256`` is the identity.

    ``kind`` is one of the built-in :class:`ArtifactKind` values OR an arbitrary
    registered kind string (Brick 7 ``@artifact_type`` — new feature kinds do
    not require enum surgery)."""

    content_sha256: str
    kind: ArtifactKind | str
    media_type: str
    byte_size: int
    object_uri: str
    created_at: datetime
    source_uri: Optional[str] = None
    source_system: Optional[str] = None
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True)
class Run:
    """An immutable record of one execution of a versioned process.

    ``process_spec_id`` is set when the run was begun from a persisted
    :class:`ProcessSpec` (``begin_run_from_spec``) — the fully-versioned path.
    Legacy ``begin_run`` leaves it ``None`` (process/version/code inline only).
    """

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
    process_spec_id: Optional[str] = None


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


# --- §7 immutable human labeling ------------------------------------------
# Human GT enters the substrate as append-only, field-level assertions hanging
# off a content-addressed labeling artifact. Two laws live here: history is
# append-only (Law 8 — a revision is a NEW assertion whose
# ``supersedes_assertion_id`` points at the old one; the old row is never
# mutated) and uncertainty is field-specific (Law 9 — every assertion carries
# its own ``uncertainty_model``; a bare point value is the violation).


@dataclass(frozen=True)
class HumanLabelBundle:
    """One labeling artifact's provenance: WHO asserted, from WHAT bytes.

    ``source_artifact_sha256`` addresses the labeling export (GT manifest /
    ``.als``) by content — the assertions are derived from that exact
    byte-sequence, not from a mutable path.
    """

    bundle_id: str
    set_id: str
    source_artifact_sha256: str
    annotator_id: str
    import_run_id: str
    created_at: datetime


@dataclass(frozen=True)
class HumanLabelAssertion:
    """One human-asserted field value on one subject — append-only.

    ``uncertainty_model`` is a field-specific error model (``categorical_prior``
    for identity-like fields, ``student_t`` for time coordinates, ``curve_error``
    for curves — §7), NEVER a global scalar and never absent (Law 9).
    ``supersedes_assertion_id`` is the only revision mechanism (Law 8).
    """

    assertion_id: str
    bundle_id: str
    subject: SubjectRef
    field: str
    observed_value: Any
    uncertainty_model: Json
    produced_run_id: str
    created_at: datetime
    source_confidence: Optional[float] = None
    supersedes_assertion_id: Optional[str] = None


# --- §2/§8 versioning backbone (Brick 7) -----------------------------------
# Everything a computation IS — code, parameters, environment, upstream models
# — versioned and content-hashed, so "identical process" has one id and a
# feature/model can name exactly what produced it (Laws 13/14).


@dataclass(frozen=True)
class ParameterSet:
    """One immutable bag of parameters. ``canonical_hash`` = sha256 of the
    sorted-JSON serialization of ``values`` — the same values always hash the
    same, key order be damned."""

    parameter_set_id: str
    namespace: str
    values: Json
    canonical_hash: str


@dataclass(frozen=True)
class EnvironmentSpec:
    """A cheap-but-honest capture of where code ran: OS, architecture, runtime,
    and a hash of the dependency lock (a requirements file, or the installed
    distribution list when none is given)."""

    environment_spec_id: str
    os: str
    architecture: str
    runtime: str
    dependency_lock_hash: str


@dataclass(frozen=True)
class ProcessSpec:
    """A versioned process: WHAT ran (name/version/stage/code), with WHICH
    parameters, in WHAT environment, on top of WHICH models.

    ``process_spec_id`` is DETERMINISTIC — sha256 over (name, version,
    code_commit, parameter canonical_hash, environment dependency_lock_hash,
    model_refs) — so recording the identical process is idempotent (one row,
    one id), never a duplicate."""

    process_spec_id: str
    name: str
    version: str
    stage: str
    code_commit: str
    parameter_set_id: str
    environment_spec_id: str
    model_refs: tuple[str, ...]
    implementation_hash: str


@dataclass(frozen=True)
class TrainingSnapshot:
    """A light frozen snapshot of WHAT a model trained on: the content shas of
    the member artifacts. ``snapshot_hash`` = sha256 over the sorted members —
    the same training set always snapshots to the same hash (Law 13)."""

    training_snapshot_id: str
    member_artifact_shas: tuple[str, ...]
    snapshot_hash: str


@dataclass(frozen=True)
class FittedModel:
    """One trained model, pinned to its axis, producing process, serialized
    state (content-addressed), and frozen training snapshot (Laws 13/14)."""

    fitted_model_id: str
    axis: Axis
    process_spec_id: str
    serialized_state_sha256: str
    training_snapshot_id: str
    model_state_hash: str
    status: str = "candidate"  # | promoted | retired (§8/§11)


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
