# Provenance-First DJ Alignment Engine

Pythonic architectural pseudocode. This is a behavioral contract, not directly
executable production code.

# Notes:
Metaprogramming with decorators
Decorators can register types and processors declaratively.
```
@artifact_type(
    namespace="ableton",
    name="live-session",
    version=1,
    media_types={"application/gzip", "application/xml"},
)
...
@transformation(
    name="ableton.parse-session",
    version=3,
    inputs={
        "session": TypePattern("ableton:live-session@>=1,<2"),
    },
    outputs={
        "clips": TypeId("ableton", "parsed-clips", 2),
        "diagnostics": TypeId("core", "diagnostics", 1),
    },
    reproducibility="exact",
)
...
@predicate(
    namespace="tracklist",
    name="claimed-title",
    version=1,
    subject_types={"set-slot"},
    value_schema={"type": "string"},
)
...

# 3. Build dependency graphs from function signatures
@asset
def track_rows(page: HtmlPage) -> list[HtmlRow]:
    ...

@asset
def observations(rows: list[HtmlRow]) -> list[Observation]:
    ...

@asset
def identity_evidence(
    observations: list[Observation],
    fingerprints: list[Fingerprint],
) -> list[EvidenceEmission]:
    ...

# Generate property-based system laws

# Protocols, generics, state machines, 
# 
```

## 0. Fundamental rules

```python
# Source bytes are immutable and content-addressed.
# Runs are immutable records of one execution.
# Observations record what a source said; they are not canonical truth.
# Source IDs and paths are locators, not musical identity.
# Claims are contestable propositions.
# Evidence supports, contradicts, or abstains on claims.
# Identity, placement, and structure have separate beliefs and decisions.
# Human labels are immutable field-level assertions with uncertainty models.
# Corrections supersede assertions/beliefs; history is never overwritten.
# Models, training sets, parameters, code, and environments are versioned.
# Round r outputs may feed round r+1, never round r itself.
# Only promoted immutable snapshots are visible to consumers.
# Every published answer must explain both data and algorithmic provenance.
```

## 1. Shared types

```python
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import (
    Any, BinaryIO, Generic, Iterable, Mapping, Optional, Protocol,
    Sequence, TypeVar, NewType,
)
from uuid import UUID, uuid4

ArtifactId = NewType("ArtifactId", UUID)
RunId = NewType("RunId", UUID)
SetId = NewType("SetId", UUID)
SlotId = NewType("SlotId", UUID)
WorkId = NewType("WorkId", UUID)
RecordingId = NewType("RecordingId", UUID)
AudioAssetId = NewType("AudioAssetId", UUID)
OccurrenceId = NewType("OccurrenceId", UUID)
ClaimId = NewType("ClaimId", UUID)
EvidenceId = NewType("EvidenceId", UUID)
BeliefId = NewType("BeliefId", UUID)
SnapshotId = NewType("SnapshotId", UUID)
ModelId = NewType("ModelId", UUID)
RoundId = NewType("RoundId", UUID)

Json = dict[str, Any]


class Axis(StrEnum):
    IDENTITY = "identity"
    PLACEMENT = "placement"
    STRUCTURE = "structure"


class Decision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"
    QUARANTINED = "quarantined"


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
    EXTERNAL_RESPONSE = "external_response"
    FEATURE_BLOB = "feature_blob"
    MODEL_CHECKPOINT = "model_checkpoint"
    FITTED_MODEL = "fitted_model"
    SOURCE_ARCHIVE = "source_archive"
    DEPENDENCY_LOCK = "dependency_lock"
    DIAGNOSTICS = "diagnostics"


class EvidenceDirection(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"
    ABSTAINS = "abstains"


@dataclass(frozen=True)
class Interval:
    start_s: Optional[float]
    end_s: Optional[float]


@dataclass(frozen=True)
class CurvePoint:
    offset_s: float
    value: float
```

## 2. Immutable artifacts and execution provenance

```python
@dataclass(frozen=True)
class Artifact:
    artifact_id: ArtifactId
    kind: ArtifactKind
    content_sha256: str
    payload_sha256: Optional[str]
    media_type: str
    byte_size: int
    object_uri: str
    created_at: datetime
    source_uri: Optional[str]
    source_system: Optional[str]
    source_observed_at: Optional[datetime]
    metadata: Json


@dataclass(frozen=True)
class ProcessSpec:
    process_spec_id: UUID
    name: str
    version: str
    stage: str
    source_code_artifact_id: ArtifactId
    code_commit: str
    entrypoint: str
    parameter_set_id: UUID
    environment_spec_id: UUID
    dependency_lock_artifact_id: ArtifactId
    model_artifact_ids: tuple[ArtifactId, ...]
    implementation_hash: str


@dataclass(frozen=True)
class ParameterSet:
    parameter_set_id: UUID
    namespace: str
    values: Json
    canonical_hash: str


@dataclass(frozen=True)
class EnvironmentSpec:
    environment_spec_id: UUID
    os: str
    architecture: str
    runtime: str
    dependency_lock_hash: str
    container_digest: Optional[str]
    numeric_backend: Json


@dataclass
class Run:
    run_id: RunId
    process_spec_id: UUID
    parent_run_id: Optional[RunId]
    status: RunStatus
    random_seed: Optional[int]
    environment_instance: Json
    started_at: datetime
    finished_at: Optional[datetime] = None
    error: Optional[Json] = None


@dataclass(frozen=True)
class RunInput:
    run_id: RunId
    artifact_id: ArtifactId
    role: str
    ordinal: int


@dataclass(frozen=True)
class RunOutput:
    run_id: RunId
    artifact_id: ArtifactId
    role: str
    ordinal: int


@dataclass(frozen=True)
class Derivation:
    child_artifact_id: ArtifactId
    parent_artifact_id: ArtifactId
    run_id: RunId
    relation: str


class ArtifactStore(Protocol):
    def put_bytes(
        self, content: bytes, *, kind: ArtifactKind,
        media_type: str, metadata: Json,
        source_uri: Optional[str] = None,
    ) -> Artifact: ...

    def put_file(
        self, path: Path, *, kind: ArtifactKind,
        media_type: str, metadata: Json,
        source_uri: Optional[str] = None,
    ) -> Artifact: ...

    def open(self, artifact_id: ArtifactId) -> BinaryIO: ...
    def verify(self, artifact_id: ArtifactId) -> bool: ...


class ProvenanceRepository(Protocol):
    def begin_run(self, spec: ProcessSpec, *, seed: Optional[int] = None) -> Run: ...
    def add_input(self, run: Run, artifact: Artifact, role: str) -> None: ...
    def add_output(self, run: Run, artifact: Artifact, role: str) -> None: ...
    def derive(self, child: Artifact, parent: Artifact, run: Run, relation: str) -> None: ...
    def succeed(self, run: Run) -> None: ...
    def fail(self, run: Run, error: Json) -> None: ...
    def quarantine(self, run: Run, diagnostics: Json) -> None: ...
    def closure(self, node_id: UUID) -> set[UUID]: ...
```

## 3. Typed domain objects

```python
@dataclass(frozen=True)
class DjSet:
    set_id: SetId
    stable_key: str
    created_at: datetime


@dataclass(frozen=True)
class SetSlot:
    slot_id: SlotId
    set_id: SetId
    row_index: int
    source_row_artifact_id: ArtifactId


@dataclass(frozen=True)
class Work:
    work_id: WorkId


@dataclass(frozen=True)
class Recording:
    recording_id: RecordingId
    work_id: Optional[WorkId]


@dataclass(frozen=True)
class AudioAsset:
    audio_asset_id: AudioAssetId
    artifact_id: ArtifactId
    duration_s: Optional[float]
    sample_rate: Optional[int]
    codec: Optional[str]
    stem: str
    variant: str


@dataclass(frozen=True)
class Occurrence:
    occurrence_id: OccurrenceId
    set_id: SetId


@dataclass(frozen=True)
class ExternalIdentifierAssertion:
    assertion_id: UUID
    namespace: str
    value: str
    proposed_subject_id: UUID
    source_observation_id: UUID
    belief_id: Optional[BeliefId]
```

## 4. Source observations

```python
@dataclass(frozen=True)
class SubjectRef:
    subject_type: str
    subject_id: UUID


@dataclass(frozen=True)
class Observation:
    observation_id: UUID
    subject: SubjectRef
    predicate: str
    value: Any
    source_artifact_id: ArtifactId
    producer_run_id: RunId
    source_locator: Json
    observed_at: datetime
    status: str                 # observed | explicit_null | abstained | malformed
    source_confidence: Optional[float]
    diagnostic_code: Optional[str]


@dataclass(frozen=True)
class RelationObservation:
    observation_id: UUID
    subject: SubjectRef
    predicate: str
    object: SubjectRef
    source_artifact_id: ArtifactId
    producer_run_id: RunId
    source_locator: Json


@dataclass(frozen=True)
class ParseDiagnostic:
    artifact_id: ArtifactId
    run_id: RunId
    code: str
    severity: str
    detail: Json


class SourceAdapter(Protocol):
    process_spec: ProcessSpec
    def acquire(self, request: Json) -> Artifact: ...
    def partition(self, source: Artifact) -> Sequence[Artifact]: ...
    def observe(self, artifact: Artifact) -> Sequence[Observation]: ...


def crawl_and_observe_tracklist(
    request: Json,
    adapter: SourceAdapter,
    artifacts: ArtifactStore,
    provenance: ProvenanceRepository,
) -> tuple[Artifact, list[Observation]]:
    run = provenance.begin_run(adapter.process_spec)
    try:
        page = adapter.acquire(request)
        provenance.add_output(run, page, "source_page")

        observations: list[Observation] = []
        for row in adapter.partition(page):
            provenance.add_output(run, row, "source_row")
            provenance.derive(row, page, run, "partitioned_from")
            try:
                parsed = list(adapter.observe(row))
                if not parsed:
                    parsed = [explicit_abstention(row, run, "unknown_row_type")]
                observations.extend(parsed)
            except Exception as exc:
                observations.append(
                    malformed_observation(row, run, "parser_error", repr(exc))
                )

        provenance.succeed(run)
        return page, observations
    except Exception as exc:
        provenance.fail(run, {"error": repr(exc)})
        raise


def materialize_track_row(row: Artifact, parsed: Json, run: Run) -> list[Observation]:
    slot = get_or_create_slot(
        set_key=parsed["set_key"],
        row_index=parsed["row_index"],
        source_row_artifact_id=row.artifact_id,
    )
    return [
        observe(slot, "source_track_key", parsed.get("track_key"), row, run),
        observe(slot, "claimed_title", parsed.get("title"), row, run),
        observe(slot, "claimed_artists", parsed.get("artists"), row, run),
        observe(slot, "claimed_version", parsed.get("version"), row, run),
        observe(slot, "claimed_stem", parsed.get("stem"), row, run),
        observe(slot, "claimed_variant", parsed.get("variant"), row, run),
        observe(slot, "cue_seconds", parsed.get("cue_seconds"), row, run),
        observe(slot, "is_id", parsed.get("is_id"), row, run),
        observe(slot, "mashup_count", parsed.get("mashup_count"), row, run),
    ]
    # Forbidden: Recording(recording_id=parsed["track_key"])
```

## 5. Claims and evidence

```python
@dataclass(frozen=True)
class Claim:
    claim_id: ClaimId
    axis: Axis
    subject: SubjectRef
    predicate: str
    object_entity: Optional[SubjectRef]
    object_value: Optional[Any]
    context: Json
    created_by_run_id: RunId
    created_at: datetime

    def validate(self) -> None:
        assert (self.object_entity is None) != (self.object_value is None)


@dataclass(frozen=True)
class Evidence:
    evidence_id: EvidenceId
    claim_id: ClaimId
    axis: Axis
    source_type: str
    source_ref_id: UUID
    producer_run_id: RunId
    direction: EvidenceDirection
    native_score: Json
    source_family: str
    uncertainty_model: Json


@dataclass(frozen=True)
class InferenceUnit:
    unit_id: UUID
    set_id: SetId
    axis: Axis
    subject: SubjectRef
    candidate: Optional[SubjectRef]
    mix_region: Optional[Interval]
    generated_by_run_id: RunId
    generation_parameters_hash: str
    split: str                   # train | development | tripwire


@dataclass(frozen=True)
class EvidenceEmission:
    emission_id: UUID
    unit_id: UUID
    source_spec_id: UUID
    producer_run_id: RunId
    value: Any
    abstained: bool
    native_score: Json
    evidence_ids: tuple[EvidenceId, ...]
    source_family: str


class EvidenceSource(Protocol):
    process_spec: ProcessSpec
    source_family: str
    def evaluate(self, unit: InferenceUnit, context: "EvidenceContext") -> EvidenceEmission: ...


@dataclass(frozen=True)
class EvidenceContext:
    observations_snapshot_id: SnapshotId
    feature_snapshot_id: SnapshotId
    human_label_snapshot_id: SnapshotId
    artifact_store: ArtifactStore


def emit_all_sources(
    units: Sequence[InferenceUnit],
    sources: Sequence[EvidenceSource],
    context: EvidenceContext,
) -> list[EvidenceEmission]:
    emissions = []
    for source in sources:
        run = begin_versioned_run(source.process_spec)
        for unit in units:
            try:
                emission = source.evaluate(unit, context)
            except UnsupportedInput as exc:
                emission = abstaining_emission(unit, source, run, str(exc))
            persist(emission)       # abstentions are persisted
            emissions.append(emission)
        finish(run)
    return emissions
```

## 6. Audio acquisition and analysis

```python
class Downloader(Protocol):
    process_spec: ProcessSpec
    def fetch(self, candidate: Json) -> Path: ...


def acquire_audio(
    link_observation: Observation,
    downloader: Downloader,
    artifacts: ArtifactStore,
) -> Optional[AudioAsset]:
    run = begin_versioned_run(downloader.process_spec)

    for candidate in rank_download_candidates(link_observation):
        record_attempt(run, candidate)
        try:
            path = downloader.fetch(candidate)
            artifact = artifacts.put_file(
                path,
                kind=ArtifactKind.AUDIO,
                media_type=probe_media_type(path),
                metadata=probe_audio(path),
                source_uri=candidate["url"],
            )
            if not duration_sane(artifact, link_observation):
                quarantine_artifact(artifact, "duration_suspect")
                continue

            asset = AudioAsset(
                audio_asset_id=AudioAssetId(uuid4()),
                artifact_id=artifact.artifact_id,
                **audio_fields(artifact),
                stem="regular",
                variant="regular",
            )
            persist(asset)
            emit_claim(
                axis=Axis.IDENTITY,
                subject=ref(asset),
                predicate="may_realize_recording",
                object_value={"source_track_key": link_observation.value},
                run=run,
            )
            finish(run)
            return asset
        except DownloadFailure as exc:
            record_attempt_failure(run, candidate, exc)

    fail(run, "all_candidates_exhausted")
    return None


class Analyzer(Protocol):
    process_spec: ProcessSpec
    feature_kind: str
    def analyze(self, audio: BinaryIO) -> Any: ...


def analyze_audio(asset: AudioAsset, analyzers: Sequence[Analyzer]) -> list[Artifact]:
    outputs = []
    for analyzer in analyzers:
        run = begin_versioned_run(analyzer.process_spec)
        source = load_artifact(asset.artifact_id)
        add_run_input(run, source, "audio")
        value = analyzer.analyze(open_artifact(source))
        feature = store_serialized_feature(
            value,
            kind=analyzer.feature_kind,
            metadata={"coordinate_system": describe_coordinates(value)},
        )
        add_run_output(run, feature, analyzer.feature_kind)
        derive(feature, source, run, "analyzed_from")
        finish(run)
        outputs.append(feature)
    return outputs
```

## 7. Immutable Ableton human labeling

```python
@dataclass(frozen=True)
class LabelingBundle:
    bundle_id: UUID
    set_id: SetId
    als_artifact_id: ArtifactId
    manifest_artifact_id: ArtifactId
    audio_artifact_ids: tuple[ArtifactId, ...]
    bundle_hash: str
    annotator_id: str
    exported_at: datetime


@dataclass(frozen=True)
class HumanLabelAssertion:
    assertion_id: UUID
    bundle_id: UUID
    import_run_id: RunId
    annotator_id: str
    subject: SubjectRef
    field: str
    observed_value: Any
    source_clip_locator: Json
    source_audio_artifact_id: Optional[ArtifactId]
    uncertainty_model: Json
    ambiguity_candidates: tuple[Any, ...]
    review_status: str
    supersedes_assertion_id: Optional[UUID]
    created_at: datetime


def import_ableton_bundle(bundle: LabelingBundle) -> list[HumanLabelAssertion]:
    run = begin_versioned_run(ABLETON_IMPORT_SPEC)
    session = parse_losslessly(bundle.als_artifact_id)
    diagnostics = validate_session(session)
    if diagnostics.fatal:
        quarantine(run, diagnostics)
        return []

    assertions = []
    for clip in session.clips:
        identity = resolve_clip_by_content(
            clip,
            permitted_artifacts=bundle.audio_artifact_ids,
        )
        if not identity.unique:
            assertions.append(human_abstention(bundle, run, clip, identity.reason))
            continue

        mapping = map_clip_to_mix_time(clip, session.mix_reference)
        assertions.extend([
            human_assertion(
                bundle, run, clip, "recording_id", identity.recording_id,
                categorical_prior(mean=.99, effective_sample_size=20),
            ),
            human_assertion(
                bundle, run, clip, "mix_start_s", mapping.mix_start_s,
                student_t_error(scale_s=.25, df=4),
            ),
            human_assertion(
                bundle, run, clip, "mix_end_s", mapping.mix_end_s,
                student_t_error(scale_s=.25, df=4),
            ),
            human_assertion(
                bundle, run, clip, "tempo_curve", mapping.tempo_curve,
                curve_error(control_point_scale=.003),
            ),
        ])
    persist_all(assertions)
    finish(run)
    return assertions


def revise_assertion(old: HumanLabelAssertion, new: HumanLabelAssertion) -> None:
    persist(replace(new, supersedes_assertion_id=old.assertion_id))
    enqueue_recomputation(
        provenance_descendants(old.assertion_id),
        reason="human_assertion_superseded",
    )
    # Never delete old, never rewrite historical label snapshots.
```

## 8. Algorithm and model provenance

```python
@dataclass(frozen=True)
class TrainingSnapshot:
    training_snapshot_id: SnapshotId
    round_id: RoundId
    unit_snapshot_id: SnapshotId
    emission_snapshot_id: SnapshotId
    human_label_snapshot_id: SnapshotId
    pseudo_label_snapshot_id: Optional[SnapshotId]
    split_assignment_id: UUID
    snapshot_hash: str


@dataclass
class ModelFitRun:
    fit_run_id: RunId
    round_id: RoundId
    axis: Axis
    algorithm_spec_id: UUID
    training_snapshot_id: SnapshotId
    initialization_model_id: Optional[ModelId]
    random_seed: int
    status: RunStatus


@dataclass(frozen=True)
class FittedModel:
    fitted_model_id: ModelId
    axis: Axis
    model_spec_id: UUID
    fit_run_id: RunId
    serialized_state_artifact_id: ArtifactId
    training_snapshot_id: SnapshotId
    parent_fitted_model_id: Optional[ModelId]
    model_state_hash: str
    status: str


class BeliefModelAlgorithm(Protocol):
    process_spec: ProcessSpec
    def fit(self, axis: Axis, training: TrainingSnapshot, seed: int) -> Any: ...
    def infer(
        self, fitted_state: Any,
        units: Sequence[InferenceUnit],
        emissions: Sequence[EvidenceEmission],
    ) -> Sequence["AxisBelief"]: ...


def fit_and_store_model(
    round_id: RoundId,
    axis: Axis,
    algorithm: BeliefModelAlgorithm,
    training: TrainingSnapshot,
    parent_model: Optional[FittedModel],
    seed: int,
) -> FittedModel:
    run = begin_fit_run(round_id, axis, algorithm.process_spec, training, seed)
    state = algorithm.fit(axis, training, seed)
    state_artifact = store_model_state(
        state,
        metadata={
            "axis": axis,
            "declared_dependencies": state.declared_dependencies,
            "learned_dependencies": state.learned_dependencies,
            "source_accuracy_estimates": state.source_accuracy_estimates,
            "stopping_reason": state.stopping_reason,
        },
    )
    model = FittedModel(
        fitted_model_id=ModelId(uuid4()),
        axis=axis,
        model_spec_id=algorithm.process_spec.process_spec_id,
        fit_run_id=run.run_id,
        serialized_state_artifact_id=state_artifact.artifact_id,
        training_snapshot_id=training.training_snapshot_id,
        parent_fitted_model_id=parent_model.fitted_model_id if parent_model else None,
        model_state_hash=state_artifact.content_sha256,
        status="candidate",
    )
    persist(model)
    finish(run)
    return model
```

## 9. Beliefs and axis-specific decisions

```python
@dataclass(frozen=True)
class AxisBelief:
    belief_id: BeliefId
    unit_id: UUID
    axis: Axis
    inference_run_id: RunId
    fitted_model_id: ModelId
    posterior: Mapping[str, float]
    entropy: float
    decision: Decision
    decision_rule_id: UUID
    contributing_emission_ids: tuple[UUID, ...]
    supersedes_belief_id: Optional[BeliefId]


@dataclass(frozen=True)
class DecisionRule:
    decision_rule_id: UUID
    axis: Axis
    version: str
    rule: Json
    selected_using_panel_id: Optional[UUID]
    calibration_status: str       # development_only | corpus_supported


def decide(posterior: Mapping[str, float], entropy: float, rule: DecisionRule) -> Decision:
    winner, probability = max(posterior.items(), key=lambda item: item[1])
    if probability < rule.rule["minimum_posterior"]:
        return Decision.UNRESOLVED
    if entropy > rule.rule["maximum_entropy"]:
        return Decision.UNRESOLVED
    return Decision.ACCEPTED


def infer_beliefs(
    model: FittedModel,
    algorithm: BeliefModelAlgorithm,
    units: Sequence[InferenceUnit],
    emissions: Sequence[EvidenceEmission],
    rule: DecisionRule,
) -> list[AxisBelief]:
    run = begin_prediction_run(model, units, emissions, rule)
    state = load_model_state(model.serialized_state_artifact_id)
    raw_beliefs = algorithm.infer(state, units, emissions)
    beliefs = [
        replace(
            belief,
            inference_run_id=run.run_id,
            fitted_model_id=model.fitted_model_id,
            decision=decide(belief.posterior, belief.entropy, rule),
            decision_rule_id=rule.decision_rule_id,
        )
        for belief in raw_beliefs
    ]
    persist_all(beliefs)
    finish(run)
    return beliefs
```

## 10. Rich timeline reconstruction

```python
@dataclass(frozen=True)
class AlignmentSegment:
    segment_id: UUID
    occurrence_id: OccurrenceId
    recording_id: Optional[RecordingId]
    source_audio_asset_id: Optional[AudioAssetId]
    mix_interval: Interval
    reference_interval: Interval
    tempo_curve: tuple[CurvePoint, ...]
    pitch_curve: tuple[CurvePoint, ...]
    gain_envelope: tuple[CurvePoint, ...]
    channel_role: str
    structure_kind: str
    parent_segment_id: Optional[UUID]


@dataclass(frozen=True)
class PredictedOccurrence:
    occurrence_id: OccurrenceId
    identity_belief_id: BeliefId
    placement_belief_id: BeliefId
    structure_belief_id: BeliefId
    recording_id: Optional[RecordingId]
    segments: tuple[AlignmentSegment, ...]


@dataclass(frozen=True)
class PredictedTimeline:
    set_id: SetId
    occurrences: tuple[PredictedOccurrence, ...]
    decoder_run_id: RunId


class TimelineDecoder(Protocol):
    process_spec: ProcessSpec
    def decode(
        self,
        set_id: SetId,
        identity: Sequence[AxisBelief],
        placement: Sequence[AxisBelief],
        structure: Sequence[AxisBelief],
    ) -> PredictedTimeline: ...


def decode_timeline(
    set_id: SetId,
    beliefs: Sequence[AxisBelief],
    decoder: TimelineDecoder,
) -> PredictedTimeline:
    run = begin_versioned_run(decoder.process_spec)
    timeline = decoder.decode(
        set_id,
        identity=by_axis(beliefs, Axis.IDENTITY),
        placement=by_axis(beliefs, Axis.PLACEMENT),
        structure=by_axis(beliefs, Axis.STRUCTURE),
    )
    validate_no_fake_coordinates(timeline)  # unknown is None, never 0.0
    persist(timeline)
    finish(run)
    return replace(timeline, decoder_run_id=run.run_id)
```

## 11. Snapshots, evaluation, and promotion

```python
@dataclass(frozen=True)
class Snapshot:
    snapshot_id: SnapshotId
    kind: str
    parent_snapshot_id: Optional[SnapshotId]
    created_by_run_id: RunId
    member_ids: tuple[UUID, ...]
    snapshot_hash: str
    status: str
    created_at: datetime


@dataclass(frozen=True)
class GoldDevelopmentPanel:
    panel_id: UUID
    label_snapshot_ids: tuple[SnapshotId, ...]
    independent_set_count: int = 2
    intended_use: str = "development"
    allows_corpus_calibration_claims: bool = False


@dataclass(frozen=True)
class EvaluationResult:
    evaluation_run_id: RunId
    prediction_snapshot_id: SnapshotId
    label_snapshot_id: SnapshotId
    set_id: SetId
    axis: Axis
    metrics: Json
    n_sets: int
    n_units: int
    interval: Optional[Json]
    interpretation: str


@dataclass(frozen=True)
class GateResult:
    gate_spec_id: UUID
    passed: bool
    diagnostics: Json


def evaluate_gold_panel(
    algorithm_bundle: "AlgorithmBundle",
    panel: GoldDevelopmentPanel,
) -> list[EvaluationResult]:
    assert panel.independent_set_count == 2
    results = []
    for development_set, held_set in leave_one_set_out(panel):
        configured = tune_on_only(algorithm_bundle, development_set)
        frozen_bundle = freeze_algorithm_bundle(configured)
        prediction = predict_without_tuning(frozen_bundle, held_set)
        results.extend(
            evaluate_per_set_per_axis(
                prediction,
                frozen_label_snapshot(held_set),
                interpretation="development_only",
            )
        )
    return results


def promotion_gates(candidate: Snapshot) -> list[GateResult]:
    return [
        gate_provenance_complete(candidate),
        gate_provenance_acyclic(candidate),
        gate_artifacts_exist(candidate),
        gate_no_path_or_source_key_identity(candidate),
        gate_axis_decisions_separate(candidate),
        gate_no_fake_coordinates(candidate),
        gate_human_labels_are_versioned(candidate),
        gate_human_uncertainty_is_field_specific(candidate),
        gate_canaries(candidate),
        gate_dependency_agreement(candidate),
        gate_leave_one_source_family_out(candidate),
        gate_reconstruction_plausibility(candidate),  # never certifies identity
        gate_metrics_per_set_per_axis(candidate),
        gate_calibration_claim_is_honest(candidate),
        gate_random_high_confidence_audit(candidate),
    ]


def promote(candidate: Snapshot) -> Snapshot:
    results = promotion_gates(candidate)
    persist_all(results)
    if any(not result.passed for result in results):
        return replace_snapshot_status(candidate, "rejected")

    with database_transaction():
        supersede_current_snapshot(candidate.kind)
        promoted = replace_snapshot_status(candidate, "promoted")
        set_current_snapshot_pointer(candidate.kind, promoted.snapshot_id)
    return promoted
```

## 12. Co-training rounds and pseudo-label lineage

```python
@dataclass(frozen=True)
class AlgorithmBundle:
    algorithm_bundle_id: UUID
    candidate_generator_spec_id: UUID
    evidence_source_spec_ids: tuple[UUID, ...]
    belief_model_spec_ids: Mapping[Axis, UUID]
    decoder_spec_id: UUID
    decision_rule_ids: Mapping[Axis, UUID]
    gate_spec_ids: tuple[UUID, ...]
    bundle_hash: str


@dataclass
class CoTrainingRound:
    round_id: RoundId
    round_index: int
    parent_round_id: Optional[RoundId]
    input_snapshot_id: SnapshotId
    algorithm_bundle_id: UUID
    fitted_model_ids: Mapping[Axis, ModelId]
    prediction_snapshot_id: Optional[SnapshotId]
    status: str


@dataclass(frozen=True)
class PseudoLabel:
    pseudo_label_id: UUID
    produced_in_round_id: RoundId
    unit_id: UUID
    axis: Axis
    value: Any
    source_belief_id: BeliefId
    source_prediction_run_id: RunId
    source_fitted_model_id: ModelId
    contributing_emission_ids: tuple[UUID, ...]
    confidence_at_acceptance: float
    entropy_at_acceptance: float
    target_view: str
    eligible_from_round_index: int
    revoked_at: Optional[datetime]


def validate_pseudo_label_for_next_round(
    label: PseudoLabel,
    target_round: CoTrainingRound,
    target_source_family: str,
) -> None:
    closure = provenance_closure(label.pseudo_label_id)
    if target_round.round_id in closure:
        raise ProvenanceCycle("round cannot consume its own descendants")
    if closure_contains_source_family(closure, target_source_family):
        raise ViewLeakage("pseudo-label returned to its originating source family")
    if not closure_has_independent_falsifiable_root(closure):
        raise UnfalsifiablePseudoLabel("no independent evidential root")


def run_cotraining_round(
    corpus: Snapshot,
    bundle: AlgorithmBundle,
    parent: Optional[CoTrainingRound],
) -> CoTrainingRound:
    round = create_round(corpus, bundle, parent)

    units = enumerate_inference_units(corpus, bundle)
    unit_snapshot = freeze_snapshot("units", units)

    emissions = emit_all_sources(
        units,
        load_evidence_sources(bundle),
        evidence_context(corpus),
    )
    emission_snapshot = freeze_snapshot("emissions", emissions)

    prior_pseudo_labels = (
        eligible_pseudo_labels(parent)
        if parent is not None
        else []
    )
    for label in prior_pseudo_labels:
        validate_pseudo_label_for_next_round(
            label, round, target_source_family=label.target_view
        )

    training = freeze_training_snapshot(
        round=round,
        unit_snapshot=unit_snapshot,
        emission_snapshot=emission_snapshot,
        human_labels=current_frozen_human_label_snapshot(corpus),
        pseudo_labels=freeze_snapshot("pseudo_labels", prior_pseudo_labels),
        split_assignment=frozen_split_assignment(corpus),
    )

    models: dict[Axis, FittedModel] = {}
    beliefs: list[AxisBelief] = []
    for axis in Axis:
        algorithm = load_belief_model_algorithm(bundle, axis)
        model = fit_and_store_model(
            round.round_id,
            axis,
            algorithm,
            training,
            parent_model=model_from(parent, axis) if parent else None,
            seed=round_seed(round, axis),
        )
        models[axis] = model
        beliefs.extend(
            infer_beliefs(
                model,
                algorithm,
                units=by_axis(units, axis),
                emissions=emissions_for_axis(emissions, axis),
                rule=load_decision_rule(bundle, axis),
            )
        )

    belief_snapshot = freeze_snapshot("beliefs", beliefs)
    timeline = decode_timeline(
        set_id=set_of(corpus),
        beliefs=beliefs,
        decoder=load_decoder(bundle),
    )
    prediction_snapshot = freeze_snapshot("prediction", [timeline])

    evaluation = evaluate_candidate(
        prediction_snapshot,
        gold_panel=current_gold_development_panel(),
    )
    candidate = assemble_candidate_snapshot(
        corpus, models, belief_snapshot, prediction_snapshot, evaluation
    )
    promoted = promote(candidate)

    if promoted.status != "promoted":
        reject_round(round)
        enqueue_active_review(disputed_or_uncertain_units(beliefs))
        return round

    promote_round(round, models, promoted)
    create_next_round_inputs(
        pseudo_labels=eligible_new_pseudo_labels(
            beliefs,
            require_falsifiable=True,
            opposite_view_only=True,
        ),
        human_tasks=disputed_or_uncertain_units(beliefs),
        acquisition_tasks=missing_evidence_requests(beliefs),
    )
    return round


def bootstrap(corpus: Snapshot, bundle: AlgorithmBundle, max_rounds: int) -> CoTrainingRound:
    parent = None
    previous_objective = None
    for _ in range(max_rounds):
        round = run_cotraining_round(corpus, bundle, parent)
        if round.status != "promoted":
            return round

        objective = mean_reconstruction_distance(round)
        if previous_objective is not None and previous_objective - objective < EPSILON:
            enqueue_active_review(unresolved_units(round), reason="objective_stalled")
            return round

        # Reconstruction is only a plausibility/stopping signal, never identity truth.
        previous_objective = objective
        corpus = build_next_generation_corpus(corpus, round)
        parent = round
    return parent
```

## 13. Corrections and selective recomputation

```python
def add_correction(
    old_claim: Claim,
    corrected_value: Any,
    correction_evidence: Evidence,
) -> Claim:
    persist(replace(correction_evidence, direction=EvidenceDirection.CONTRADICTS))
    replacement = emit_claim(
        axis=old_claim.axis,
        subject=old_claim.subject,
        predicate=old_claim.predicate,
        object_value=corrected_value,
        context={"supersedes_claim_id": str(old_claim.claim_id)},
    )
    add_supporting_evidence(replacement, correction_evidence)
    enqueue_recomputation(
        topological_sort(provenance_descendants(old_claim.claim_id)),
        reason="upstream_claim_corrected",
    )
    return replacement
```

## 14. Explanation API

```python
def explain_prediction(snapshot_id: SnapshotId, occurrence_id: OccurrenceId) -> Json:
    occurrence = load_occurrence(snapshot_id, occurrence_id)
    axis_beliefs = load_axis_beliefs(occurrence)
    prediction_run = load_prediction_run(occurrence)

    return {
        "published_snapshot": snapshot_id,
        "occurrence": occurrence,
        "axes": {
            axis.value: {
                "posterior": belief.posterior,
                "entropy": belief.entropy,
                "decision": belief.decision,
                "evidence": explain_emissions(belief.contributing_emission_ids),
            }
            for axis, belief in axis_beliefs.items()
        },
        "data_provenance": {
            "source_artifacts": provenance_roots(occurrence),
            "observations": observations_in_closure(occurrence),
            "human_assertions": human_assertions_in_closure(occurrence),
        },
        "algorithmic_provenance": {
            "prediction_run": prediction_run,
            "algorithm_spec": load_process_spec(prediction_run.process_spec_id),
            "fitted_models": [
                explain_fitted_model(belief.fitted_model_id)
                for belief in axis_beliefs.values()
            ],
            "decoder_trace": load_decoder_trace(occurrence),
            "gate_results": load_gate_results(snapshot_id),
        },
        "round_provenance": {
            "round": round_for_prediction(occurrence),
            "pseudo_label_ancestry": pseudo_label_ancestry(occurrence),
        },
        "evaluation_scope": {
            "status": "development_only",
            "independent_gold_sets": 2,
            "corpus_calibration_supported": False,
        },
    }
```

## 15. Top-level engine

```python
@dataclass(frozen=True)
class Engine:
    artifacts: ArtifactStore
    provenance: ProvenanceRepository
    tracklist_adapter: SourceAdapter
    downloaders: Sequence[Downloader]
    analyzers: Sequence[Analyzer]
    evidence_sources: Sequence[EvidenceSource]
    belief_algorithms: Mapping[Axis, BeliefModelAlgorithm]
    decoder: TimelineDecoder
    algorithm_bundle: AlgorithmBundle

    def ingest_set(self, request: Json) -> Snapshot:
        page, observations = crawl_and_observe_tracklist(
            request,
            self.tracklist_adapter,
            self.artifacts,
            self.provenance,
        )
        observation_snapshot = freeze_snapshot("observations", observations)

        source_entities = materialize_typed_domain_objects(observations)
        audio_assets = acquire_relevant_audio(
            observations,
            self.downloaders,
            self.artifacts,
        )
        feature_artifacts = [
            feature
            for asset in audio_assets
            for feature in analyze_audio(asset, self.analyzers)
        ]

        return freeze_corpus_snapshot(
            source_page=page,
            observations=observation_snapshot,
            domain_objects=source_entities,
            audio_assets=audio_assets,
            feature_artifacts=feature_artifacts,
            human_labels=current_frozen_human_label_snapshot_for(request),
        )

    def infer(self, corpus: Snapshot, max_rounds: int = 5) -> CoTrainingRound:
        return bootstrap(corpus, self.algorithm_bundle, max_rounds)

    def explain(self, snapshot_id: SnapshotId, occurrence_id: OccurrenceId) -> Json:
        return explain_prediction(snapshot_id, occurrence_id)
```

## 16. Required executable laws

```python
def assert_system_laws(snapshot: Snapshot) -> None:
    assert provenance_is_complete(snapshot)
    assert provenance_is_acyclic(snapshot)
    assert all_artifacts_verify(snapshot)
    assert no_source_key_is_canonical_recording_id(snapshot)
    assert no_path_was_used_as_identity(snapshot)
    assert every_unknown_parser_row_has_diagnostic(snapshot)
    assert all_abstentions_are_persisted(snapshot)
    assert human_history_is_append_only(snapshot)
    assert human_uncertainty_is_field_specific(snapshot)
    assert identity_placement_structure_are_separate(snapshot)
    assert no_unknown_coordinate_equals_zero(snapshot)
    assert decoder_consumes_posteriors_not_raw_margins(snapshot)
    assert every_model_has_training_snapshot(snapshot)
    assert every_model_has_code_and_environment(snapshot)
    assert no_round_consumes_its_own_outputs(snapshot)
    assert pseudo_label_graph_is_acyclic(snapshot)
    assert rejected_rounds_are_reproducible(snapshot)
    assert evaluation_is_per_set_and_per_axis(snapshot)
    assert not claims_corpus_calibration_from_two_sets(snapshot)
    assert reconstruction_does_not_certify_identity(snapshot)
    assert every_published_value_is_explainable(snapshot)
```

## 17. The complete conceptual function

```python
def system(source_world_traces):
    artifacts = preserve_exact_bytes(source_world_traces)
    observations = interpret_sources_without_declaring_truth(artifacts)
    domain_objects = establish_stable_referents(observations)
    hypotheses = enumerate_axis_specific_claims(domain_objects, observations)
    evidence = measure_support_contradiction_or_abstention(hypotheses)

    training_generation = freeze(
        hypotheses,
        evidence,
        human_assertions,
        eligible_prior_round_pseudo_labels,
    )
    fitted_models = fit_versioned_axis_models(training_generation)
    beliefs = infer_axis_posteriors(fitted_models, evidence)
    decisions = apply_versioned_decision_rules(beliefs)
    timeline = decode_globally_coherent_partial_timeline(decisions)

    candidate = freeze_everything(
        data=artifacts,
        observations=observations,
        algorithms=fitted_models,
        beliefs=beliefs,
        timeline=timeline,
    )
    return promote_only_if_auditable_and_plausible(candidate)
```
