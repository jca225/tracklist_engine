//! Provenance-first DJ alignment kernel (`tracklist_engine/core`).
//!
//! # TEMP LEARNING COMMENTS
//!
//! Many `TEMP:` / expanded `//!` notes in this crate are **temporary pedagogy**
//! until the architecture feels familiar. Grep `TEMP:` to find/strip them later.
//! Prefer [`docs/canary_walkthrough.md`](../../docs/canary_walkthrough.md) for
//! the durable map; these comments are the in-code tour.
//!
//! # What this crate is
//!
//! The **system of record** for identity, artifacts, provenance, weak-supervision
//! Decide/Promote, and co-training Act. Python (`python/sensors`) only
//! **Proposes** (votes / abstain → JSON). This crate decides what may ship.
//!
//! Behavioral contract: repo-root `dj_engine_pseudocode.md`.
//!
//! # Module map (read in this order for canary smoke)
//!
//! 1. [`ids`] — opaque IDs; `RecordingId` mint rules (paths never mint identity)
//! 2. [`types`] — Artifact, Run, ProcessSpec, AxisBelief, Claim, …
//! 3. [`pws`] — InferenceUnit × EvidenceEmission → majority_vote → canary gate
//! 4. [`cotraining`] — round lifecycle: fit → gate → promote/reject → PseudoLabels
//! 5. [`artifact`] + [`features`] — content-addressed bytes + FeatureBlob registration
//! 6. [`provenance`] — Run / Derivation graph (SQLite or memory)
//! 7. [`laws`] — promotion checks (complete, acyclic, verify, abstentions)
//!
//! CLI that drives these: sibling crate `dj_migrate` (`pws-fit`, `run-round`).
//!
//! Programmatic weak supervision (Snorkel-like) lives in [`pws`]:
//! [`InferenceUnit`] × [`EvidenceSource`] → [`EvidenceEmission`] → beliefs.
//! Co-training rounds / pseudo-labels: [`cotraining`].

pub mod artifact;
pub mod cotraining;
pub mod error;
pub mod features;
pub mod ids;
pub mod laws;
pub mod provenance;
pub mod pws;
pub mod types;

pub use artifact::{ArtifactStore, FsArtifactStore};
pub use cotraining::{
    eligible_new_pseudo_labels, opposite_view, run_cotraining_round,
    validate_pseudo_label_for_next_round, AlgorithmBundle, CoTrainingRound, EmissionIndex,
    PseudoLabel, RoundOutcome, RoundRequest, RoundStatus,
};
pub use error::KernelError;
pub use features::{
    guess_audio_media_type, process_spec, put_audio, put_model_card, register_feature,
    RegisterFeatureOpts,
};
pub use ids::{
    is_sentinel_hash, resolve_recording, ArtifactId, AudioAssetId, BeliefId, ClaimId, EvidenceAtom,
    EvidenceId, ModelId, OccurrenceId, RecordingEvidence, RecordingEvidenceKind, RecordingId,
    RoundId, RunId, SetId, SlotId, SnapshotId, WorkId,
};
pub use laws::{check_extensional, Law, LawCheck, LawContext, LawId, LAWS};
pub use provenance::{
    MemoryProvenanceRepository, ProvenanceRepository, SqliteProvenanceRepository,
};
pub use pws::{
    decide, emit_all_sources, estimate_source_accuracies, gate_canaries, majority_vote_fit,
    CanarySpec, DecisionRuleSpec, EvidenceContext, EvidenceEmission, EvidenceSource, GateResult,
    InferenceUnit, SourceAccuracy, VerticalSliceReport,
};
pub use types::{
    implementation_hash, Artifact, ArtifactKind, Axis, AxisBelief, Claim, ClaimContext, Decision,
    Derivation, Evidence, EvidenceDirection, Interval, NativeScore, Observation, ProcessSpec, Run,
    RunInput, RunOutput, RunStatus, SourceLocator, SubjectRef, Unset, UNSET,
};
