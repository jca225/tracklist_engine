//! Shared domain types from the behavioral contract (§1–§5).
//!
//! # TEMP learning tour
//!
//! Think of these as the **nouns** of the engine. Verbs (fit, gate, promote) live
//! in [`crate::pws`] and [`crate::cotraining`].
//!
//! | Type | Plain English |
//! |------|----------------|
//! | [`Artifact`] | Immutable bytes keyed by sha256 (audio, features, HTML, …) |
//! | [`ProcessSpec`] | Recipe that produced a Run (code+params+models pinned) |
//! | [`Run`] | One execution of a ProcessSpec |
//! | [`Derivation`] | Edge: child artifact came from parent via a Run |
//! | [`Observation`] | What a source *said* (not yet belief/truth) |
//! | [`Claim`] | Contestable proposition on an axis |
//! | [`Evidence`] | Claim-scoped support/contradict (older ontology) |
//! | [`AxisBelief`] | Posterior + decision for one InferenceUnit (PWS output) |
//! | [`Axis`] | identity / placement / structure — never collapse to one scalar |
//!
//! Mapping fields use [`IndexMap`] and are treated as frozen by convention —
//! mutate-and-rehash is forbidden (law `immutability.deep`). Prefer depositing
//! large tensors as [`ArtifactKind::FeatureBlob`] rather than embedding them in
//! these records.
//!
//! Grep `TEMP:` across this crate for temporary pedagogy to strip later.

use crate::error::KernelError;
use crate::ids::{ArtifactId, BeliefId, ClaimId, EvidenceId, ModelId, RunId, SnapshotId};
use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// Sentinel distinguishing "no value supplied" from a value of null/None.
/// TEMP: use when a field must say "explicitly unset" vs "null is a real value".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Unset;

pub const UNSET: Unset = Unset;

/// TEMP: the three alignment axes — progress is three curves, not one score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    /// WHICH recording (strongest today).
    Identity,
    /// WHERE in mix-time it starts (hardest).
    Placement,
    /// WHICH internal spans / order (loops, cuts, tempo).
    Structure,
}

/// TEMP: Decide outcome for one unit (Accept may ship; else don't pretend).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Accepted,
    Rejected,
    Unresolved,
    Quarantined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    Quarantined,
}

/// TEMP: what kind of bytes live in the object store (`docs/storage.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    HtmlPage,
    HtmlRow,
    Audio,
    AbletonSession,
    LabelManifest,
    ExternalResponse,
    /// Dense features, fingerprints, beat grids, OD JSON, etc. (see `docs/storage.md`).
    FeatureBlob,
    ModelCheckpoint,
    FittedModel,
    SourceArchive,
    DependencyLock,
    Diagnostics,
    StagingManifest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceDirection {
    Supports,
    Contradicts,
    Neutral,
    Abstains,
}

/// Unknown is None, never 0.0 as a stand-in for unmeasured.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    pub start_s: Option<f64>,
    pub end_s: Option<f64>,
}

impl Interval {
    pub fn new(start_s: Option<f64>, end_s: Option<f64>) -> Result<Self, KernelError> {
        if let (Some(s), Some(e)) = (start_s, end_s) {
            if e < s {
                return Err(KernelError::InvertedInterval);
            }
        }
        Ok(Self { start_s, end_s })
    }
}

/// TEMP: LF/sensor confidence in *its* native scale (not a calibrated probability).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeScore {
    /// e.g. symbolic | signal | vision | agent
    pub family: String,
    pub value: f64,
    /// e.g. count | abstain | od_confidence — how to interpret `value`
    pub scale: String,
    pub support: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocator {
    pub artifact_id: ArtifactId,
    pub selector: String,
    pub ordinal: Option<i64>,
}

/// TEMP: what we are talking about (set-slot, mix-region, …) — not a RecordingId.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubjectRef {
    pub subject_type: String,
    pub subject_id: Uuid,
}

/// Immutable content-addressed object. Identity of *bytes* = sha256, not path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Artifact {
    pub artifact_id: ArtifactId,
    pub kind: ArtifactKind,
    /// Full-object digest — keys the object store.
    pub content_sha256: String,
    pub payload_sha256: Option<String>,
    pub media_type: String,
    pub byte_size: u64,
    pub object_uri: String,
    pub created_at: DateTime<Utc>,
    /// TEMP: human/agent locator only (URL, old path). Never mint RecordingId from this.
    pub source_uri: Option<String>,
    pub source_system: Option<String>,
    pub source_observed_at: Option<DateTime<Utc>>,
    pub metadata: IndexMap<String, serde_json::Value>,
}

/// TEMP: the *recipe* for a Run — pin code, params, env, model cards.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessSpec {
    pub process_spec_id: Uuid,
    pub name: String,
    pub version: String,
    /// ingest | propose | decide | … (stage label for humans)
    pub stage: String,
    pub source_code_artifact_id: ArtifactId,
    pub code_commit: String,
    pub entrypoint: String,
    pub parameter_set_id: Uuid,
    pub environment_spec_id: Uuid,
    pub dependency_lock_artifact_id: ArtifactId,
    pub model_artifact_ids: Vec<ArtifactId>,
    /// Computed — never hand-authored. See [`implementation_hash`].
    pub implementation_hash: String,
}

/// Hash of a stable description of the entrypoint. Callers must pass the
/// transitive description; authors must not invent this string.
pub fn implementation_hash(entrypoint_description: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(entrypoint_description.as_bytes());
    hex::encode(hasher.finalize())
}

/// TEMP: one execution of a ProcessSpec. Provenance edges hang off this.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Run {
    pub run_id: RunId,
    pub process_spec_id: Uuid,
    pub parent_run_id: Option<RunId>,
    pub status: RunStatus,
    pub random_seed: Option<u64>,
    pub environment_instance: IndexMap<String, serde_json::Value>,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub error: Option<IndexMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunInput {
    pub run_id: RunId,
    pub artifact_id: ArtifactId,
    pub role: String,
    pub ordinal: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunOutput {
    pub run_id: RunId,
    pub artifact_id: ArtifactId,
    pub role: String,
    pub ordinal: i64,
}

/// TEMP: lineage edge — e.g. FeatureBlob --analyzed_from--> Audio via this Run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Derivation {
    pub child_artifact_id: ArtifactId,
    pub parent_artifact_id: ArtifactId,
    pub run_id: RunId,
    /// e.g. analyzed_from | partitioned_from
    pub relation: String,
}

/// What a sensor/source observed — not canonical truth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub observation_id: Uuid,
    pub subject: SubjectRef,
    pub predicate: String,
    pub value: Option<serde_json::Value>,
    pub source_artifact_id: ArtifactId,
    pub producer_run_id: RunId,
    pub source_locator: SourceLocator,
    pub observed_at: DateTime<Utc>,
    /// observed | explicit_null | abstained | malformed
    pub status: String,
    pub source_confidence: Option<f64>,
    pub diagnostic_code: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimContext {
    pub supersedes_claim_id: Option<ClaimId>,
    pub mix_region: Option<Interval>,
}

/// Contestable proposition (axis + subject + predicate + object).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Claim {
    pub claim_id: ClaimId,
    pub axis: Axis,
    pub subject: SubjectRef,
    pub predicate: String,
    pub object_entity: Option<SubjectRef>,
    pub object_value: Option<serde_json::Value>,
    pub object_unset: bool,
    pub context: ClaimContext,
    pub created_by_run_id: RunId,
    pub created_at: DateTime<Utc>,
}

impl Claim {
    pub fn new_value(
        axis: Axis,
        subject: SubjectRef,
        predicate: impl Into<String>,
        object_value: Option<serde_json::Value>,
        created_by_run_id: RunId,
    ) -> Result<Self, KernelError> {
        Ok(Self {
            claim_id: ClaimId::new(),
            axis,
            subject,
            predicate: predicate.into(),
            object_entity: None,
            object_value,
            object_unset: false,
            context: ClaimContext {
                supersedes_claim_id: None,
                mix_region: None,
            },
            created_by_run_id,
            created_at: Utc::now(),
        })
    }

    pub fn new_entity(
        axis: Axis,
        subject: SubjectRef,
        predicate: impl Into<String>,
        object_entity: SubjectRef,
        created_by_run_id: RunId,
    ) -> Result<Self, KernelError> {
        Ok(Self {
            claim_id: ClaimId::new(),
            axis,
            subject,
            predicate: predicate.into(),
            object_entity: Some(object_entity),
            object_value: None,
            object_unset: false,
            context: ClaimContext {
                supersedes_claim_id: None,
                mix_region: None,
            },
            created_by_run_id,
            created_at: Utc::now(),
        })
    }

    pub fn validate(&self) -> Result<(), KernelError> {
        // Pseudocode: a claim carries an entity XOR a value (value may be null).
        let entity_given = self.object_entity.is_some();
        if entity_given && self.object_value.is_some() {
            return Err(KernelError::InvalidClaimShape);
        }
        if !entity_given && self.object_unset && self.object_value.is_some() {
            return Err(KernelError::InvalidClaimShape);
        }
        Ok(())
    }
}

/// Claim-scoped evidence. Distinct from PWS `crate::pws::EvidenceEmission`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub evidence_id: EvidenceId,
    pub claim_id: ClaimId,
    pub axis: Axis,
    pub source_type: String,
    pub source_ref_id: Uuid,
    pub producer_run_id: RunId,
    pub direction: EvidenceDirection,
    pub native_score: NativeScore,
    pub source_family: String,
}

/// TEMP: Decide output for one unit — posterior + Accept/Reject/…
/// Built by `crate::pws::majority_vote_fit` today.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisBelief {
    pub belief_id: BeliefId,
    /// Links back to `crate::pws::InferenceUnit::unit_id`.
    pub unit_id: Uuid,
    pub axis: Axis,
    pub inference_run_id: RunId,
    pub fitted_model_id: ModelId,
    /// label → probability (toy majority today).
    pub posterior: IndexMap<String, f64>,
    pub entropy: f64,
    pub decision: Decision,
    pub decision_rule_id: Uuid,
    /// Every emission for this unit (including abstains) — belief provenance.
    pub contributing_emission_ids: Vec<Uuid>,
    pub supersedes_belief_id: Option<BeliefId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotMeta {
    pub snapshot_id: SnapshotId,
    pub kind: String,
    pub parent_snapshot_id: Option<SnapshotId>,
    pub created_by_run_id: RunId,
    pub member_ids: Vec<Uuid>,
    pub snapshot_hash: String,
    pub status: String,
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_rejects_inverted() {
        assert!(Interval::new(Some(2.0), Some(1.0)).is_err());
    }

    #[test]
    fn interval_allows_unknown() {
        let i = Interval::new(None, Some(1.0)).unwrap();
        assert!(i.start_s.is_none());
    }

    #[test]
    fn implementation_hash_is_stable() {
        let a = implementation_hash("entrypoint:parse_session");
        let b = implementation_hash("entrypoint:parse_session");
        assert_eq!(a, b);
        assert_ne!(a, implementation_hash("other"));
    }
}
