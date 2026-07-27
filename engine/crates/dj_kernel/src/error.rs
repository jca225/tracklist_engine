//! Kernel error types.
//!
//! TEMP: IdentityWithoutEvidence = tried to mint RecordingId without substantive atoms.
//! ProvenanceCycle = round tried to consume its own descendants.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("identity cannot be minted without evidence (law identity.no_locator)")]
    IdentityWithoutEvidence,

    #[error("interval is inverted: end_s < start_s")]
    InvertedInterval,

    #[error("claim must carry exactly one of object_entity or object_value")]
    InvalidClaimShape,

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("artifact not found: {0}")]
    ArtifactNotFound(String),

    #[error("artifact hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },

    #[error("run not found: {0}")]
    RunNotFound(String),

    #[error("law failed: {law_id}: {statement}")]
    LawFailed { law_id: String, statement: String },

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("provenance cycle: round cannot consume its own descendants")]
    ProvenanceCycle,

    #[error("view leakage: pseudo-label returned to its originating source family ({0})")]
    ViewLeakage(String),

    #[error("unfalsifiable pseudo-label: no independent evidential root")]
    UnfalsifiablePseudoLabel,

    #[error("{0}")]
    Other(String),
}
