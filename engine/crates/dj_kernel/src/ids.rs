//! Opaque typed IDs.
//!
//! # TEMP learning tour
//!
//! Most IDs are just newtype UUIDs (`ArtifactId`, `RunId`, …) so you cannot
//! accidentally pass a set_id where a run_id belongs.
//!
//! **`RecordingId` is special:** it has no `From<String>` / `From<&str>` and is
//! **not** `Deserialize` — musical identity cannot be smuggled in via JSON or a
//! path. The only mint path is [`resolve_recording`]. Persist with
//! [`RecordingId::from_persisted`] only when loading a previously minted id.
//!
//! Content hashes used as evidence must be real digests ([`is_sentinel_hash`]
//! rejects empty, wrong-length, and all-zero placeholders used in gold fixtures).
//!
//! Grep `TEMP:` to strip pedagogy later.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// TEMP: boilerplate newtype UUID wrappers — prevents ID mix-ups at compile time.
macro_rules! opaque_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(Uuid);

        impl $name {
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }

            pub fn from_uuid(id: Uuid) -> Self {
                Self(id)
            }

            pub fn as_uuid(&self) -> Uuid {
                self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

opaque_id!(ArtifactId);
opaque_id!(RunId);
opaque_id!(SetId);
opaque_id!(SlotId);
opaque_id!(WorkId);
opaque_id!(AudioAssetId);
opaque_id!(OccurrenceId);
opaque_id!(ClaimId);
opaque_id!(EvidenceId);
opaque_id!(BeliefId);
opaque_id!(SnapshotId);
opaque_id!(ModelId);
opaque_id!(RoundId);

/// Canonical recording identity. Minted only by [`resolve_recording`].
/// TEMP: NOT Deserialize — JSON cannot invent identity; paths never become this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct RecordingId(Uuid);

impl RecordingId {
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }

    /// Load a previously minted id from durable storage. Not a mint path.
    pub fn from_persisted(id: Uuid) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for RecordingId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Kind of atom that may support minting a [`RecordingId`].
/// TEMP: only ContentHash / HumanAbleton / AudioFingerprint are *substantive*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordingEvidenceKind {
    /// Content-addressed audio / clip bytes.
    ContentHash,
    /// Human Ableton assertion bound to content.
    HumanAbleton,
    /// Acoustic fingerprint match.
    AudioFingerprint,
    /// Legacy DB row — locator only, never sufficient alone.
    LegacyRecordingRow,
    /// Slot claim text — locator only, never sufficient alone.
    SlotClaim,
}

impl RecordingEvidenceKind {
    pub fn is_substantive(self) -> bool {
        matches!(
            self,
            Self::ContentHash | Self::HumanAbleton | Self::AudioFingerprint
        )
    }
}

/// One atom of evidence for identity minting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceAtom {
    pub evidence_id: EvidenceId,
    pub kind: RecordingEvidenceKind,
    /// Required for substantive kinds: content sha256 or artifact URI.
    pub content_ref: Option<String>,
}

impl EvidenceAtom {
    pub fn validate(&self) -> Result<(), crate::KernelError> {
        if self.kind.is_substantive() {
            match &self.content_ref {
                Some(r) if !r.is_empty() && !is_sentinel_hash(r) => Ok(()),
                _ => Err(crate::KernelError::IdentityWithoutEvidence),
            }
        } else {
            Ok(())
        }
    }
}

/// Evidence required to mint a [`RecordingId`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingEvidence {
    pub atoms: Vec<EvidenceAtom>,
    pub note: Option<String>,
}

impl RecordingEvidence {
    pub fn from_atoms(
        atoms: impl IntoIterator<Item = EvidenceAtom>,
    ) -> Result<Self, crate::KernelError> {
        let atoms: Vec<_> = atoms.into_iter().collect();
        if atoms.is_empty() {
            return Err(crate::KernelError::IdentityWithoutEvidence);
        }
        for a in &atoms {
            a.validate()?;
        }
        Ok(Self { atoms, note: None })
    }
}

/// True for empty, wrong-length, non-hex, or all-zero placeholder digests.
pub fn is_sentinel_hash(s: &str) -> bool {
    let t = s.trim();
    if t.len() != 64 {
        return true;
    }
    if !t.chars().all(|c| c.is_ascii_hexdigit()) {
        return true;
    }
    t.chars().all(|c| c == '0')
}

/// The only way a [`RecordingId`] comes into existence.
///
/// Requires at least one *substantive* atom (content hash, human Ableton, or
/// fingerprint) with a non-sentinel `content_ref`. Locator-only atoms never mint.
///
/// TEMP: when debugging "why isn't identity resolving?", check atoms + sentinel hashes first.
pub fn resolve_recording(evidence: &RecordingEvidence) -> Result<RecordingId, crate::KernelError> {
    let mut ok = false;
    for atom in &evidence.atoms {
        atom.validate()?;
        if atom.kind.is_substantive() {
            ok = true;
        }
    }
    if !ok {
        return Err(crate::KernelError::IdentityWithoutEvidence);
    }
    Ok(RecordingId(Uuid::new_v4()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recording_id_rejects_empty() {
        let empty = RecordingEvidence {
            atoms: vec![],
            note: None,
        };
        assert!(matches!(
            resolve_recording(&empty),
            Err(crate::KernelError::IdentityWithoutEvidence)
        ));
    }

    #[test]
    fn recording_id_rejects_locator_only() {
        let ev = RecordingEvidence::from_atoms([EvidenceAtom {
            evidence_id: EvidenceId::new(),
            kind: RecordingEvidenceKind::LegacyRecordingRow,
            content_ref: Some("abc".into()),
        }])
        .unwrap();
        assert!(matches!(
            resolve_recording(&ev),
            Err(crate::KernelError::IdentityWithoutEvidence)
        ));
    }

    #[test]
    fn recording_id_rejects_sentinel_content_hash() {
        let err = RecordingEvidence::from_atoms([EvidenceAtom {
            evidence_id: EvidenceId::new(),
            kind: RecordingEvidenceKind::ContentHash,
            content_ref: Some("0".repeat(64)),
        }]);
        assert!(err.is_err());
    }

    #[test]
    fn recording_id_mints_with_substantive_evidence() {
        let ev = RecordingEvidence::from_atoms([
            EvidenceAtom {
                evidence_id: EvidenceId::new(),
                kind: RecordingEvidenceKind::LegacyRecordingRow,
                content_ref: None,
            },
            EvidenceAtom {
                evidence_id: EvidenceId::new(),
                kind: RecordingEvidenceKind::ContentHash,
                content_ref: Some("a".repeat(64)),
            },
        ])
        .unwrap();
        let id = resolve_recording(&ev).unwrap();
        assert_ne!(id.as_uuid(), Uuid::nil());
    }

    #[test]
    fn sentinel_detection() {
        assert!(is_sentinel_hash(&"0".repeat(64)));
        assert!(is_sentinel_hash("abc"));
        assert!(!is_sentinel_hash(&"ab".repeat(32)));
    }
}
