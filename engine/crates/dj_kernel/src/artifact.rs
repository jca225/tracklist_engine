//! Content-addressed artifact store.
//!
//! # TEMP learning tour — WHERE BYTES LIVE
//!
//! ```text
//! put_bytes / put_file
//!   → objects/{aa}/{sha256}   (payload)
//!   → meta/{artifact_id}.json (Artifact record)
//! ```
//!
//! # Storage policy
//!
//! - **Audio and features are bytes under sha256**, not paths keyed by
//!   legacy `track_id`. See `docs/storage.md`.
//! - `put_*` is idempotent on content: the same bytes reuse the object file and
//!   mint a new `Artifact` meta record (one logical deposit per call).
//! - `source_uri` is a locator for humans/agents (URL or old path); it must never
//!   be treated as musical identity ([`crate::resolve_recording`]).
//! - Embeddings and MIR outputs use [`ArtifactKind::FeatureBlob`] and should be
//!   linked with a provenance `derive(..., "analyzed_from")` from the parent audio.
//!
//! Grep `TEMP:` to strip pedagogy later.

use crate::error::KernelError;
use crate::ids::ArtifactId;
use crate::types::{Artifact, ArtifactKind};
use chrono::Utc;
use indexmap::IndexMap;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Durable byte store for source media and derived feature blobs.
/// TEMP: FsArtifactStore is the default local impl (staging/artifacts in demos).
pub trait ArtifactStore {
    /// Deposit raw bytes; identity is `sha256(content)`.
    fn put_bytes(
        &mut self,
        content: &[u8],
        kind: ArtifactKind,
        media_type: &str,
        metadata: IndexMap<String, serde_json::Value>,
        source_uri: Option<String>,
    ) -> Result<Artifact, KernelError>;

    /// Read a file from disk and deposit it (path is recorded as locator only).
    fn put_file(
        &mut self,
        path: &Path,
        kind: ArtifactKind,
        media_type: &str,
        metadata: IndexMap<String, serde_json::Value>,
        source_uri: Option<String>,
    ) -> Result<Artifact, KernelError>;

    fn open(&self, artifact_id: ArtifactId) -> Result<Vec<u8>, KernelError>;

    fn get(&self, artifact_id: ArtifactId) -> Result<Artifact, KernelError>;

    /// Re-hash object bytes and compare to the stored digest (law `artifacts.verify`).
    fn verify(&self, artifact_id: ArtifactId) -> Result<bool, KernelError>;

    fn list(&self) -> Result<Vec<Artifact>, KernelError>;
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Local filesystem content-addressed store.
///
/// Layout:
/// - `{root}/objects/{sha256[0..2]}/{sha256}` — payload bytes
/// - `{root}/meta/{artifact_id}.json` — [`Artifact`] sidecar
///
/// Safe to mirror `objects/` to object storage (S3/R2) using the same keys.
pub struct FsArtifactStore {
    root: PathBuf,
}

impl FsArtifactStore {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, KernelError> {
        let root = root.into();
        fs::create_dir_all(root.join("objects"))?;
        fs::create_dir_all(root.join("meta"))?;
        Ok(Self { root })
    }

    fn object_path(&self, sha: &str) -> PathBuf {
        self.root.join("objects").join(&sha[..2]).join(sha)
    }

    fn meta_path(&self, id: ArtifactId) -> PathBuf {
        self.root.join("meta").join(format!("{id}.json"))
    }

    fn write_meta(&self, artifact: &Artifact) -> Result<(), KernelError> {
        let path = self.meta_path(artifact.artifact_id);
        let f = File::create(path)?;
        serde_json::to_writer_pretty(f, artifact)?;
        Ok(())
    }

    fn read_meta(&self, id: ArtifactId) -> Result<Artifact, KernelError> {
        let path = self.meta_path(id);
        if !path.exists() {
            return Err(KernelError::ArtifactNotFound(id.to_string()));
        }
        let f = File::open(path)?;
        Ok(serde_json::from_reader(f)?)
    }
}

impl ArtifactStore for FsArtifactStore {
    fn put_bytes(
        &mut self,
        content: &[u8],
        kind: ArtifactKind,
        media_type: &str,
        metadata: IndexMap<String, serde_json::Value>,
        source_uri: Option<String>,
    ) -> Result<Artifact, KernelError> {
        let content_sha256 = sha256_hex(content);
        let object_path = self.object_path(&content_sha256);
        if let Some(parent) = object_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if !object_path.exists() {
            let mut f = File::create(&object_path)?;
            f.write_all(content)?;
        }
        let artifact = Artifact {
            artifact_id: ArtifactId::new(),
            kind,
            content_sha256: content_sha256.clone(),
            payload_sha256: None,
            media_type: media_type.to_string(),
            byte_size: content.len() as u64,
            object_uri: object_path.to_string_lossy().into_owned(),
            created_at: Utc::now(),
            source_uri,
            source_system: None,
            source_observed_at: None,
            metadata,
        };
        self.write_meta(&artifact)?;
        Ok(artifact)
    }

    fn put_file(
        &mut self,
        path: &Path,
        kind: ArtifactKind,
        media_type: &str,
        metadata: IndexMap<String, serde_json::Value>,
        source_uri: Option<String>,
    ) -> Result<Artifact, KernelError> {
        let mut f = File::open(path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        let source = source_uri.or_else(|| Some(path.to_string_lossy().into_owned()));
        self.put_bytes(&buf, kind, media_type, metadata, source)
    }

    fn open(&self, artifact_id: ArtifactId) -> Result<Vec<u8>, KernelError> {
        let meta = self.read_meta(artifact_id)?;
        let mut f = File::open(&meta.object_uri)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        Ok(buf)
    }

    fn get(&self, artifact_id: ArtifactId) -> Result<Artifact, KernelError> {
        self.read_meta(artifact_id)
    }

    fn verify(&self, artifact_id: ArtifactId) -> Result<bool, KernelError> {
        let meta = self.read_meta(artifact_id)?;
        let bytes = self.open(artifact_id)?;
        let actual = sha256_hex(&bytes);
        if actual != meta.content_sha256 {
            return Err(KernelError::HashMismatch {
                expected: meta.content_sha256,
                actual,
            });
        }
        Ok(true)
    }

    fn list(&self) -> Result<Vec<Artifact>, KernelError> {
        let meta_dir = self.root.join("meta");
        let mut out = Vec::new();
        if !meta_dir.exists() {
            return Ok(out);
        }
        for entry in fs::read_dir(meta_dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let f = File::open(entry.path())?;
            out.push(serde_json::from_reader(f)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn put_and_verify_roundtrip() {
        let dir = tempdir().unwrap();
        let mut store = FsArtifactStore::open(dir.path()).unwrap();
        let art = store
            .put_bytes(
                b"hello",
                ArtifactKind::Diagnostics,
                "text/plain",
                IndexMap::new(),
                None,
            )
            .unwrap();
        assert!(store.verify(art.artifact_id).unwrap());
        assert_eq!(store.open(art.artifact_id).unwrap(), b"hello");
    }
}
