//! Audio deposit + feature registration (content-addressed).
//!
//! # TEMP learning tour — STORE helpers used by `make features-demo`
//!
//! # Pipeline
//!
//! ```text
//! put_audio(path) -> Artifact(Audio, sha256)
//!   │
//! analyze.mert (Python sensor) -> feature bytes + model card
//!   │
//! register_feature(parent, bytes, ProcessSpec) -> Artifact(FeatureBlob)
//!   + Run inputs/outputs + Derivation(analyzed_from)
//! ```
//!
//! LFs should reference the feature [`ArtifactId`], never a filesystem path.
//! See `docs/storage.md`. Grep `TEMP:` to strip pedagogy later.

use crate::artifact::ArtifactStore;
use crate::error::KernelError;
use crate::ids::ArtifactId;
use crate::provenance::ProvenanceRepository;
use crate::types::{implementation_hash, Artifact, ArtifactKind, ProcessSpec, Run};
use indexmap::IndexMap;
use std::path::Path;
use uuid::Uuid;

/// Guess a media type from extension; default `application/octet-stream`.
pub fn guess_audio_media_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("mp3") => "audio/mpeg",
        Some("wav") => "audio/wav",
        Some("flac") => "audio/flac",
        Some("ogg") => "audio/ogg",
        Some("m4a") | Some("mp4") => "audio/mp4",
        Some("aiff") | Some("aif") => "audio/aiff",
        _ => "application/octet-stream",
    }
}

/// Build a ProcessSpec with a computed implementation hash.
pub fn process_spec(
    name: &str,
    version: &str,
    stage: &str,
    entrypoint: &str,
    code_commit: &str,
    model_artifact_ids: Vec<ArtifactId>,
) -> ProcessSpec {
    let desc = format!("{name}@{version}:{entrypoint}:{code_commit}");
    ProcessSpec {
        process_spec_id: Uuid::new_v4(),
        name: name.into(),
        version: version.into(),
        stage: stage.into(),
        source_code_artifact_id: ArtifactId::new(),
        code_commit: code_commit.into(),
        entrypoint: entrypoint.into(),
        parameter_set_id: Uuid::new_v4(),
        environment_spec_id: Uuid::new_v4(),
        dependency_lock_artifact_id: ArtifactId::new(),
        model_artifact_ids,
        implementation_hash: implementation_hash(&desc),
    }
}

/// Deposit audio bytes into the store and record a producing run.
pub fn put_audio<S: ArtifactStore, P: ProvenanceRepository>(
    store: &mut S,
    provenance: &mut P,
    path: &Path,
    source_uri: Option<String>,
) -> Result<(Artifact, Run), KernelError> {
    let media = guess_audio_media_type(path);
    let mut meta = IndexMap::new();
    meta.insert(
        "original_filename".into(),
        serde_json::Value::String(
            path.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
        ),
    );

    let spec = process_spec(
        "artifacts.put_audio",
        "0.1.0",
        "ingest",
        "dj_kernel::features::put_audio",
        "local",
        vec![],
    );
    let mut run = provenance.begin_run(&spec, None)?;
    let artifact = store.put_file(
        path,
        ArtifactKind::Audio,
        media,
        meta,
        source_uri.or_else(|| Some(path.to_string_lossy().into_owned())),
    )?;
    provenance.add_output(&run, &artifact, "audio")?;
    provenance.succeed(&mut run)?;
    Ok((artifact, run))
}

/// Register a model card / checkpoint stub so features can pin `model_artifact_ids`.
pub fn put_model_card<S: ArtifactStore, P: ProvenanceRepository>(
    store: &mut S,
    provenance: &mut P,
    model_id: &str,
    model_revision: &str,
    card_json: &[u8],
) -> Result<Artifact, KernelError> {
    let spec = process_spec(
        "artifacts.put_model_card",
        "0.1.0",
        "models",
        "dj_kernel::features::put_model_card",
        "local",
        vec![],
    );
    let mut run = provenance.begin_run(&spec, None)?;
    let mut meta = IndexMap::new();
    meta.insert(
        "model_id".into(),
        serde_json::Value::String(model_id.into()),
    );
    meta.insert(
        "model_revision".into(),
        serde_json::Value::String(model_revision.into()),
    );
    let artifact = store.put_bytes(
        card_json,
        ArtifactKind::ModelCheckpoint,
        "application/json",
        meta,
        Some(format!("model://{model_id}@{model_revision}")),
    )?;
    provenance.add_output(&run, &artifact, "model_checkpoint")?;
    provenance.succeed(&mut run)?;
    Ok(artifact)
}

/// Options for registering a derived feature blob.
pub struct RegisterFeatureOpts<'a> {
    pub parent_audio: &'a Artifact,
    pub feature_bytes: &'a [u8],
    pub feature_kind: &'a str,
    pub media_type: &'a str,
    pub process_name: &'a str,
    pub process_version: &'a str,
    pub entrypoint: &'a str,
    pub model_artifact_ids: Vec<ArtifactId>,
    pub metadata: IndexMap<String, serde_json::Value>,
}

/// Deposit feature bytes, link to parent audio, pin models on the ProcessSpec.
pub fn register_feature<S: ArtifactStore, P: ProvenanceRepository>(
    store: &mut S,
    provenance: &mut P,
    opts: RegisterFeatureOpts<'_>,
) -> Result<(Artifact, Run), KernelError> {
    let spec = process_spec(
        opts.process_name,
        opts.process_version,
        "analysis",
        opts.entrypoint,
        "local",
        opts.model_artifact_ids.clone(),
    );
    let mut run = provenance.begin_run(&spec, None)?;
    provenance.add_input(&run, opts.parent_audio, "audio")?;

    let mut meta = opts.metadata;
    meta.insert(
        "feature_kind".into(),
        serde_json::Value::String(opts.feature_kind.into()),
    );
    meta.insert(
        "parent_audio_artifact_id".into(),
        serde_json::Value::String(opts.parent_audio.artifact_id.to_string()),
    );
    meta.insert(
        "parent_audio_sha256".into(),
        serde_json::Value::String(opts.parent_audio.content_sha256.clone()),
    );

    let feature = store.put_bytes(
        opts.feature_bytes,
        ArtifactKind::FeatureBlob,
        opts.media_type,
        meta,
        None,
    )?;
    provenance.add_output(&run, &feature, opts.feature_kind)?;
    provenance.derive(&feature, opts.parent_audio, &run, "analyzed_from")?;
    provenance.succeed(&mut run)?;
    Ok((feature, run))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::FsArtifactStore;
    use crate::provenance::MemoryProvenanceRepository;
    use tempfile::tempdir;

    #[test]
    fn put_audio_and_register_feature() {
        let dir = tempdir().unwrap();
        let mut store = FsArtifactStore::open(dir.path().join("arts")).unwrap();
        let mut prov = MemoryProvenanceRepository::default();

        let wav = dir.path().join("beep.wav");
        // Minimal RIFF header + silence bytes (not a valid full wav — still content).
        std::fs::write(&wav, b"RIFF....WAVEfmt ").unwrap();

        let (audio, _) = put_audio(&mut store, &mut prov, &wav, None).unwrap();
        assert_eq!(audio.kind, ArtifactKind::Audio);

        let card = br#"{"model_id":"stub-mert","revision":"0"}"#;
        let model = put_model_card(&mut store, &mut prov, "stub-mert", "0", card).unwrap();

        let feat_bytes = b"FAKE_MERT_EMBEDDING_v1";
        let (feat, _) = register_feature(
            &mut store,
            &mut prov,
            RegisterFeatureOpts {
                parent_audio: &audio,
                feature_bytes: feat_bytes,
                feature_kind: "mert_embedding",
                media_type: "application/octet-stream",
                process_name: "analyze.mert",
                process_version: "0.1.0",
                entrypoint: "sensors.mert",
                model_artifact_ids: vec![model.artifact_id],
                metadata: IndexMap::new(),
            },
        )
        .unwrap();
        assert_eq!(feat.kind, ArtifactKind::FeatureBlob);
        assert!(store.verify(feat.artifact_id).unwrap());

        let derivations = prov.all_derivations().unwrap();
        assert!(derivations.iter().any(|d| {
            d.child_artifact_id == feat.artifact_id
                && d.parent_artifact_id == audio.artifact_id
                && d.relation == "analyzed_from"
        }));
    }
}
