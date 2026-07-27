//! CLI: put_audio + register mert FeatureBlob (no live DB).

use anyhow::{bail, Context, Result};
use dj_kernel::{
    put_audio, put_model_card, register_feature, ArtifactStore, FsArtifactStore,
    MemoryProvenanceRepository, RegisterFeatureOpts, SqliteProvenanceRepository,
};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize)]
pub struct PutAudioReport {
    pub artifact_id: String,
    pub content_sha256: String,
    pub kind: String,
    pub media_type: String,
    pub byte_size: u64,
    pub object_uri: String,
    pub run_id: String,
    pub artifact_root: String,
}

#[derive(Debug, Serialize)]
pub struct RegisterFeatureReport {
    pub feature_artifact_id: String,
    pub feature_sha256: String,
    pub parent_audio_artifact_id: String,
    pub model_artifact_id: String,
    pub feature_kind: String,
    pub run_id: String,
    pub process_name: String,
}

/// Sensor output from `sensors.mert` (JSON sidecar).
#[derive(Debug, Deserialize)]
pub struct MertSensorResult {
    pub ok: bool,
    pub feature_kind: String,
    pub feature_path: PathBuf,
    pub media_type: String,
    pub model_id: String,
    pub model_revision: String,
    pub dims: Option<Vec<usize>>,
    pub stub: bool,
    pub content_sha256: Option<String>,
    pub error: Option<String>,
    #[serde(default)]
    pub metadata: IndexMap<String, serde_json::Value>,
}

pub fn audio_put(
    path: &Path,
    artifact_root: &Path,
    provenance_db: Option<&Path>,
) -> Result<PutAudioReport> {
    let mut store = FsArtifactStore::open(artifact_root)?;
    if let Some(db) = provenance_db {
        let mut prov = SqliteProvenanceRepository::open(db)?;
        let (art, run) = put_audio(&mut store, &mut prov, path, None)?;
        Ok(report_put(art, run.run_id.to_string(), artifact_root))
    } else {
        let mut prov = MemoryProvenanceRepository::default();
        let (art, run) = put_audio(&mut store, &mut prov, path, None)?;
        Ok(report_put(art, run.run_id.to_string(), artifact_root))
    }
}

fn report_put(art: dj_kernel::Artifact, run_id: String, root: &Path) -> PutAudioReport {
    PutAudioReport {
        artifact_id: art.artifact_id.to_string(),
        content_sha256: art.content_sha256,
        kind: "audio".into(),
        media_type: art.media_type,
        byte_size: art.byte_size,
        object_uri: art.object_uri,
        run_id,
        artifact_root: root.display().to_string(),
    }
}

pub fn feature_register_mert(
    artifact_root: &Path,
    parent_audio_id: &str,
    mert_json: &Path,
    provenance_db: Option<&Path>,
) -> Result<RegisterFeatureReport> {
    let text =
        fs::read_to_string(mert_json).with_context(|| format!("read {}", mert_json.display()))?;
    let sensor: MertSensorResult = serde_json::from_str(&text)?;
    if !sensor.ok {
        bail!(
            "mert sensor reported failure: {}",
            sensor.error.unwrap_or_else(|| "unknown".into())
        );
    }
    let feature_bytes = fs::read(&sensor.feature_path)
        .with_context(|| format!("read feature {}", sensor.feature_path.display()))?;
    if let Some(expected) = &sensor.content_sha256 {
        use sha2::{Digest, Sha256};
        let actual = hex::encode(Sha256::digest(&feature_bytes));
        if &actual != expected {
            bail!("mert feature hash mismatch: sensor says {expected}, file is {actual}");
        }
    }

    let mut store = FsArtifactStore::open(artifact_root)?;
    let parent = store
        .list()?
        .into_iter()
        .find(|a| a.artifact_id.to_string() == parent_audio_id)
        .with_context(|| format!("parent audio artifact not found: {parent_audio_id}"))?;

    let card = serde_json::json!({
        "model_id": sensor.model_id,
        "model_revision": sensor.model_revision,
        "stub": sensor.stub,
        "dims": sensor.dims,
    });
    let card_bytes = serde_json::to_vec_pretty(&card)?;

    let mut meta = sensor.metadata;
    if let Some(dims) = &sensor.dims {
        meta.insert("dims".into(), serde_json::json!(dims));
    }
    meta.insert("stub".into(), serde_json::Value::Bool(sensor.stub));
    meta.insert(
        "model_id".into(),
        serde_json::Value::String(sensor.model_id.clone()),
    );
    meta.insert(
        "model_revision".into(),
        serde_json::Value::String(sensor.model_revision.clone()),
    );

    if let Some(db) = provenance_db {
        let mut prov = SqliteProvenanceRepository::open(db)?;
        let model = put_model_card(
            &mut store,
            &mut prov,
            &sensor.model_id,
            &sensor.model_revision,
            &card_bytes,
        )?;
        let (feat, run) = register_feature(
            &mut store,
            &mut prov,
            RegisterFeatureOpts {
                parent_audio: &parent,
                feature_bytes: &feature_bytes,
                feature_kind: &sensor.feature_kind,
                media_type: &sensor.media_type,
                process_name: "analyze.mert",
                process_version: "0.1.0",
                entrypoint: "sensors.mert",
                model_artifact_ids: vec![model.artifact_id],
                metadata: meta,
            },
        )?;
        Ok(RegisterFeatureReport {
            feature_artifact_id: feat.artifact_id.to_string(),
            feature_sha256: feat.content_sha256,
            parent_audio_artifact_id: parent.artifact_id.to_string(),
            model_artifact_id: model.artifact_id.to_string(),
            feature_kind: sensor.feature_kind,
            run_id: run.run_id.to_string(),
            process_name: "analyze.mert".into(),
        })
    } else {
        let mut prov = MemoryProvenanceRepository::default();
        let model = put_model_card(
            &mut store,
            &mut prov,
            &sensor.model_id,
            &sensor.model_revision,
            &card_bytes,
        )?;
        let (feat, run) = register_feature(
            &mut store,
            &mut prov,
            RegisterFeatureOpts {
                parent_audio: &parent,
                feature_bytes: &feature_bytes,
                feature_kind: &sensor.feature_kind,
                media_type: &sensor.media_type,
                process_name: "analyze.mert",
                process_version: "0.1.0",
                entrypoint: "sensors.mert",
                model_artifact_ids: vec![model.artifact_id],
                metadata: meta,
            },
        )?;
        Ok(RegisterFeatureReport {
            feature_artifact_id: feat.artifact_id.to_string(),
            feature_sha256: feat.content_sha256,
            parent_audio_artifact_id: parent.artifact_id.to_string(),
            model_artifact_id: model.artifact_id.to_string(),
            feature_kind: sensor.feature_kind,
            run_id: run.run_id.to_string(),
            process_name: "analyze.mert".into(),
        })
    }
}
