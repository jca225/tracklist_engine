//! Stage a map plan into a staging manifest (no bulk audio copy).
//!
//! `apply_ready` is true only when every audio entry has a **non-sentinel**
//! sha256 and a **reachable** path. All-zero placeholder digests from gold
//! fixtures intentionally keep `apply_ready=false`.

use crate::map::MapPlan;
use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagingArtifact {
    pub staging_id: String,
    pub kind: String,
    pub media_type: String,
    pub source_uri: Option<String>,
    pub content_sha256: Option<String>,
    pub byte_size: Option<u64>,
    pub reachable: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagingEdge {
    pub parent_staging_id: String,
    pub child_staging_id: String,
    pub relation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagingManifest {
    pub created_at: String,
    pub dry_run: bool,
    pub map_source: String,
    pub set_ids: Vec<String>,
    pub artifacts: Vec<StagingArtifact>,
    pub edges: Vec<StagingEdge>,
    pub observation_count: usize,
    pub evidence_bundle_count: usize,
    pub human_assertion_count: usize,
    pub apply_ready: bool,
    pub notes: Vec<String>,
}

pub fn run(
    map_path: &Path,
    staging_dir: &Path,
    dry_run: bool,
    write_fixtures: bool,
    fixtures_dir: &Path,
) -> Result<PathBuf> {
    let text =
        fs::read_to_string(map_path).with_context(|| format!("read {}", map_path.display()))?;
    let plan: MapPlan = serde_json::from_str(&text)?;

    fs::create_dir_all(staging_dir)?;
    let mut manifest = StagingManifest {
        created_at: Utc::now().to_rfc3339(),
        dry_run,
        map_source: map_path.display().to_string(),
        set_ids: plan.set_ids.clone(),
        artifacts: vec![],
        edges: vec![],
        observation_count: plan.observations.len(),
        evidence_bundle_count: plan.evidence_bundles.len(),
        human_assertion_count: plan.human_assertions.len(),
        apply_ready: false,
        notes: vec![
            "dry-run staging: no audio bytes copied into artifact store".into(),
            "apply requires --i-know-this-copies-bytes".into(),
        ],
    };

    // Stage map plan itself as a diagnostics artifact (content-addressed description).
    let map_bytes = text.as_bytes();
    let map_sha = hex::encode(Sha256::digest(map_bytes));
    let map_staging_id = format!("map-{map_sha}");
    let map_uri = staging_dir.join("map_plan.json");
    fs::write(&map_uri, map_bytes)?;
    manifest.artifacts.push(StagingArtifact {
        staging_id: map_staging_id.clone(),
        kind: "staging_manifest".into(),
        media_type: "application/json".into(),
        source_uri: Some(map_uri.display().to_string()),
        content_sha256: Some(map_sha),
        byte_size: Some(map_bytes.len() as u64),
        reachable: true,
        notes: vec!["embedded map plan".into()],
    });

    for (i, audio) in plan.audio_staging.iter().enumerate() {
        let staging_id = format!("audio-{i}");
        let mut notes = vec![];
        let mut reachable = false;
        let mut byte_size = None;
        if let Some(path) = &audio.legacy_path {
            let p = Path::new(path);
            if p.exists() {
                reachable = true;
                byte_size = Some(fs::metadata(p)?.len());
            } else {
                notes.push(format!(
                    "path not reachable (expected until transfer): {path}"
                ));
            }
        } else {
            notes.push("no legacy_path".into());
        }
        if audio.sha256.is_none() {
            notes.push("missing sha256 — verify-manifest will fail hard if required".into());
        }
        let art = StagingArtifact {
            staging_id: staging_id.clone(),
            kind: audio.kind.clone(),
            media_type: audio.media_type.clone(),
            source_uri: audio.legacy_path.clone(),
            content_sha256: audio.sha256.clone(),
            byte_size,
            reachable,
            notes,
        };
        manifest.edges.push(StagingEdge {
            parent_staging_id: map_staging_id.clone(),
            child_staging_id: staging_id,
            relation: "planned_from_map".into(),
        });
        manifest.artifacts.push(art);
    }

    // Human assertions / observations as JSON side-cars (tiny, safe to write).
    let obs_path = staging_dir.join("observations_plan.json");
    fs::write(&obs_path, serde_json::to_string_pretty(&plan.observations)?)?;
    let ha_path = staging_dir.join("human_assertions_plan.json");
    fs::write(
        &ha_path,
        serde_json::to_string_pretty(&plan.human_assertions)?,
    )?;
    let ev_path = staging_dir.join("evidence_bundles_plan.json");
    fs::write(
        &ev_path,
        serde_json::to_string_pretty(&plan.evidence_bundles)?,
    )?;

    if write_fixtures {
        fs::create_dir_all(fixtures_dir)?;
        let sample = serde_json::json!({
            "set_ids": plan.set_ids,
            "observation_count": plan.observations.len(),
            "human_assertion_count": plan.human_assertions.len(),
            "note": "tiny non-audio sample for CI; not a byte transfer",
        });
        fs::write(
            fixtures_dir.join("stage_sample.json"),
            serde_json::to_string_pretty(&sample)?,
        )?;
        manifest
            .notes
            .push(format!("wrote tiny sample to {}", fixtures_dir.display()));
    }

    // apply_ready only when every audio entry has a real (non-sentinel) sha256 AND
    // a reachable path. Placeholder digests / remote-only paths are not ready.
    manifest.apply_ready = if plan.audio_staging.is_empty() {
        true
    } else {
        plan.audio_staging.iter().all(|a| {
            let hash_ok = a
                .sha256
                .as_ref()
                .map(|s| !dj_kernel::is_sentinel_hash(s))
                .unwrap_or(false);
            let reachable = a
                .legacy_path
                .as_ref()
                .map(|p| Path::new(p).exists())
                .unwrap_or(false);
            hash_ok && reachable
        })
    };
    if !manifest.apply_ready {
        manifest.notes.push(
            "apply_ready=false: need non-sentinel sha256 and reachable paths for all audio".into(),
        );
    }

    let manifest_path = staging_dir.join("staging_manifest.json");
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    Ok(manifest_path)
}
