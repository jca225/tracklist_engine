//! Gated byte copy into the kernel artifact store.
//!
//! Intentionally requires `--i-know-this-copies-bytes`. When enabled:
//! - refuses sentinel sha256s and `apply_ready=false` manifests
//! - deposits audio via [`FsArtifactStore`] (content-addressed)
//! - records a provenance run so law `provenance.complete` can pass
//!
//! Do not point this at the live pi object tree until transfer is deliberate.
//! See `docs/storage.md`.

use crate::stage::StagingManifest;
use anyhow::{bail, Context, Result};
use dj_kernel::{
    implementation_hash, ArtifactKind, ArtifactStore, FsArtifactStore, MemoryProvenanceRepository,
    ProcessSpec, ProvenanceRepository,
};
use indexmap::IndexMap;
use std::fs;
use std::path::Path;
use uuid::Uuid;

pub fn run(manifest_path: &Path, artifact_root: &Path, i_know: bool) -> Result<()> {
    if !i_know {
        bail!(
            "refusing to copy bytes. Re-run with --i-know-this-copies-bytes when you are ready to transfer."
        );
    }

    let text = fs::read_to_string(manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest: StagingManifest = serde_json::from_str(&text)?;

    if !manifest.apply_ready {
        bail!("manifest.apply_ready is false; fix missing/sentinel sha256 and reachability first");
    }

    let mut store = FsArtifactStore::open(artifact_root)?;
    let mut provenance = MemoryProvenanceRepository::default();
    let spec = ProcessSpec {
        process_spec_id: Uuid::new_v4(),
        name: "dj_migrate.apply".into(),
        version: "0.1.0".into(),
        stage: "migrate".into(),
        source_code_artifact_id: dj_kernel::ArtifactId::new(),
        code_commit: "local".into(),
        entrypoint: "dj_migrate::apply".into(),
        parameter_set_id: Uuid::new_v4(),
        environment_spec_id: Uuid::new_v4(),
        dependency_lock_artifact_id: dj_kernel::ArtifactId::new(),
        model_artifact_ids: vec![],
        implementation_hash: implementation_hash("dj_migrate::apply"),
    };
    let mut run = provenance.begin_run(&spec, None)?;
    let mut copied = 0usize;

    for art in &manifest.artifacts {
        if art.kind != "audio" {
            continue;
        }
        let Some(uri) = &art.source_uri else {
            bail!("{} missing source_uri", art.staging_id);
        };
        let path = Path::new(uri);
        if !path.exists() {
            bail!("{} source not reachable: {uri}", art.staging_id);
        }
        let stored = store.put_file(
            path,
            ArtifactKind::Audio,
            &art.media_type,
            IndexMap::new(),
            Some(uri.clone()),
        )?;
        if let Some(expected) = &art.content_sha256 {
            if dj_kernel::is_sentinel_hash(expected) {
                bail!("{} sentinel sha256 refused", art.staging_id);
            }
            if &stored.content_sha256 != expected {
                bail!(
                    "{} hash mismatch after copy: expected {expected}, got {}",
                    art.staging_id,
                    stored.content_sha256
                );
            }
        }
        provenance.add_output(&run, &stored, "audio")?;
        copied += 1;
        eprintln!("copied {} -> artifact {}", uri, stored.artifact_id);
    }

    provenance.succeed(&mut run)?;
    println!(
        "apply complete: copied {copied} audio artifact(s) into {} (run {})",
        artifact_root.display(),
        run.run_id
    );
    Ok(())
}
