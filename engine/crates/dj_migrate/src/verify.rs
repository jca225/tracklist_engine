use crate::stage::StagingManifest;
use anyhow::{bail, Context, Result};
use dj_kernel::laws::graph_is_acyclic;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::Path;

pub fn run(manifest_path: &Path, require_reachable: bool) -> Result<()> {
    let text = fs::read_to_string(manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest: StagingManifest = serde_json::from_str(&text)?;

    let edges: Vec<(String, String)> = manifest
        .edges
        .iter()
        .map(|e| (e.parent_staging_id.clone(), e.child_staging_id.clone()))
        .collect();
    if !graph_is_acyclic(&edges) {
        bail!("staging derivation graph has a cycle");
    }

    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    for art in &manifest.artifacts {
        if art.kind == "audio" && art.content_sha256.is_none() {
            errors.push(format!(
                "{}: audio artifact missing sha256 (path-as-identity forbidden)",
                art.staging_id
            ));
        }
        if let Some(uri) = &art.source_uri {
            let p = Path::new(uri);
            if p.exists() {
                if let Some(expected) = &art.content_sha256 {
                    let mut f = fs::File::open(p)?;
                    let mut buf = Vec::new();
                    f.read_to_end(&mut buf)?;
                    let actual = hex::encode(Sha256::digest(&buf));
                    if &actual != expected {
                        errors.push(format!(
                            "{}: hash mismatch expected={expected} actual={actual}",
                            art.staging_id
                        ));
                    }
                }
            } else if require_reachable {
                errors.push(format!("{}: unreachable {}", art.staging_id, uri));
            } else {
                warnings.push(format!("{}: unreachable {}", art.staging_id, uri));
            }
        }
    }

    for w in &warnings {
        eprintln!("warn: {w}");
    }
    if !errors.is_empty() {
        for e in &errors {
            eprintln!("error: {e}");
        }
        bail!("{} verify error(s)", errors.len());
    }

    println!(
        "verify-manifest ok: {} artifacts, apply_ready={}",
        manifest.artifacts.len(),
        manifest.apply_ready
    );
    Ok(())
}
