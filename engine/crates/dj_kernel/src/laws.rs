//! Law registry (§16) — subset used as promotion / verify gates.
//!
//! # TEMP learning tour — FAIL-CLOSED CHECKS
//!
//! | LawId | Plain English |
//! |-------|----------------|
//! | ProvenanceComplete | every artifact has a producing Run |
//! | ProvenanceAcyclic | derivation graph has no cycles |
//! | ArtifactsVerify | bytes still match stored sha256 |
//! | AbstentionsPersisted | don't silently drop abstains |
//! | IdentityNoLocator | paths can't mint RecordingId (type system) |
//!
//! Extensional laws run against a [`LawContext`] (snapshot-shaped witness).
//! Intensional laws (e.g. `identity.no_locator`) are discharged at construction
//! sites in [`crate::ids`] and the type system — not here.
//!
//! Grep `TEMP:` to strip pedagogy later.

use crate::artifact::ArtifactStore;
use crate::error::KernelError;
use crate::provenance::ProvenanceRepository;
use crate::types::Observation;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LawId {
    ProvenanceComplete,
    ProvenanceAcyclic,
    ArtifactsVerify,
    AbstentionsPersisted,
    IdentityNoLocator,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Law {
    pub law_id: LawId,
    pub statement: &'static str,
    pub enforcement: &'static str,
    pub tags: &'static [&'static str],
}

pub const LAWS: &[Law] = &[
    Law {
        law_id: LawId::ProvenanceComplete,
        statement: "every artifact has a producing run",
        enforcement: "extensional",
        tags: &["promotion"],
    },
    Law {
        law_id: LawId::ProvenanceAcyclic,
        statement: "the derivation graph has no cycles",
        enforcement: "extensional",
        tags: &["promotion"],
    },
    Law {
        law_id: LawId::ArtifactsVerify,
        statement: "every referenced artifact exists and rehashes",
        enforcement: "extensional",
        tags: &["promotion"],
    },
    Law {
        law_id: LawId::AbstentionsPersisted,
        statement: "abstentions are stored, not dropped",
        enforcement: "extensional",
        tags: &["promotion"],
    },
    Law {
        law_id: LawId::IdentityNoLocator,
        statement: "no path, source key, or locator may construct an identity value",
        enforcement: "intensional",
        tags: &["promotion"],
    },
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LawCheck {
    pub law_id: LawId,
    pub passed: bool,
    pub detail: String,
}

pub struct LawContext<'a> {
    pub artifacts: &'a dyn ArtifactStore,
    pub provenance: &'a dyn ProvenanceRepository,
    pub observations: &'a [Observation],
    /// Count of abstentions known to have been dropped (must be 0).
    pub dropped_abstention_count: u64,
}

/// TEMP: run all extensional laws; used as a promotion witness.
pub fn check_extensional(ctx: &LawContext<'_>) -> Result<Vec<LawCheck>, KernelError> {
    let mut results = Vec::new();
    for law in LAWS.iter().filter(|l| l.enforcement == "extensional") {
        let check = match law.law_id {
            LawId::ProvenanceComplete => check_provenance_complete(ctx)?,
            LawId::ProvenanceAcyclic => check_provenance_acyclic(ctx)?,
            LawId::ArtifactsVerify => check_artifacts_verify(ctx)?,
            LawId::AbstentionsPersisted => check_abstentions_persisted(ctx)?,
            LawId::IdentityNoLocator => unreachable!("intensional"),
        };
        results.push(check);
    }
    Ok(results)
}

fn check_provenance_complete(ctx: &LawContext<'_>) -> Result<LawCheck, KernelError> {
    let artifacts = ctx.artifacts.list()?;
    let outputs = ctx.provenance.all_outputs()?;
    let produced: HashSet<_> = outputs.iter().map(|o| o.artifact_id).collect();
    let mut missing = Vec::new();
    for a in &artifacts {
        if !produced.contains(&a.artifact_id) {
            missing.push(a.artifact_id.to_string());
        }
    }
    // Allow empty store.
    let passed = missing.is_empty();
    Ok(LawCheck {
        law_id: LawId::ProvenanceComplete,
        passed,
        detail: if passed {
            "all artifacts have producing runs".into()
        } else {
            format!("artifacts without producing run: {missing:?}")
        },
    })
}

fn check_provenance_acyclic(ctx: &LawContext<'_>) -> Result<LawCheck, KernelError> {
    let derivations = ctx.provenance.all_derivations()?;
    let mut adj: HashMap<_, Vec<_>> = HashMap::new();
    for d in &derivations {
        adj.entry(d.parent_artifact_id.as_uuid())
            .or_default()
            .push(d.child_artifact_id.as_uuid());
    }
    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();

    fn dfs(
        node: uuid::Uuid,
        adj: &HashMap<uuid::Uuid, Vec<uuid::Uuid>>,
        visiting: &mut HashSet<uuid::Uuid>,
        visited: &mut HashSet<uuid::Uuid>,
    ) -> bool {
        if visited.contains(&node) {
            return false;
        }
        if !visiting.insert(node) {
            return true;
        }
        if let Some(children) = adj.get(&node) {
            for &c in children {
                if dfs(c, adj, visiting, visited) {
                    return true;
                }
            }
        }
        visiting.remove(&node);
        visited.insert(node);
        false
    }

    let nodes: HashSet<_> = adj
        .keys()
        .copied()
        .chain(adj.values().flatten().copied())
        .collect();
    for node in nodes {
        if dfs(node, &adj, &mut visiting, &mut visited) {
            return Ok(LawCheck {
                law_id: LawId::ProvenanceAcyclic,
                passed: false,
                detail: "derivation graph contains a cycle".into(),
            });
        }
    }
    Ok(LawCheck {
        law_id: LawId::ProvenanceAcyclic,
        passed: true,
        detail: "derivation graph is acyclic".into(),
    })
}

fn check_artifacts_verify(ctx: &LawContext<'_>) -> Result<LawCheck, KernelError> {
    let artifacts = ctx.artifacts.list()?;
    for a in &artifacts {
        match ctx.artifacts.verify(a.artifact_id) {
            Ok(true) => {}
            Ok(false) => {
                return Ok(LawCheck {
                    law_id: LawId::ArtifactsVerify,
                    passed: false,
                    detail: format!("verify returned false for {}", a.artifact_id),
                });
            }
            Err(e) => {
                return Ok(LawCheck {
                    law_id: LawId::ArtifactsVerify,
                    passed: false,
                    detail: format!("{}: {e}", a.artifact_id),
                });
            }
        }
    }
    Ok(LawCheck {
        law_id: LawId::ArtifactsVerify,
        passed: true,
        detail: format!("{} artifacts verified", artifacts.len()),
    })
}

fn check_abstentions_persisted(ctx: &LawContext<'_>) -> Result<LawCheck, KernelError> {
    let abstentions: Vec<_> = ctx
        .observations
        .iter()
        .filter(|o| o.status == "abstained" || o.status == "malformed")
        .collect();
    let passed = ctx.dropped_abstention_count == 0;
    Ok(LawCheck {
        law_id: LawId::AbstentionsPersisted,
        passed,
        detail: if passed {
            format!(
                "{} abstention/malformed observations persisted; none dropped",
                abstentions.len()
            )
        } else {
            format!(
                "dropped_abstention_count={} (must be 0); persisted={}",
                ctx.dropped_abstention_count,
                abstentions.len()
            )
        },
    })
}

/// Detect cycles via Kahn for staging manifests that encode edges as pairs.
pub fn graph_is_acyclic(edges: &[(String, String)]) -> bool {
    let mut indeg: HashMap<&str, usize> = HashMap::new();
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for (parent, child) in edges {
        adj.entry(parent.as_str()).or_default().push(child.as_str());
        indeg.entry(child.as_str()).or_default();
        *indeg.entry(child.as_str()).or_default() += 1;
        indeg.entry(parent.as_str()).or_default();
    }
    let mut q: VecDeque<&str> = indeg
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(k, _)| *k)
        .collect();
    let mut seen = 0;
    while let Some(n) = q.pop_front() {
        seen += 1;
        if let Some(children) = adj.get(n) {
            for &c in children {
                let e = indeg.get_mut(c).unwrap();
                *e -= 1;
                if *e == 0 {
                    q.push_back(c);
                }
            }
        }
    }
    seen == indeg.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::FsArtifactStore;
    use crate::provenance::MemoryProvenanceRepository;
    use crate::types::{implementation_hash, ArtifactKind, ProcessSpec};
    use indexmap::IndexMap;
    use tempfile::tempdir;
    use uuid::Uuid;

    #[test]
    fn empty_store_passes_verify_and_acyclic() {
        let dir = tempdir().unwrap();
        let store = FsArtifactStore::open(dir.path()).unwrap();
        let repo = MemoryProvenanceRepository::default();
        let ctx = LawContext {
            artifacts: &store,
            provenance: &repo,
            observations: &[],
            dropped_abstention_count: 0,
        };
        let checks = check_extensional(&ctx).unwrap();
        assert!(checks.iter().all(|c| c.passed));
    }

    #[test]
    fn dropped_abstentions_fail_law() {
        let dir = tempdir().unwrap();
        let store = FsArtifactStore::open(dir.path()).unwrap();
        let repo = MemoryProvenanceRepository::default();
        let ctx = LawContext {
            artifacts: &store,
            provenance: &repo,
            observations: &[],
            dropped_abstention_count: 1,
        };
        let checks = check_extensional(&ctx).unwrap();
        let a = checks
            .iter()
            .find(|c| c.law_id == LawId::AbstentionsPersisted)
            .unwrap();
        assert!(!a.passed);
    }

    #[test]
    fn artifact_with_producing_run_passes_complete() {
        let dir = tempdir().unwrap();
        let mut store = FsArtifactStore::open(dir.path()).unwrap();
        let mut repo = MemoryProvenanceRepository::default();
        let spec = ProcessSpec {
            process_spec_id: Uuid::new_v4(),
            name: "t".into(),
            version: "1".into(),
            stage: "t".into(),
            source_code_artifact_id: crate::ids::ArtifactId::new(),
            code_commit: "x".into(),
            entrypoint: "t".into(),
            parameter_set_id: Uuid::new_v4(),
            environment_spec_id: Uuid::new_v4(),
            dependency_lock_artifact_id: crate::ids::ArtifactId::new(),
            model_artifact_ids: vec![],
            implementation_hash: implementation_hash("t"),
        };
        let mut run = repo.begin_run(&spec, None).unwrap();
        let art = store
            .put_bytes(
                b"x",
                ArtifactKind::Diagnostics,
                "text/plain",
                IndexMap::new(),
                None,
            )
            .unwrap();
        repo.add_output(&run, &art, "out").unwrap();
        repo.succeed(&mut run).unwrap();
        let ctx = LawContext {
            artifacts: &store,
            provenance: &repo,
            observations: &[],
            dropped_abstention_count: 0,
        };
        let checks = check_extensional(&ctx).unwrap();
        let complete = checks
            .iter()
            .find(|c| c.law_id == LawId::ProvenanceComplete)
            .unwrap();
        assert!(complete.passed, "{}", complete.detail);
    }
}
