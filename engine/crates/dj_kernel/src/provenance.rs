//! Execution provenance repository.
//!
//! # TEMP learning tour — THE RECEIPT GRAPH
//!
//! Provenance answers: "what bytes + what recipe produced this?"
//!
//! ```text
//! begin_run(ProcessSpec)
//!   add_input(audio) / add_output(feature)
//!   derive(child, parent, "analyzed_from")
//! succeed / fail / quarantine
//! closure(node) → transitive dependency set
//! ```
//!
//! Every artifact deposit that matters for promotion should be an output of a
//! [`Run`] begun from a [`ProcessSpec`]. Migrations that `put_file` without a
//! run violate law `provenance.complete` — see `dj_migrate::apply`, which wires
//! begin_run / add_output / succeed around copies.
//!
//! Implementations: [`MemoryProvenanceRepository`] (tests) and
//! [`SqliteProvenanceRepository`] (durable). Grep `TEMP:` to strip later.

use crate::error::KernelError;
use crate::ids::{ArtifactId, RunId};
use crate::types::{Artifact, Derivation, ProcessSpec, Run, RunInput, RunOutput, RunStatus};
use chrono::Utc;
use indexmap::IndexMap;
use rusqlite::{params, Connection};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use uuid::Uuid;

/// TEMP: the provenance API — durable *receipts* for every execution that
/// produces or consumes artifacts.
///
/// # Mental model
///
/// A [`ProcessSpec`] is the **recipe** (code + params + models).
/// A [`Run`] is **one cooking** of that recipe.
/// [`add_input`] / [`add_output`] attach **which artifacts** were used/made.
/// [`derive`] draws an **edge** (child came from parent under this Run).
/// [`succeed`] / [`fail`] / [`quarantine`] close the Run.
///
/// Typical happy path (also used by `features::put_audio` / `register_feature`):
///
/// ```text
/// let mut run = repo.begin_run(&spec, None)?;
/// repo.add_input(&run, &audio, "audio")?;
/// // … sensor work …
/// repo.add_output(&run, &feature, "feature")?;
/// repo.derive(&feature, &audio, &run, "analyzed_from")?;
/// repo.succeed(&mut run)?;
/// ```
///
/// Law `provenance.complete`: every artifact in the store should appear as a
/// [`RunOutput`] of some Run — bare `put_file` without a Run is incomplete.
pub trait ProvenanceRepository {
    /// TEMP: open a new [`Run`] from a [`ProcessSpec`] (status = Running).
    ///
    /// - `spec` — pinned recipe (name/version/entrypoint/models/hash).
    /// - `seed` — optional RNG seed for reproducible stochastic sensors.
    ///
    /// Returns the Run handle you pass to `add_*` / `derive` / terminal methods.
    /// Does **not** finish the Run — call [`succeed`] / [`fail`] / [`quarantine`].
    fn begin_run(&mut self, spec: &ProcessSpec, seed: Option<u64>) -> Result<Run, KernelError>;

    /// TEMP: record that `artifact` was an **input** to this Run.
    ///
    /// `role` is a human/machine label for the edge, e.g. `"audio"`, `"session"`,
    /// `"model_card"`. Multiple inputs per Run are fine (ordinal is assigned).
    ///
    /// Example: MERT feature registration takes the parent audio as input.
    fn add_input(&mut self, run: &Run, artifact: &Artifact, role: &str) -> Result<(), KernelError>;

    /// TEMP: record that `artifact` was **produced** by this Run.
    ///
    /// This is what law `provenance.complete` looks for: every stored artifact
    /// should show up here for some Run. `role` examples: `"audio"`, `"feature"`,
    /// `"diagnostics"`.
    fn add_output(&mut self, run: &Run, artifact: &Artifact, role: &str)
        -> Result<(), KernelError>;

    /// TEMP: lineage edge — `child` was derived from `parent` during `run`.
    ///
    /// `relation` is a short verb string, conventionally:
    /// - `"analyzed_from"` — FeatureBlob ← Audio (MIR / embeddings)
    /// - `"partitioned_from"` — HTML row ← page
    /// - `"extracted_from"` — stem ← mix (future)
    ///
    /// Used by [`closure`] and acyclicity checks. Does not move bytes — only
    /// records the graph edge.
    fn derive(
        &mut self,
        child: &Artifact,
        parent: &Artifact,
        run: &Run,
        relation: &str,
    ) -> Result<(), KernelError>;

    /// TEMP: mark the Run **Succeeded** and stamp `finished_at`.
    ///
    /// Call after all inputs/outputs/derivations for a happy path are recorded.
    /// Mutates `run` in place and persists the final status.
    fn succeed(&mut self, run: &mut Run) -> Result<(), KernelError>;

    /// TEMP: mark the Run **Failed** with a structured error map.
    ///
    /// Use when the process crashed or returned an unrecoverable error.
    /// `error` is free-form JSON-ish fields (`"message"`, `"kind"`, …).
    /// Outputs recorded before fail may still exist — consumers must check status.
    fn fail(
        &mut self,
        run: &mut Run,
        error: IndexMap<String, serde_json::Value>,
    ) -> Result<(), KernelError>;

    /// TEMP: mark the Run **Quarantined** (ran, but results must not be trusted).
    ///
    /// Softer than fail: used when diagnostics say "looks wrong / contaminated"
    /// without a hard crash. `diagnostics` holds the reasons. Quarantined Runs
    /// should not feed Promote/Act products.
    fn quarantine(
        &mut self,
        run: &mut Run,
        diagnostics: IndexMap<String, serde_json::Value>,
    ) -> Result<(), KernelError>;

    /// TEMP: transitive dependency set around an artifact (or related) UUID.
    ///
    /// BFS over [`Derivation`] edges in **both** directions (parents and children)
    /// starting at `node_id`. Answers "what else is connected to this artifact
    /// in the lineage graph?" Used for audits and cycle-related reasoning.
    ///
    /// Returns a set of artifact UUIDs including `node_id` itself.
    fn closure(&self, node_id: Uuid) -> Result<HashSet<Uuid>, KernelError>;

    /// TEMP: dump every Run (for law checks / debugging). Prefer filtered queries later.
    fn all_runs(&self) -> Result<Vec<Run>, KernelError>;

    /// TEMP: dump every RunOutput row — law `provenance.complete` scans this vs artifacts.
    fn all_outputs(&self) -> Result<Vec<RunOutput>, KernelError>;

    /// TEMP: dump every Derivation edge — law `provenance.acyclic` builds a graph from this.
    fn all_derivations(&self) -> Result<Vec<Derivation>, KernelError>;
}

/// In-memory provenance for tests and dry-runs.
/// TEMP: same trait contract as SQLite — swap impls without changing call sites.
#[derive(Default)]
pub struct MemoryProvenanceRepository {
    /// TEMP: run_id → Run record (status, times, error).
    runs: HashMap<RunId, Run>,
    /// TEMP: RunInput edges (artifact consumed by run under a role).
    inputs: Vec<RunInput>,
    /// TEMP: RunOutput edges (artifact produced by run under a role).
    outputs: Vec<RunOutput>,
    /// TEMP: parent↔child lineage edges with relation strings.
    derivations: Vec<Derivation>,
}

impl ProvenanceRepository for MemoryProvenanceRepository {
    fn begin_run(&mut self, spec: &ProcessSpec, seed: Option<u64>) -> Result<Run, KernelError> {
        let run = Run {
            run_id: RunId::new(),
            process_spec_id: spec.process_spec_id,
            parent_run_id: None,
            status: RunStatus::Running,
            random_seed: seed,
            environment_instance: IndexMap::new(),
            started_at: Utc::now(),
            finished_at: None,
            error: None,
        };
        self.runs.insert(run.run_id, run.clone());
        Ok(run)
    }

    fn add_input(&mut self, run: &Run, artifact: &Artifact, role: &str) -> Result<(), KernelError> {
        let ordinal = self
            .inputs
            .iter()
            .filter(|i| i.run_id == run.run_id)
            .count() as i64;
        self.inputs.push(RunInput {
            run_id: run.run_id,
            artifact_id: artifact.artifact_id,
            role: role.to_string(),
            ordinal,
        });
        Ok(())
    }

    fn add_output(
        &mut self,
        run: &Run,
        artifact: &Artifact,
        role: &str,
    ) -> Result<(), KernelError> {
        let ordinal = self
            .outputs
            .iter()
            .filter(|o| o.run_id == run.run_id)
            .count() as i64;
        self.outputs.push(RunOutput {
            run_id: run.run_id,
            artifact_id: artifact.artifact_id,
            role: role.to_string(),
            ordinal,
        });
        Ok(())
    }

    fn derive(
        &mut self,
        child: &Artifact,
        parent: &Artifact,
        run: &Run,
        relation: &str,
    ) -> Result<(), KernelError> {
        self.derivations.push(Derivation {
            child_artifact_id: child.artifact_id,
            parent_artifact_id: parent.artifact_id,
            run_id: run.run_id,
            relation: relation.to_string(),
        });
        Ok(())
    }

    fn succeed(&mut self, run: &mut Run) -> Result<(), KernelError> {
        run.status = RunStatus::Succeeded;
        run.finished_at = Some(Utc::now());
        self.runs.insert(run.run_id, run.clone());
        Ok(())
    }

    fn fail(
        &mut self,
        run: &mut Run,
        error: IndexMap<String, serde_json::Value>,
    ) -> Result<(), KernelError> {
        run.status = RunStatus::Failed;
        run.finished_at = Some(Utc::now());
        run.error = Some(error);
        self.runs.insert(run.run_id, run.clone());
        Ok(())
    }

    fn quarantine(
        &mut self,
        run: &mut Run,
        diagnostics: IndexMap<String, serde_json::Value>,
    ) -> Result<(), KernelError> {
        run.status = RunStatus::Quarantined;
        run.finished_at = Some(Utc::now());
        run.error = Some(diagnostics);
        self.runs.insert(run.run_id, run.clone());
        Ok(())
    }

    fn closure(&self, node_id: Uuid) -> Result<HashSet<Uuid>, KernelError> {
        // TEMP: BFS undirected over derivation edges — parents AND children.
        let mut seen = HashSet::new();
        let mut q = VecDeque::from([node_id]);
        while let Some(id) = q.pop_front() {
            if !seen.insert(id) {
                continue; // already visited
            }
            for d in &self.derivations {
                // TEMP: walk toward parents (child → parent).
                if d.child_artifact_id.as_uuid() == id {
                    q.push_back(d.parent_artifact_id.as_uuid());
                }
                // TEMP: walk toward children (parent → child).
                if d.parent_artifact_id.as_uuid() == id {
                    q.push_back(d.child_artifact_id.as_uuid());
                }
            }
        }
        Ok(seen)
    }

    fn all_runs(&self) -> Result<Vec<Run>, KernelError> {
        Ok(self.runs.values().cloned().collect())
    }

    fn all_outputs(&self) -> Result<Vec<RunOutput>, KernelError> {
        Ok(self.outputs.clone())
    }

    fn all_derivations(&self) -> Result<Vec<Derivation>, KernelError> {
        Ok(self.derivations.clone())
    }
}

/// SQLite-backed provenance store (separate from legacy music_database.db).
/// TEMP: durable twin of [`MemoryProvenanceRepository`]; same trait methods.
/// Open with [`SqliteProvenanceRepository::open`] (file) or `open_in_memory`.
pub struct SqliteProvenanceRepository {
    conn: Connection,
}

impl SqliteProvenanceRepository {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, KernelError> {
        let conn = Connection::open(path)?;
        let repo = Self { conn };
        repo.init_schema()?;
        Ok(repo)
    }

    pub fn open_in_memory() -> Result<Self, KernelError> {
        let conn = Connection::open_in_memory()?;
        let repo = Self { conn };
        repo.init_schema()?;
        Ok(repo)
    }

    fn init_schema(&self) -> Result<(), KernelError> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                process_spec_id TEXT NOT NULL,
                parent_run_id TEXT,
                status TEXT NOT NULL,
                random_seed INTEGER,
                environment_instance TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS run_inputs (
                run_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                role TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                PRIMARY KEY (run_id, role, ordinal)
            );
            CREATE TABLE IF NOT EXISTS run_outputs (
                run_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                role TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                PRIMARY KEY (run_id, role, ordinal)
            );
            CREATE TABLE IF NOT EXISTS derivations (
                child_artifact_id TEXT NOT NULL,
                parent_artifact_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                PRIMARY KEY (child_artifact_id, parent_artifact_id, run_id, relation)
            );
            "#,
        )?;
        Ok(())
    }

    fn upsert_run(&self, run: &Run) -> Result<(), KernelError> {
        self.conn.execute(
            r#"
            INSERT INTO runs (
                run_id, process_spec_id, parent_run_id, status, random_seed,
                environment_instance, started_at, finished_at, error
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            ON CONFLICT(run_id) DO UPDATE SET
                status=excluded.status,
                finished_at=excluded.finished_at,
                error=excluded.error
            "#,
            params![
                run.run_id.to_string(),
                run.process_spec_id.to_string(),
                run.parent_run_id.map(|id| id.to_string()),
                serde_json::to_string(&run.status)?,
                run.random_seed.map(|s| s as i64),
                serde_json::to_string(&run.environment_instance)?,
                run.started_at.to_rfc3339(),
                run.finished_at.map(|t| t.to_rfc3339()),
                run.error.as_ref().map(serde_json::to_string).transpose()?,
            ],
        )?;
        Ok(())
    }
}

impl ProvenanceRepository for SqliteProvenanceRepository {
    fn begin_run(&mut self, spec: &ProcessSpec, seed: Option<u64>) -> Result<Run, KernelError> {
        let run = Run {
            run_id: RunId::new(),
            process_spec_id: spec.process_spec_id,
            parent_run_id: None,
            status: RunStatus::Running,
            random_seed: seed,
            environment_instance: IndexMap::new(),
            started_at: Utc::now(),
            finished_at: None,
            error: None,
        };
        self.upsert_run(&run)?;
        Ok(run)
    }

    fn add_input(&mut self, run: &Run, artifact: &Artifact, role: &str) -> Result<(), KernelError> {
        let ordinal: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM run_inputs WHERE run_id = ?1",
            params![run.run_id.to_string()],
            |row| row.get(0),
        )?;
        self.conn.execute(
            "INSERT INTO run_inputs (run_id, artifact_id, role, ordinal) VALUES (?1, ?2, ?3, ?4)",
            params![
                run.run_id.to_string(),
                artifact.artifact_id.to_string(),
                role,
                ordinal
            ],
        )?;
        Ok(())
    }

    fn add_output(
        &mut self,
        run: &Run,
        artifact: &Artifact,
        role: &str,
    ) -> Result<(), KernelError> {
        let ordinal: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM run_outputs WHERE run_id = ?1",
            params![run.run_id.to_string()],
            |row| row.get(0),
        )?;
        self.conn.execute(
            "INSERT INTO run_outputs (run_id, artifact_id, role, ordinal) VALUES (?1, ?2, ?3, ?4)",
            params![
                run.run_id.to_string(),
                artifact.artifact_id.to_string(),
                role,
                ordinal
            ],
        )?;
        Ok(())
    }

    fn derive(
        &mut self,
        child: &Artifact,
        parent: &Artifact,
        run: &Run,
        relation: &str,
    ) -> Result<(), KernelError> {
        self.conn.execute(
            "INSERT OR IGNORE INTO derivations (child_artifact_id, parent_artifact_id, run_id, relation) VALUES (?1, ?2, ?3, ?4)",
            params![
                child.artifact_id.to_string(),
                parent.artifact_id.to_string(),
                run.run_id.to_string(),
                relation
            ],
        )?;
        Ok(())
    }

    fn succeed(&mut self, run: &mut Run) -> Result<(), KernelError> {
        run.status = RunStatus::Succeeded;
        run.finished_at = Some(Utc::now());
        self.upsert_run(run)
    }

    fn fail(
        &mut self,
        run: &mut Run,
        error: IndexMap<String, serde_json::Value>,
    ) -> Result<(), KernelError> {
        run.status = RunStatus::Failed;
        run.finished_at = Some(Utc::now());
        run.error = Some(error);
        self.upsert_run(run)
    }

    fn quarantine(
        &mut self,
        run: &mut Run,
        diagnostics: IndexMap<String, serde_json::Value>,
    ) -> Result<(), KernelError> {
        run.status = RunStatus::Quarantined;
        run.finished_at = Some(Utc::now());
        run.error = Some(diagnostics);
        self.upsert_run(run)
    }

    fn closure(&self, node_id: Uuid) -> Result<HashSet<Uuid>, KernelError> {
        let derivations = self.all_derivations()?;
        let mut seen = HashSet::new();
        let mut q = VecDeque::from([node_id]);
        while let Some(id) = q.pop_front() {
            if !seen.insert(id) {
                continue;
            }
            for d in &derivations {
                if d.child_artifact_id.as_uuid() == id {
                    q.push_back(d.parent_artifact_id.as_uuid());
                }
                if d.parent_artifact_id.as_uuid() == id {
                    q.push_back(d.child_artifact_id.as_uuid());
                }
            }
        }
        Ok(seen)
    }

    fn all_runs(&self) -> Result<Vec<Run>, KernelError> {
        let mut stmt = self.conn.prepare(
            "SELECT run_id, process_spec_id, parent_run_id, status, random_seed, environment_instance, started_at, finished_at, error FROM runs",
        )?;
        let rows = stmt.query_map([], |row| {
            let run_id: String = row.get(0)?;
            let process_spec_id: String = row.get(1)?;
            let parent_run_id: Option<String> = row.get(2)?;
            let status: String = row.get(3)?;
            let random_seed: Option<i64> = row.get(4)?;
            let environment_instance: String = row.get(5)?;
            let started_at: String = row.get(6)?;
            let finished_at: Option<String> = row.get(7)?;
            let error: Option<String> = row.get(8)?;
            Ok((
                run_id,
                process_spec_id,
                parent_run_id,
                status,
                random_seed,
                environment_instance,
                started_at,
                finished_at,
                error,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (
                run_id,
                process_spec_id,
                parent_run_id,
                status,
                random_seed,
                environment_instance,
                started_at,
                finished_at,
                error,
            ) = row?;
            out.push(Run {
                run_id: RunId::from_uuid(
                    Uuid::parse_str(&run_id)
                        .map_err(|e| KernelError::Other(format!("bad run_id: {e}")))?,
                ),
                process_spec_id: Uuid::parse_str(&process_spec_id)
                    .map_err(|e| KernelError::Other(format!("bad process_spec_id: {e}")))?,
                parent_run_id: parent_run_id
                    .map(|id| {
                        Uuid::parse_str(&id)
                            .map(RunId::from_uuid)
                            .map_err(|e| KernelError::Other(format!("bad parent_run_id: {e}")))
                    })
                    .transpose()?,
                status: serde_json::from_str(&status)?,
                random_seed: random_seed.map(|s| s as u64),
                environment_instance: serde_json::from_str(&environment_instance)?,
                started_at: DateTime::parse_from_rfc3339(&started_at)
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(|e| KernelError::Other(format!("bad started_at: {e}")))?,
                finished_at: finished_at
                    .map(|t| {
                        DateTime::parse_from_rfc3339(&t)
                            .map(|dt| dt.with_timezone(&Utc))
                            .map_err(|e| KernelError::Other(format!("bad finished_at: {e}")))
                    })
                    .transpose()?,
                error: error.map(|e| serde_json::from_str(&e)).transpose()?,
            });
        }
        Ok(out)
    }

    fn all_outputs(&self) -> Result<Vec<RunOutput>, KernelError> {
        let mut stmt = self
            .conn
            .prepare("SELECT run_id, artifact_id, role, ordinal FROM run_outputs")?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (run_id, artifact_id, role, ordinal) = row?;
            out.push(RunOutput {
                run_id: RunId::from_uuid(
                    Uuid::parse_str(&run_id).map_err(|e| KernelError::Other(e.to_string()))?,
                ),
                artifact_id: ArtifactId::from_uuid(
                    Uuid::parse_str(&artifact_id).map_err(|e| KernelError::Other(e.to_string()))?,
                ),
                role,
                ordinal,
            });
        }
        Ok(out)
    }

    fn all_derivations(&self) -> Result<Vec<Derivation>, KernelError> {
        let mut stmt = self.conn.prepare(
            "SELECT child_artifact_id, parent_artifact_id, run_id, relation FROM derivations",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (child, parent, run_id, relation) = row?;
            out.push(Derivation {
                child_artifact_id: ArtifactId::from_uuid(
                    Uuid::parse_str(&child).map_err(|e| KernelError::Other(e.to_string()))?,
                ),
                parent_artifact_id: ArtifactId::from_uuid(
                    Uuid::parse_str(&parent).map_err(|e| KernelError::Other(e.to_string()))?,
                ),
                run_id: RunId::from_uuid(
                    Uuid::parse_str(&run_id).map_err(|e| KernelError::Other(e.to_string()))?,
                ),
                relation,
            });
        }
        Ok(out)
    }
}

use chrono::DateTime;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{implementation_hash, ArtifactKind};
    use indexmap::IndexMap;

    fn dummy_spec() -> ProcessSpec {
        ProcessSpec {
            process_spec_id: Uuid::new_v4(),
            name: "test".into(),
            version: "1".into(),
            stage: "test".into(),
            source_code_artifact_id: ArtifactId::new(),
            code_commit: "deadbeef".into(),
            entrypoint: "test".into(),
            parameter_set_id: Uuid::new_v4(),
            environment_spec_id: Uuid::new_v4(),
            dependency_lock_artifact_id: ArtifactId::new(),
            model_artifact_ids: vec![],
            implementation_hash: implementation_hash("test"),
        }
    }

    fn dummy_artifact() -> Artifact {
        Artifact {
            artifact_id: ArtifactId::new(),
            kind: ArtifactKind::Diagnostics,
            content_sha256: "abc".into(),
            payload_sha256: None,
            media_type: "text/plain".into(),
            byte_size: 3,
            object_uri: "memory://abc".into(),
            created_at: Utc::now(),
            source_uri: None,
            source_system: None,
            source_observed_at: None,
            metadata: IndexMap::new(),
        }
    }

    #[test]
    fn memory_run_lifecycle() {
        let mut repo = MemoryProvenanceRepository::default();
        let spec = dummy_spec();
        let mut run = repo.begin_run(&spec, Some(1)).unwrap();
        let art = dummy_artifact();
        repo.add_output(&run, &art, "out").unwrap();
        repo.succeed(&mut run).unwrap();
        assert_eq!(run.status, RunStatus::Succeeded);
        assert_eq!(repo.all_outputs().unwrap().len(), 1);
    }

    #[test]
    fn sqlite_run_lifecycle() {
        let mut repo = SqliteProvenanceRepository::open_in_memory().unwrap();
        let spec = dummy_spec();
        let mut run = repo.begin_run(&spec, None).unwrap();
        let parent = dummy_artifact();
        let child = dummy_artifact();
        repo.add_output(&run, &child, "child").unwrap();
        repo.derive(&child, &parent, &run, "from").unwrap();
        repo.succeed(&mut run).unwrap();
        assert_eq!(repo.all_derivations().unwrap().len(), 1);
        let c = repo.closure(child.artifact_id.as_uuid()).unwrap();
        assert!(c.contains(&parent.artifact_id.as_uuid()));
    }
}
