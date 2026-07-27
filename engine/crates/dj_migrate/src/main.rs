//! Transfer + feature registration CLI for the provenance kernel.
//!
//! # TEMP learning tour — ENTRYPOINT FOR DEBUG SESSIONS
//!
//! You almost never call `dj_kernel` directly from a binary; this CLI does:
//!
//! | Subcommand | Spine port | Notes |
//! |------------|------------|--------|
//! | `pws-fit` | Decide + canary report | no Act / no pseudo-labels |
//! | `run-round` | Decide → Promote → Act | canary smoke path |
//! | `inventory`…`apply` | migrate transfer | fixtures-only dry-run first |
//! | `audio-put` / `feature-register` | Store | features-demo |
//!
//! # Pipelines
//!
//! - Transfer (dry-run): `inventory` → `map` → `stage` → `verify-manifest` → gated `apply`
//! - Features (no live DB): `audio-put` → Python `analyze.mert` → `feature-register`
//! - PWS: `pws-fit` on a Python LF emission bundle
//! - Co-training: `run-round` (fit → canary → promote/reject → opposite-view pseudo-labels)
//!
//! Grep `TEMP:` to strip pedagogy later.

mod apply;
mod cotraining_cli;
mod features_cli;
mod inventory;
mod legacy;
mod map;
mod pws_fit;
mod stage;
mod verify;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Gold development sets (BB11 / BB12).
/// TEMP: these fixture set ids appear throughout canary smoke.
pub const GOLD_SET_IDS: &[&str] = &["2nvzlh2k", "1fsnxchk"];

#[derive(Parser, Debug)]
#[command(
    name = "dj_migrate",
    about = "Legacy transfer + content-addressed audio/feature registration"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Read-only inventory of a legacy SQLite DB or gold fixtures.
    Inventory {
        #[arg(long, env = "LEGACY_DB_PATH")]
        db: Option<PathBuf>,
        #[arg(long, default_value = "fixtures/gold")]
        fixtures: PathBuf,
        #[arg(long, value_delimiter = ',')]
        sets: Option<Vec<String>>,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        fixtures_only: bool,
    },
    /// Produce id crosswalk + observation rewrite plan from an inventory JSON.
    Map {
        #[arg(long)]
        inventory: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Write a staging manifest (no byte copies unless --write-fixtures for tiny samples).
    Stage {
        #[arg(long)]
        map: PathBuf,
        #[arg(long, default_value = "staging")]
        staging_dir: PathBuf,
        #[arg(long, default_value_t = true)]
        dry_run: bool,
        #[arg(long)]
        write_fixtures: bool,
        #[arg(long, default_value = "fixtures/gold")]
        fixtures_dir: PathBuf,
    },
    /// Verify a staging manifest (existence + sha when paths reachable).
    VerifyManifest {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        require_reachable: bool,
    },
    /// Copy bytes into the kernel artifact store. Gated; off by default.
    Apply {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        artifact_root: PathBuf,
        #[arg(long)]
        i_know_this_copies_bytes: bool,
    },
    /// Fit a vertical LF slice bundle (Python emissions → majority vote → canary).
    PwsFit {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long, default_value = "staging/vertical_lf_report.json")]
        out: PathBuf,
    },
    /// One co-training round: fit → canary gate → promote/reject + pseudo-labels.
    RunRound {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long, default_value = "staging/cotrain_round_report.json")]
        out: PathBuf,
        /// Written only when the round is promoted (next-round inputs).
        #[arg(long, default_value = "staging/cotrain_pseudo_labels.json")]
        out_pseudo: PathBuf,
        /// Prior promoted round JSON (or report with a `round` field).
        #[arg(long)]
        prior_round: Option<PathBuf>,
        /// Pseudo-labels from a prior promoted round (array or `{pseudo_labels:[…]}`).
        #[arg(long)]
        prior_pseudo: Option<PathBuf>,
    },
    /// Deposit a local audio file into the content-addressed store (ArtifactKind::Audio).
    AudioPut {
        #[arg(long)]
        file: PathBuf,
        #[arg(long, default_value = "staging/artifacts")]
        artifact_root: PathBuf,
        /// Optional SQLite provenance DB (default: in-memory for the process).
        #[arg(long)]
        provenance_db: Option<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Register a FeatureBlob from a mert sensor JSON sidecar + parent audio id.
    FeatureRegister {
        #[arg(long)]
        parent_audio_id: String,
        /// Path to JSON written by `python -m sensors.mert`.
        #[arg(long)]
        mert_json: PathBuf,
        #[arg(long, default_value = "staging/artifacts")]
        artifact_root: PathBuf,
        #[arg(long)]
        provenance_db: Option<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Inventory {
            db,
            fixtures,
            sets,
            out,
            fixtures_only,
        } => {
            let set_ids =
                sets.unwrap_or_else(|| GOLD_SET_IDS.iter().map(|s| (*s).to_string()).collect());
            let report = inventory::run(db.as_deref(), &fixtures, &set_ids, fixtures_only)?;
            write_json(out.as_deref(), &report)?;
        }
        Commands::Map { inventory, out } => {
            let plan = map::run(&inventory)?;
            write_json(out.as_deref(), &plan)?;
        }
        Commands::Stage {
            map,
            staging_dir,
            dry_run,
            write_fixtures,
            fixtures_dir,
        } => {
            let manifest = stage::run(&map, &staging_dir, dry_run, write_fixtures, &fixtures_dir)?;
            println!("wrote staging manifest: {}", manifest.display());
        }
        Commands::VerifyManifest {
            manifest,
            require_reachable,
        } => {
            verify::run(&manifest, require_reachable)?;
        }
        Commands::Apply {
            manifest,
            artifact_root,
            i_know_this_copies_bytes,
        } => {
            apply::run(&manifest, &artifact_root, i_know_this_copies_bytes)?;
        }
        Commands::PwsFit { bundle, out } => {
            let report = pws_fit::run(&bundle, &out)?;
            println!(
                "pws-fit: units={} emissions={} abstained={} gate={} ({})",
                report.n_units,
                report.n_emissions,
                report.n_abstained,
                if report.gate.passed { "PASS" } else { "FAIL" },
                report.gate.detail
            );
            println!("wrote {}", out.display());
            if !report.gate.passed {
                anyhow::bail!("canary gate failed");
            }
        }
        Commands::RunRound {
            bundle,
            out,
            out_pseudo,
            prior_round,
            prior_pseudo,
        } => {
            let outcome = cotraining_cli::run(
                &bundle,
                &out,
                &out_pseudo,
                prior_pseudo.as_deref(),
                prior_round.as_deref(),
            )?;
            println!(
                "run-round: status={:?} reason={} units={} pseudo={} review={} gate={} ({})",
                outcome.round.status,
                outcome.reason,
                outcome.n_units,
                outcome.new_pseudo_labels.len(),
                outcome.review_unit_ids.len(),
                if outcome.gate.passed { "PASS" } else { "FAIL" },
                outcome.gate.detail
            );
            println!("wrote {}", out.display());
            if outcome.round.status == dj_kernel::RoundStatus::Promoted {
                println!("wrote {}", out_pseudo.display());
            }
            if outcome.reason != "promoted" {
                anyhow::bail!("co-training round not promoted: {}", outcome.reason);
            }
        }
        Commands::AudioPut {
            file,
            artifact_root,
            provenance_db,
            out,
        } => {
            let report = features_cli::audio_put(&file, &artifact_root, provenance_db.as_deref())?;
            write_json(out.as_deref(), &report)?;
            eprintln!(
                "audio-put: artifact_id={} sha256={}",
                report.artifact_id, report.content_sha256
            );
        }
        Commands::FeatureRegister {
            parent_audio_id,
            mert_json,
            artifact_root,
            provenance_db,
            out,
        } => {
            let report = features_cli::feature_register_mert(
                &artifact_root,
                &parent_audio_id,
                &mert_json,
                provenance_db.as_deref(),
            )?;
            write_json(out.as_deref(), &report)?;
            eprintln!(
                "feature-register: feature_id={} parent={} model={}",
                report.feature_artifact_id,
                report.parent_audio_artifact_id,
                report.model_artifact_id
            );
        }
    }
    Ok(())
}

fn write_json<T: serde::Serialize>(path: Option<&std::path::Path>, value: &T) -> Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    match path {
        Some(p) => {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(p, text)?;
            eprintln!("wrote {}", p.display());
        }
        None => println!("{text}"),
    }
    Ok(())
}
