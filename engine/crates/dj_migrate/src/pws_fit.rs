//! Fit a vertical LF slice bundle (Python LF → majority vote → canary gate).
//!
//! # TEMP learning tour
//!
//! 1. `load_bundle` — JSON strings → kernel InferenceUnit / EvidenceEmission
//! 2. `majority_vote_fit` — Decide
//! 3. `estimate_source_accuracies` + `gate_canaries` — Promote report
//! 4. write `VerticalSliceReport` JSON (no pseudo-labels — that's `run-round`)
//!
//! Input JSON shape is produced by `sensors.lfs.write_vertical_slice`.
//! This is the first cortex loop: symbolic LF emissions in, `AxisBelief`s out.
//!
//! Grep `TEMP:` to strip pedagogy later.

use anyhow::{Context, Result};
use chrono::Utc;
use dj_kernel::{
    estimate_source_accuracies, gate_canaries, majority_vote_fit, Axis, CanarySpec,
    DecisionRuleSpec, EvidenceEmission, InferenceUnit, ModelId, NativeScore, RunId, SubjectRef,
    VerticalSliceReport,
};
use indexmap::IndexMap;
use serde::Deserialize;
use std::fs;
use std::path::Path;
use uuid::Uuid;

/// TEMP: serde shape of `staging/vertical_lf_bundle.json` from Python.
#[derive(Debug, Deserialize)]
pub struct Bundle {
    pub units: Vec<UnitIn>,
    pub emissions: Vec<EmissionIn>,
    /// Prefer ``canaries``; singular kept for older bundles.
    #[serde(default)]
    pub canary: Option<CanaryIn>,
    #[serde(default)]
    pub canaries: Vec<CanaryIn>,
    /// Held-out gold keyed by unit_id string — independent of LF votes.
    pub gold_by_unit: IndexMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct UnitIn {
    pub unit_id: String,
    pub set_id: String,
    pub axis: String,
    pub subject_type: String,
    pub subject_id: String,
    pub generation_parameters_hash: String,
    pub split: String,
}

#[derive(Debug, Deserialize)]
pub struct EmissionIn {
    pub emission_id: String,
    pub unit_id: String,
    pub source_spec_id: String,
    pub source_name: String,
    pub source_family: String,
    pub producer_run_id: String,
    pub value: Option<serde_json::Value>,
    pub abstained: bool,
    pub native_score: NativeScoreIn,
    #[serde(default)]
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct NativeScoreIn {
    pub family: String,
    pub value: f64,
    pub scale: String,
    pub support: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CanaryIn {
    pub source_name: String,
    pub known_accuracy: f64,
    pub tolerance: f64,
}

pub struct ParsedBundle {
    pub units: Vec<InferenceUnit>,
    pub emissions: Vec<EvidenceEmission>,
    pub canaries: Vec<CanarySpec>,
    pub gold_by_unit: IndexMap<Uuid, String>,
    pub source_spec_ids: Vec<Uuid>,
}

/// TEMP: JSON boundary → typed kernel structs (shared by pws-fit and run-round).
pub fn load_bundle(bundle_path: &Path) -> Result<ParsedBundle> {
    // TEMP: read Python's staging/vertical_lf_bundle.json as UTF-8 text.
    let text = fs::read_to_string(bundle_path)
        .with_context(|| format!("read {}", bundle_path.display()))?;
    let bundle: Bundle = serde_json::from_str(&text)?;

    // TEMP: one RunId stamps all units loaded in this CLI invocation.
    let run_id = RunId::new();
    // TEMP: string UUIDs / axis names from JSON → opaque kernel types.
    let units: Vec<InferenceUnit> = bundle
        .units
        .iter()
        .map(|u| {
            Ok(InferenceUnit {
                unit_id: Uuid::parse_str(&u.unit_id)?,
                set_id: u.set_id.clone(),
                axis: parse_axis(&u.axis)?,
                subject: SubjectRef {
                    subject_type: u.subject_type.clone(),
                    subject_id: Uuid::parse_str(&u.subject_id)?,
                },
                candidate: None,
                mix_region: None,
                generated_by_run_id: run_id,
                generation_parameters_hash: u.generation_parameters_hash.clone(),
                split: u.split.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // TEMP: collect unique LF process-spec ids for AlgorithmBundle hashing.
    let mut source_spec_ids = Vec::new();
    let emissions: Vec<EvidenceEmission> = bundle
        .emissions
        .iter()
        .map(|e| {
            let spec_id = Uuid::parse_str(&e.source_spec_id)?;
            if !source_spec_ids.contains(&spec_id) {
                source_spec_ids.push(spec_id);
            }
            let em = EvidenceEmission {
                emission_id: Uuid::parse_str(&e.emission_id)?,
                unit_id: Uuid::parse_str(&e.unit_id)?,
                source_spec_id: spec_id,
                source_name: e.source_name.clone(),
                source_family: e.source_family.clone(),
                producer_run_id: RunId::from_uuid(Uuid::parse_str(&e.producer_run_id)?),
                value: e.value.clone(),
                abstained: e.abstained,
                native_score: NativeScore {
                    family: e.native_score.family.clone(),
                    value: e.native_score.value,
                    scale: e.native_score.scale.clone(),
                    support: e.native_score.support,
                },
                evidence_ids: e
                    .evidence_ids
                    .iter()
                    .map(|id| Uuid::parse_str(id))
                    .collect::<Result<Vec<_>, _>>()?,
            };
            // TEMP: enforce abstain XOR value at the Rust boundary too.
            em.validate()?;
            Ok(em)
        })
        .collect::<Result<Vec<_>>>()?;

    // TEMP: gold keys are unit_id strings in JSON → Uuid map for canary scoring.
    let mut gold: IndexMap<Uuid, String> = IndexMap::new();
    for (k, v) in &bundle.gold_by_unit {
        gold.insert(Uuid::parse_str(k)?, v.clone());
    }

    // TEMP: prefer `canaries[]`; fall back to singular `canary` for old bundles.
    let mut canary_ins = bundle.canaries.clone();
    if canary_ins.is_empty() {
        if let Some(c) = &bundle.canary {
            canary_ins.push(c.clone());
        }
    }
    if canary_ins.is_empty() {
        anyhow::bail!("bundle missing canaries (and no singular canary)");
    }
    let canaries: Vec<CanarySpec> = canary_ins
        .iter()
        .map(|c| CanarySpec {
            source_name: c.source_name.clone(),
            known_accuracy: c.known_accuracy,
            tolerance: c.tolerance,
        })
        .collect();

    Ok(ParsedBundle {
        units,
        emissions,
        canaries,
        gold_by_unit: gold,
        source_spec_ids,
    })
}

/// TEMP: `dj_migrate pws-fit` body — Decide + canary report only.
pub fn run(bundle_path: &Path, out_path: &Path) -> Result<VerticalSliceReport> {
    // TEMP: parse Propose JSON → kernel types.
    let parsed = load_bundle(bundle_path)?;
    let run_id = RunId::new();

    // TEMP ——— DECIDE ———
    let beliefs = majority_vote_fit(
        &parsed.units,
        &parsed.emissions,
        run_id,
        ModelId::new(),
        &DecisionRuleSpec::default(),
    )?;

    // TEMP ——— PROMOTE REPORT (no Act) ———
    let accuracies = estimate_source_accuracies(&parsed.emissions, &parsed.gold_by_unit)?;
    let gate = gate_canaries(&accuracies, &parsed.canaries);

    let report = VerticalSliceReport {
        created_at: Utc::now(),
        n_units: parsed.units.len(),
        n_emissions: parsed.emissions.len(),
        n_abstained: parsed.emissions.iter().filter(|e| e.abstained).count(),
        beliefs,
        accuracies,
        gate,
    };

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    // TEMP: write staging/vertical_lf_report.json for humans / CI.
    fs::write(out_path, serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

fn parse_axis(s: &str) -> Result<Axis> {
    match s {
        "identity" => Ok(Axis::Identity),
        "placement" => Ok(Axis::Placement),
        "structure" => Ok(Axis::Structure),
        other => anyhow::bail!("unknown axis: {other}"),
    }
}
