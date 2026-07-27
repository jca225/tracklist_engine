//! Co-training round CLI: LF bundle → promote/reject + opposite-view pseudo-labels.
//!
//! # TEMP learning tour
//!
//! Called by `make canary-smoke` / `make cotrain-round`:
//! 1. `pws_fit::load_bundle` (same JSON as pws-fit)
//! 2. optional prior pseudo-labels from a previous promoted round
//! 3. `dj_kernel::run_cotraining_round`
//! 4. always write report; write pseudo-labels **only if** status == Promoted
//!
//! Consumes the same vertical LF JSON as `pws-fit`. No live DB / byte copy.
//! Grep `TEMP:` to strip pedagogy later.

use anyhow::{Context, Result};
use dj_kernel::{
    run_cotraining_round, AlgorithmBundle, CoTrainingRound, DecisionRuleSpec, PseudoLabel,
    RoundOutcome, RoundRequest, SnapshotId,
};
use serde::Deserialize;
use std::fs;
use std::path::Path;

use crate::pws_fit;

#[derive(Debug, Deserialize)]
struct PriorRoundFile {
    round: CoTrainingRound,
}

/// TEMP: `dj_migrate run-round` body — full Decide → Promote → Act.
pub fn run(
    bundle_path: &Path,
    out_report: &Path,
    out_pseudo: &Path,
    prior_pseudo_path: Option<&Path>,
    prior_round_path: Option<&Path>,
) -> Result<RoundOutcome> {
    // TEMP: same JSON parse as pws-fit (units + emissions + gold + canaries).
    let parsed = pws_fit::load_bundle(bundle_path)?;
    // TEMP: fingerprint which LF specs participated this round.
    let bundle = AlgorithmBundle::from_source_specs(parsed.source_spec_ids.clone());

    // TEMP: round>0 may feed prior pseudo-labels; round 0 has none.
    let prior_labels: Vec<PseudoLabel> = match prior_pseudo_path {
        Some(p) => {
            let text = fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
            // TEMP: accept bare array OR { "pseudo_labels": [...] } wrapper.
            if let Ok(v) = serde_json::from_str::<Vec<PseudoLabel>>(&text) {
                v
            } else {
                #[derive(Deserialize)]
                struct Wrap {
                    pseudo_labels: Vec<PseudoLabel>,
                }
                serde_json::from_str::<Wrap>(&text)?.pseudo_labels
            }
        }
        None => vec![],
    };

    // TEMP: parent CoTrainingRound required if we have prior labels (lineage).
    let parent: Option<CoTrainingRound> = match prior_round_path {
        Some(p) => {
            let text = fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
            if let Ok(r) = serde_json::from_str::<CoTrainingRound>(&text) {
                Some(r)
            } else {
                Some(serde_json::from_str::<PriorRoundFile>(&text)?.round)
            }
        }
        None => None,
    };

    if !prior_labels.is_empty() && parent.is_none() {
        anyhow::bail!("--prior-pseudo requires --prior-round (parent CoTrainingRound)");
    }

    // TEMP: kernel does Decide → gate → promote/reject → optional Act.
    let outcome = run_cotraining_round(RoundRequest {
        units: &parsed.units,
        emissions: &parsed.emissions,
        gold_by_unit: &parsed.gold_by_unit,
        canaries: &parsed.canaries,
        rule: &DecisionRuleSpec::default(),
        bundle: &bundle,
        parent: parent.as_ref(),
        prior_pseudo_labels: &prior_labels,
        input_snapshot_id: SnapshotId::new(),
    })?;

    // TEMP: always write the report (even on reject — gate detail lives here).
    if let Some(parent) = out_report.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out_report, serde_json::to_string_pretty(&outcome)?)?;

    // TEMP: Act products exist only on promote — rejected rounds leave no pseudo file.
    if outcome.round.status == dj_kernel::RoundStatus::Promoted {
        if let Some(parent) = out_pseudo.parent() {
            fs::create_dir_all(parent)?;
        }
        let payload = serde_json::json!({
            "round": outcome.round,
            "pseudo_labels": outcome.new_pseudo_labels,
            "review_unit_ids": outcome.review_unit_ids,
        });
        fs::write(out_pseudo, serde_json::to_string_pretty(&payload)?)?;
    }

    Ok(outcome)
}
