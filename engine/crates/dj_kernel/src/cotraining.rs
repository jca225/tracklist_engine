//! Co-training rounds and pseudo-label lineage (pseudocode §12).
//!
//! # TEMP learning tour — THIS IS PROMOTE + ACT
//!
//! ```text
//! run_cotraining_round:
//!   majority_vote_fit          (Decide)
//!   estimate_source_accuracies
//!   gate_canaries              (Promote YES/NO)
//!   if fail → Rejected, no pseudo-labels
//!   if pass → Promoted + eligible_new_pseudo_labels (Act)
//!             target_view = opposite of originating family
//! ```
//!
//! Round 0: fit on LF emissions → canary gate → promote or reject.
//! On promote, emit [`PseudoLabel`]s targeted at the *opposite* view only.
//! Round *r+1* may consume those labels only after
//! [`validate_pseudo_label_for_next_round`] (no self-consumption, no view leakage).
//!
//! This is the Tesla data-engine loop in miniature: auto-label → gate →
//! promote → mine hard cases → next generation. Full EM / generative Snorkel
//! fit still lives behind [`crate::pws::majority_vote_fit`] as a stand-in.
//!
//! Grep `TEMP:` to strip pedagogy later.

use crate::error::KernelError;
use crate::ids::{BeliefId, ModelId, RoundId, RunId, SnapshotId};
use crate::pws::{
    estimate_source_accuracies, gate_canaries, majority_vote_fit, CanarySpec, DecisionRuleSpec,
    EvidenceEmission, GateResult, InferenceUnit, SourceAccuracy,
};
use crate::types::{Axis, AxisBelief, Decision, NativeScore};
use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// Declared algorithm bundle for a co-training generation (spec ids, not code).
/// TEMP: fingerprint of which LFs/gates ran this round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlgorithmBundle {
    pub algorithm_bundle_id: Uuid,
    pub evidence_source_spec_ids: Vec<Uuid>,
    pub gate_spec_ids: Vec<Uuid>,
    pub bundle_hash: String,
}

impl AlgorithmBundle {
    pub fn from_source_specs(spec_ids: impl IntoIterator<Item = Uuid>) -> Self {
        let evidence_source_spec_ids: Vec<Uuid> = spec_ids.into_iter().collect();
        let mut hasher = Sha256::new();
        for id in &evidence_source_spec_ids {
            hasher.update(id.as_bytes());
        }
        let bundle_hash = hex::encode(hasher.finalize());
        Self {
            algorithm_bundle_id: Uuid::new_v4(),
            evidence_source_spec_ids,
            gate_spec_ids: vec![],
            bundle_hash,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoundStatus {
    Candidate,
    Promoted,
    Rejected,
}

/// One bootstrap generation (candidate → promoted|rejected).
/// TEMP: look at `status` in staging/cotrain_round_report.json after smoke.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoTrainingRound {
    pub round_id: RoundId,
    pub round_index: i32,
    pub parent_round_id: Option<RoundId>,
    pub input_snapshot_id: SnapshotId,
    pub algorithm_bundle_id: Uuid,
    pub fitted_model_ids: IndexMap<String, ModelId>,
    pub prediction_snapshot_id: Option<SnapshotId>,
    pub status: RoundStatus,
}

/// Accepted belief promoted into training material for a later round / opposite view.
/// TEMP: written only on promote; next round must not self-consume (view leakage).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PseudoLabel {
    pub pseudo_label_id: Uuid,
    pub produced_in_round_id: RoundId,
    pub unit_id: Uuid,
    pub axis: Axis,
    pub value: serde_json::Value,
    pub source_belief_id: BeliefId,
    pub source_prediction_run_id: RunId,
    pub source_fitted_model_id: ModelId,
    pub contributing_emission_ids: Vec<Uuid>,
    pub confidence_at_acceptance: f64,
    pub entropy_at_acceptance: f64,
    /// Source family that may *consume* this label next (never the originating family).
    pub target_view: String,
    pub eligible_from_round_index: i32,
    pub revoked_at: Option<DateTime<Utc>>,
}

/// Why bootstrap / a single round stopped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoundOutcome {
    pub round: CoTrainingRound,
    /// promoted | round_rejected
    pub reason: String,
    pub gate: GateResult,
    pub beliefs: Vec<AxisBelief>,
    pub accuracies: Vec<SourceAccuracy>,
    pub new_pseudo_labels: Vec<PseudoLabel>,
    /// Units worth human / agent attention (unresolved, rejected, quarantined).
    pub review_unit_ids: Vec<Uuid>,
    pub n_units: usize,
    pub n_emissions: usize,
    pub n_abstained: usize,
    pub created_at: DateTime<Utc>,
}

/// Opposite co-training view: symbolic ↔ signal.
/// TEMP: claimed_stem is symbolic → pseudo-labels target signal (and vice versa).
pub fn opposite_view(family: &str) -> Option<&'static str> {
    match family {
        "symbolic" => Some("signal"),
        "signal" => Some("symbolic"),
        _ => None,
    }
}

/// Index of emissions for provenance checks (id → source_family).
/// TEMP: used to block pseudo-labels from leaking back into their own view.
#[derive(Debug, Clone, Default)]
pub struct EmissionIndex {
    family_by_id: IndexMap<Uuid, String>,
}

impl EmissionIndex {
    pub fn from_emissions(emissions: &[EvidenceEmission]) -> Self {
        let mut family_by_id = IndexMap::new();
        for e in emissions {
            family_by_id.insert(e.emission_id, e.source_family.clone());
        }
        Self { family_by_id }
    }

    pub fn insert(&mut self, emission_id: Uuid, source_family: impl Into<String>) {
        self.family_by_id.insert(emission_id, source_family.into());
    }

    pub fn family(&self, emission_id: Uuid) -> Option<&str> {
        self.family_by_id.get(&emission_id).map(|s| s.as_str())
    }

    /// Families reachable from a pseudo-label's contributing emissions.
    pub fn closure_families(&self, label: &PseudoLabel) -> Result<Vec<String>, KernelError> {
        let mut out = Vec::new();
        for id in &label.contributing_emission_ids {
            match self.family(*id) {
                Some(f) => {
                    if !out.iter().any(|x| x == f) {
                        out.push(f.to_string());
                    }
                }
                None => {
                    return Err(KernelError::UnfalsifiablePseudoLabel);
                }
            }
        }
        Ok(out)
    }
}

/// Reject labels that would create cycles, view leakage, or unfalsifiable roots.
pub fn validate_pseudo_label_for_next_round(
    label: &PseudoLabel,
    target_round: &CoTrainingRound,
    target_source_family: &str,
    index: &EmissionIndex,
) -> Result<(), KernelError> {
    // TEMP: revoked labels must never re-enter training.
    if label.revoked_at.is_some() {
        return Err(KernelError::Other("pseudo-label revoked".into()));
    }
    // TEMP: label created in round r is only eligible from r+1 onward.
    if label.eligible_from_round_index > target_round.round_index {
        return Err(KernelError::Other(format!(
            "pseudo-label not eligible until round {}",
            label.eligible_from_round_index
        )));
    }
    // TEMP: a round cannot consume labels it itself produced (cycle).
    if label.produced_in_round_id == target_round.round_id {
        return Err(KernelError::ProvenanceCycle);
    }
    // TEMP: v1 requires labels come from the immediate parent round.
    if let Some(parent) = target_round.parent_round_id {
        if label.produced_in_round_id != parent {
            return Err(KernelError::ProvenanceCycle);
        }
    }

    // TEMP: walk contributing emissions → their source_family set.
    let families = index.closure_families(label)?;
    if families.is_empty() {
        return Err(KernelError::UnfalsifiablePseudoLabel);
    }
    // TEMP: consumer family must NOT appear in the label's provenance (no view leak).
    if families.iter().any(|f| f == target_source_family) {
        return Err(KernelError::ViewLeakage(target_source_family.into()));
    }
    // TEMP: label.target_view must equal the family about to consume it.
    if label.target_view != target_source_family {
        return Err(KernelError::Other(format!(
            "label target_view {} != consumer family {}",
            label.target_view, target_source_family
        )));
    }
    Ok(())
}

/// Accepted beliefs → pseudo-labels for the opposite view only.
pub fn eligible_new_pseudo_labels(
    beliefs: &[AxisBelief],
    emissions: &[EvidenceEmission],
    round: &CoTrainingRound,
    rule: &DecisionRuleSpec,
) -> Result<Vec<PseudoLabel>, KernelError> {
    let mut out = Vec::new();

    for belief in beliefs {
        // TEMP: only Accepted beliefs may become training material.
        if belief.decision != Decision::Accepted {
            continue;
        }
        // TEMP: MAP label excluding the synthetic __abstain__ bucket.
        let Some((value_key, &confidence)) = belief
            .posterior
            .iter()
            .filter(|(k, _)| k.as_str() != "__abstain__")
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        else {
            continue;
        };
        // TEMP: re-check thresholds (belt + suspenders vs decide()).
        if confidence < rule.minimum_posterior || belief.entropy > rule.maximum_entropy {
            continue;
        }

        // TEMP: which view produced this belief? Need exactly one non-abstain family.
        let mut families: Vec<String> = Vec::new();
        for eid in &belief.contributing_emission_ids {
            let Some(em) = emissions.iter().find(|e| e.emission_id == *eid) else {
                continue;
            };
            if em.abstained {
                continue;
            }
            if !families.iter().any(|f| f == &em.source_family) {
                families.push(em.source_family.clone());
            }
        }
        if families.len() != 1 {
            // TEMP: mixed families → don't invent a target view.
            continue;
        }
        // TEMP: symbolic → signal (and reverse); unknown families skipped.
        let Some(target) = opposite_view(&families[0]) else {
            continue;
        };

        out.push(PseudoLabel {
            pseudo_label_id: Uuid::new_v4(),
            produced_in_round_id: round.round_id,
            unit_id: belief.unit_id,
            axis: belief.axis,
            value: serde_json::Value::String(value_key.clone()),
            source_belief_id: belief.belief_id,
            source_prediction_run_id: belief.inference_run_id,
            source_fitted_model_id: belief.fitted_model_id,
            contributing_emission_ids: belief.contributing_emission_ids.clone(),
            confidence_at_acceptance: confidence,
            entropy_at_acceptance: belief.entropy,
            target_view: target.to_string(),
            // TEMP: next round index is the earliest that may consume this.
            eligible_from_round_index: round.round_index + 1,
            revoked_at: None,
        });
    }
    Ok(out)
}

fn inject_pseudo_as_emissions(
    labels: &[PseudoLabel],
    run_id: RunId,
    index: &mut EmissionIndex,
) -> Result<Vec<EvidenceEmission>, KernelError> {
    // TEMP: turn prior-round PseudoLabels into synthetic emissions for this fit.
    let mut out = Vec::new();
    for label in labels {
        let emission_id = Uuid::new_v4();
        let em = EvidenceEmission {
            emission_id,
            unit_id: label.unit_id,
            source_spec_id: Uuid::nil(),
            source_name: "pseudo_label".into(),
            // TEMP: family = target_view so the *consumer* view sees the vote.
            source_family: label.target_view.clone(),
            producer_run_id: run_id,
            value: Some(label.value.clone()),
            abstained: false,
            native_score: NativeScore {
                family: label.target_view.clone(),
                value: label.confidence_at_acceptance,
                scale: "posterior".into(),
                support: Some(1),
            },
            evidence_ids: label.contributing_emission_ids.clone(),
        };
        em.validate()?;
        index.insert(emission_id, label.target_view.clone());
        out.push(em);
    }
    Ok(out)
}

fn disputed_or_uncertain_units(beliefs: &[AxisBelief]) -> Vec<Uuid> {
    beliefs
        .iter()
        .filter(|b| {
            matches!(
                b.decision,
                Decision::Unresolved | Decision::Rejected | Decision::Quarantined
            )
        })
        .map(|b| b.unit_id)
        .collect()
}

/// Inputs for one co-training generation (avoids a 9-arg fit call).
#[derive(Debug, Clone)]
pub struct RoundRequest<'a> {
    pub units: &'a [InferenceUnit],
    pub emissions: &'a [EvidenceEmission],
    pub gold_by_unit: &'a IndexMap<Uuid, String>,
    pub canaries: &'a [CanarySpec],
    pub rule: &'a DecisionRuleSpec,
    pub bundle: &'a AlgorithmBundle,
    pub parent: Option<&'a CoTrainingRound>,
    pub prior_pseudo_labels: &'a [PseudoLabel],
    pub input_snapshot_id: SnapshotId,
}

/// Run one co-training round (fit → gate → promote/reject → optional pseudo-labels).
///
/// TEMP debugger breakpoints: start here, then step into majority_vote_fit / gate_canaries.
pub fn run_cotraining_round(req: RoundRequest<'_>) -> Result<RoundOutcome, KernelError> {
    // TEMP step 0: refuse LF rows before anything else.
    for e in req.emissions {
        e.validate()?;
    }

    // TEMP: round 0 if no parent; else parent.index + 1.
    let round_index = req.parent.map(|p| p.round_index + 1).unwrap_or(0);
    let mut round = CoTrainingRound {
        round_id: RoundId::new(),
        round_index,
        parent_round_id: req.parent.map(|p| p.round_id),
        input_snapshot_id: req.input_snapshot_id,
        algorithm_bundle_id: req.bundle.algorithm_bundle_id,
        fitted_model_ids: IndexMap::new(),
        prediction_snapshot_id: None,
        status: RoundStatus::Candidate, // TEMP: not promoted until gate passes
    };

    // TEMP: index emission_id → family for leakage checks.
    let mut index = EmissionIndex::from_emissions(req.emissions);
    let run_id = RunId::new();
    let model_id = ModelId::new();

    // TEMP: if prior pseudo-labels exist, prove they're safe for *this* round.
    for label in req.prior_pseudo_labels {
        validate_pseudo_label_for_next_round(label, &round, &label.target_view, &index)?;
    }
    // TEMP: LF emissions + injected prior pseudo-labels = full matrix for fit.
    let mut all_emissions = req.emissions.to_vec();
    all_emissions.extend(inject_pseudo_as_emissions(
        req.prior_pseudo_labels,
        run_id,
        &mut index,
    )?);

    // TEMP ——— DECIDE ———
    let beliefs = majority_vote_fit(req.units, &all_emissions, run_id, model_id, req.rule)?;
    round.fitted_model_ids.insert("identity".into(), model_id);

    // TEMP ——— PROMOTE GATE ———
    let accuracies = estimate_source_accuracies(&all_emissions, req.gold_by_unit)?;
    let gate = gate_canaries(&accuracies, req.canaries);

    let n_abstained = all_emissions.iter().filter(|e| e.abstained).count();
    let created_at = Utc::now();

    // TEMP: fail-closed — reject writes NO Act products.
    if !gate.passed {
        round.status = RoundStatus::Rejected;
        return Ok(RoundOutcome {
            round,
            reason: "round_rejected".into(),
            gate,
            beliefs: beliefs.clone(),
            accuracies,
            new_pseudo_labels: vec![], // TEMP: empty on reject
            review_unit_ids: disputed_or_uncertain_units(&beliefs),
            n_units: req.units.len(),
            n_emissions: all_emissions.len(),
            n_abstained,
            created_at,
        });
    }

    // TEMP ——— ACT ——— gate passed → promote + opposite-view pseudo-labels.
    round.status = RoundStatus::Promoted;
    round.prediction_snapshot_id = Some(SnapshotId::new());
    let new_pseudo_labels = eligible_new_pseudo_labels(&beliefs, &all_emissions, &round, req.rule)?;

    Ok(RoundOutcome {
        round,
        reason: "promoted".into(),
        gate,
        beliefs: beliefs.clone(),
        accuracies,
        new_pseudo_labels,
        review_unit_ids: disputed_or_uncertain_units(&beliefs),
        n_units: req.units.len(),
        n_emissions: all_emissions.len(),
        n_abstained,
        created_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pws::DecisionRuleSpec;
    use crate::types::{Axis, SubjectRef};

    fn unit(id: Uuid, run: RunId) -> InferenceUnit {
        InferenceUnit {
            unit_id: id,
            set_id: "2nvzlh2k".into(),
            axis: Axis::Identity,
            subject: SubjectRef {
                subject_type: "set-slot".into(),
                subject_id: Uuid::new_v4(),
            },
            candidate: None,
            mix_region: None,
            generated_by_run_id: run,
            generation_parameters_hash: "test".into(),
            split: "train".into(),
        }
    }

    fn vote(unit_id: Uuid, run: RunId, family: &str, name: &str, value: &str) -> EvidenceEmission {
        EvidenceEmission {
            emission_id: Uuid::new_v4(),
            unit_id,
            source_spec_id: Uuid::new_v4(),
            source_name: name.into(),
            source_family: family.into(),
            producer_run_id: run,
            value: Some(serde_json::Value::String(value.into())),
            abstained: false,
            native_score: NativeScore {
                family: family.into(),
                value: 1.0,
                scale: "count".into(),
                support: Some(1),
            },
            evidence_ids: vec![],
        }
    }

    #[test]
    fn opposite_view_symbolic_signal() {
        assert_eq!(opposite_view("symbolic"), Some("signal"));
        assert_eq!(opposite_view("signal"), Some("symbolic"));
        assert_eq!(opposite_view("human"), None);
    }

    #[test]
    fn promote_emits_opposite_view_pseudo_labels() {
        let run = RunId::new();
        let u1 = Uuid::new_v4();
        let units = vec![unit(u1, run)];
        let em = vote(u1, run, "symbolic", "claimed_stem_lf", "acappella");
        let emissions = vec![em.clone()];
        let mut gold = IndexMap::new();
        gold.insert(u1, "acappella".into());
        let bundle = AlgorithmBundle::from_source_specs([em.source_spec_id]);
        let outcome = run_cotraining_round(RoundRequest {
            units: &units,
            emissions: &emissions,
            gold_by_unit: &gold,
            canaries: &[CanarySpec {
                source_name: "claimed_stem_lf".into(),
                known_accuracy: 1.0,
                tolerance: 0.05,
            }],
            rule: &DecisionRuleSpec::default(),
            bundle: &bundle,
            parent: None,
            prior_pseudo_labels: &[],
            input_snapshot_id: SnapshotId::new(),
        })
        .unwrap();
        assert_eq!(outcome.reason, "promoted");
        assert_eq!(outcome.round.status, RoundStatus::Promoted);
        assert_eq!(outcome.new_pseudo_labels.len(), 1);
        assert_eq!(outcome.new_pseudo_labels[0].target_view, "signal");
        assert_eq!(outcome.new_pseudo_labels[0].eligible_from_round_index, 1);
    }

    #[test]
    fn view_leakage_rejected() {
        let run = RunId::new();
        let u1 = Uuid::new_v4();
        let em = vote(u1, run, "symbolic", "claimed_stem_lf", "acappella");
        let parent_round = CoTrainingRound {
            round_id: RoundId::new(),
            round_index: 0,
            parent_round_id: None,
            input_snapshot_id: SnapshotId::new(),
            algorithm_bundle_id: Uuid::new_v4(),
            fitted_model_ids: IndexMap::new(),
            prediction_snapshot_id: None,
            status: RoundStatus::Promoted,
        };
        let label = PseudoLabel {
            pseudo_label_id: Uuid::new_v4(),
            produced_in_round_id: parent_round.round_id,
            unit_id: u1,
            axis: Axis::Identity,
            value: serde_json::json!("acappella"),
            source_belief_id: BeliefId::new(),
            source_prediction_run_id: run,
            source_fitted_model_id: ModelId::new(),
            contributing_emission_ids: vec![em.emission_id],
            confidence_at_acceptance: 1.0,
            entropy_at_acceptance: 0.0,
            // Wrong: trying to feed symbolic back into symbolic.
            target_view: "symbolic".into(),
            eligible_from_round_index: 1,
            revoked_at: None,
        };
        let mut child = parent_round.clone();
        child.round_id = RoundId::new();
        child.round_index = 1;
        child.parent_round_id = Some(parent_round.round_id);
        child.status = RoundStatus::Candidate;

        let index = EmissionIndex::from_emissions(&[em]);
        let err = validate_pseudo_label_for_next_round(&label, &child, "symbolic", &index);
        assert!(matches!(err, Err(KernelError::ViewLeakage(_))));
    }

    #[test]
    fn canary_fail_rejects_without_pseudo_labels() {
        let run = RunId::new();
        let u1 = Uuid::new_v4();
        let units = vec![unit(u1, run)];
        // Vote disagrees with gold → canary accuracy 0.
        let em = vote(u1, run, "symbolic", "claimed_stem_lf", "wrong");
        let emissions = vec![em.clone()];
        let mut gold = IndexMap::new();
        gold.insert(u1, "acappella".into());
        let bundle = AlgorithmBundle::from_source_specs([em.source_spec_id]);
        let outcome = run_cotraining_round(RoundRequest {
            units: &units,
            emissions: &emissions,
            gold_by_unit: &gold,
            canaries: &[CanarySpec {
                source_name: "claimed_stem_lf".into(),
                known_accuracy: 1.0,
                tolerance: 0.05,
            }],
            rule: &DecisionRuleSpec::default(),
            bundle: &bundle,
            parent: None,
            prior_pseudo_labels: &[],
            input_snapshot_id: SnapshotId::new(),
        })
        .unwrap();
        assert_eq!(outcome.reason, "round_rejected");
        assert_eq!(outcome.round.status, RoundStatus::Rejected);
        assert!(outcome.new_pseudo_labels.is_empty());
    }
}
