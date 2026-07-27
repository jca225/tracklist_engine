//! Programmatic weak-supervision (Snorkel-like) matrix types.
//!
//! # TEMP learning tour — THIS IS THE DECIDE CORE
//!
//! Flow you step in the debugger (`make canary-smoke` / `pws-fit` / `run-round`):
//!
//! ```text
//! Python LFs  →  EvidenceEmission rows (Propose)
//!                    ↓
//!            majority_vote_fit  →  AxisBeliefs (Decide)
//!                    ↓
//!            estimate_source_accuracies vs gold_by_unit
//!                    ↓
//!            gate_canaries  →  pass/fail (Promote gate)
//! ```
//!
//! These are distinct from claim-scoped [`crate::types::Evidence`]: an
//! [`EvidenceEmission`] is one labeling-function vote on one [`InferenceUnit`].
//! Abstention is first-class (`abstained: bool` is required, never optional).
//!
//! Sensors / LFs (Python) emit rows; the kernel fits a toy majority vote today
//! and will host generative label models later. Feature artifacts (embeddings)
//! should be referenced from LF context by `ArtifactId`, not by filesystem path
//! — see `docs/storage.md`.
//!
//! Grep `TEMP:` to strip pedagogy later.

use crate::error::KernelError;
use crate::ids::{BeliefId, ModelId, RunId};
use crate::types::{Axis, AxisBelief, Decision, Interval, NativeScore, ProcessSpec, SubjectRef};
use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// One cell being labeled — the PK of the weak-supervision matrix.
/// TEMP: think "one question we want answered" (e.g. stem for slot 3 on set X).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceUnit {
    pub unit_id: Uuid,
    pub set_id: String,
    pub axis: Axis,
    pub subject: SubjectRef,
    pub candidate: Option<SubjectRef>,
    pub mix_region: Option<Interval>,
    pub generated_by_run_id: RunId,
    pub generation_parameters_hash: String,
    /// train | development | tripwire
    pub split: String,
}

/// One labeling-function vote (long-form matrix row).
/// TEMP: Python writes these into `staging/vertical_lf_bundle.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceEmission {
    pub emission_id: Uuid,
    /// Which InferenceUnit this vote is about.
    pub unit_id: Uuid,
    pub source_spec_id: Uuid,
    /// e.g. claimed_stem_lf | title_stem_heuristic_lf
    pub source_name: String,
    /// e.g. symbolic | signal | vision — used for co-training opposite_view
    pub source_family: String,
    pub producer_run_id: RunId,
    pub value: Option<serde_json::Value>,
    /// EXPLICIT. Must always be set — never inferred from `value.is_none()` alone
    /// (a deliberate null vote is not an abstention).
    pub abstained: bool,
    pub native_score: NativeScore,
    /// Optional ArtifactIds cited as supporting bytes (features, images, …).
    pub evidence_ids: Vec<Uuid>,
}

impl EvidenceEmission {
    /// TEMP: law of abstain — abstained XOR value; never both, never neither.
    pub fn validate(&self) -> Result<(), KernelError> {
        if self.abstained && self.value.is_some() {
            return Err(KernelError::Other(
                "abstained emission must not carry a value".into(),
            ));
        }
        if !self.abstained && self.value.is_none() {
            return Err(KernelError::Other(
                "non-abstaining emission requires a value (use abstained=true to skip)".into(),
            ));
        }
        if self.source_family.is_empty() || self.source_name.is_empty() {
            return Err(KernelError::Other(
                "emission requires source_name and source_family".into(),
            ));
        }
        Ok(())
    }

    pub fn abstain(
        unit_id: Uuid,
        source: &dyn EvidenceSource,
        run_id: RunId,
        _reason: impl Into<String>,
    ) -> Self {
        Self {
            emission_id: Uuid::new_v4(),
            unit_id,
            source_spec_id: source.process_spec().process_spec_id,
            source_name: source.name().to_string(),
            source_family: source.source_family().to_string(),
            producer_run_id: run_id,
            value: None,
            abstained: true,
            native_score: NativeScore {
                family: source.source_family().to_string(),
                value: 0.0,
                scale: "abstain".into(),
                support: None,
            },
            evidence_ids: vec![],
        }
    }
}

/// Labeling-function interface (Snorkel LF / EvidenceSource).
/// TEMP: in-process LFs; Python LFs usually skip this and emit JSON instead.
pub trait EvidenceSource {
    fn name(&self) -> &str;
    fn source_family(&self) -> &str;
    fn process_spec(&self) -> &ProcessSpec;

    fn evaluate(
        &self,
        unit: &InferenceUnit,
        context: &EvidenceContext,
    ) -> Result<EvidenceEmission, KernelError>;
}

#[derive(Debug, Clone, Default)]
pub struct EvidenceContext {
    pub observations: Vec<serde_json::Value>,
    pub metadata: IndexMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRuleSpec {
    pub minimum_posterior: f64,
    pub maximum_entropy: f64,
    pub reject_below_posterior: Option<f64>,
    pub quarantine_on: Vec<String>,
}

impl Default for DecisionRuleSpec {
    fn default() -> Self {
        Self {
            minimum_posterior: 0.55,
            maximum_entropy: 1.5,
            reject_below_posterior: Some(0.15),
            quarantine_on: vec![],
        }
    }
}

/// Map posterior → Accepted/Rejected/Unresolved/Quarantined (pseudocode §9).
/// TEMP: thresholds live on DecisionRuleSpec (min posterior, max entropy).
pub fn decide(
    posterior: &IndexMap<String, f64>,
    entropy: f64,
    rule: &DecisionRuleSpec,
    diagnostics: &[String],
) -> Decision {
    // TEMP: hard quarantine if any diagnostic code is on the deny-list.
    if diagnostics.iter().any(|c| rule.quarantine_on.contains(c)) {
        return Decision::Quarantined;
    }
    // TEMP: pick the MAP label (highest posterior mass). Empty → Unresolved.
    let Some((_, probability)) = posterior
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return Decision::Unresolved;
    };
    // TEMP: floor — too weak to even count as a contested rejection path.
    if let Some(floor) = rule.reject_below_posterior {
        if *probability < floor {
            return Decision::Rejected;
        }
    }
    // TEMP: not confident enough OR too messy (high entropy) → abstain as Unresolved.
    if *probability < rule.minimum_posterior || entropy > rule.maximum_entropy {
        return Decision::Unresolved;
    }
    // TEMP: clear winner → Accepted (eligible for Act / pseudo-labels later).
    Decision::Accepted
}

/// Toy generative stand-in: majority vote over non-abstaining emissions.
///
/// Real Snorkel/FlyingSquid fit comes later; this proves the matrix → belief path.
///
/// TEMP walkthrough:
/// 1. validate every emission
/// 2. per unit: count non-abstaining votes by label string
/// 3. normalize counts → posterior; empty → `__abstain__`
/// 4. entropy + decide() → AxisBelief
pub fn majority_vote_fit(
    units: &[InferenceUnit],
    emissions: &[EvidenceEmission],
    inference_run_id: RunId,
    fitted_model_id: ModelId,
    rule: &DecisionRuleSpec,
) -> Result<Vec<AxisBelief>, KernelError> {
    // TEMP: fail fast if any LF row violates abstain XOR value.
    for e in emissions {
        e.validate()?;
    }
    let mut beliefs = Vec::new();
    // TEMP: one belief per InferenceUnit (matrix column → posterior).
    for unit in units {
        // TEMP: only counting votes — abstentions do not move the posterior.
        let votes: Vec<&EvidenceEmission> = emissions
            .iter()
            .filter(|e| e.unit_id == unit.unit_id && !e.abstained)
            .collect();
        // TEMP: provenance list includes abstains so we know who looked at this unit.
        let contributing: Vec<Uuid> = emissions
            .iter()
            .filter(|e| e.unit_id == unit.unit_id)
            .map(|e| e.emission_id)
            .collect();

        // TEMP: tally label strings (e.g. "acappella" / "instrumental").
        let mut counts: IndexMap<String, f64> = IndexMap::new();
        for v in &votes {
            let key = match &v.value {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => continue, // should not happen after validate, but skip
            };
            *counts.entry(key).or_insert(0.0) += 1.0;
        }

        // TEMP: convert counts → probabilities (toy generative model).
        let total: f64 = counts.values().sum::<f64>().max(1.0);
        let mut posterior = IndexMap::new();
        if counts.is_empty() {
            // TEMP: everyone abstained → synthetic mass on __abstain__.
            posterior.insert("__abstain__".into(), 1.0);
        } else {
            for (k, c) in counts {
                posterior.insert(k, c / total);
            }
        }

        // TEMP: how messy is this posterior? feeds decide().
        let entropy = shannon_entropy(&posterior);
        let decision = decide(&posterior, entropy, rule, &[]);
        beliefs.push(AxisBelief {
            belief_id: BeliefId::new(),
            unit_id: unit.unit_id,
            axis: unit.axis,
            inference_run_id,
            fitted_model_id,
            posterior,
            entropy,
            decision,
            decision_rule_id: Uuid::nil(), // TEMP: placeholder until rules are versioned
            contributing_emission_ids: contributing,
            supersedes_belief_id: None,
        });
    }
    Ok(beliefs)
}

/// TEMP: uncertainty of the posterior; high entropy ⇒ Unresolved more often.
fn shannon_entropy(posterior: &IndexMap<String, f64>) -> f64 {
    // TEMP: H = -Σ p ln p  (natural log; scale is relative for our thresholds).
    posterior
        .values()
        .filter(|p| **p > 0.0)
        .map(|p| -p * p.ln())
        .sum()
}

/// Canary gate: a named LF must recover `known_accuracy` within `tolerance`.
/// TEMP: today canary source = `title_stem_heuristic_lf`, NOT `claimed_stem_lf`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanarySpec {
    pub source_name: String,
    pub known_accuracy: f64,
    pub tolerance: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceAccuracy {
    pub source_name: String,
    pub est_accuracy: f64,
    pub coverage: f64,
    pub n_votes: usize,
    pub n_abstain: usize,
}

/// Score each LF against held-out gold (only units present in `gold_by_unit`).
/// TEMP: gold comes from `canary_stem_gold.json` — never regenerate from LF votes.
pub fn estimate_source_accuracies(
    emissions: &[EvidenceEmission],
    gold_by_unit: &IndexMap<Uuid, String>,
) -> Result<Vec<SourceAccuracy>, KernelError> {
    // TEMP: bucket every emission by LF name (claimed_stem_lf, title_stem_…, …).
    let mut by_source: IndexMap<String, Vec<&EvidenceEmission>> = IndexMap::new();
    for e in emissions {
        e.validate()?;
        by_source.entry(e.source_name.clone()).or_default().push(e);
    }
    let mut out = Vec::new();
    for (name, rows) in by_source {
        let n_abstain = rows.iter().filter(|e| e.abstained).count();
        let votes: Vec<_> = rows.iter().filter(|e| !e.abstained).copied().collect();
        let n_votes = votes.len();
        let mut correct = 0usize;
        let mut scored = 0usize;
        // TEMP: only score units that have held-out gold (canary slots).
        for v in votes {
            if let Some(gold) = gold_by_unit.get(&v.unit_id) {
                scored += 1;
                if let Some(serde_json::Value::String(s)) = &v.value {
                    // TEMP: string equality on stem label today.
                    if s == gold {
                        correct += 1;
                    }
                }
            }
        }
        // TEMP: est_accuracy = correct/scored on gold units only (not full corpus).
        let est = if scored == 0 {
            0.0
        } else {
            correct as f64 / scored as f64
        };
        // TEMP: coverage = how often this LF voted vs abstained.
        let coverage = if rows.is_empty() {
            0.0
        } else {
            n_votes as f64 / rows.len() as f64
        };
        out.push(SourceAccuracy {
            source_name: name,
            est_accuracy: est,
            coverage,
            n_votes,
            n_abstain,
        });
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateResult {
    pub gate: String,
    pub passed: bool,
    pub detail: String,
}

/// Promote gate: each canary source's estimated accuracy must match held-out
/// `known_accuracy` within `tolerance`. Fail-closed if a canary source is missing.
///
/// TEMP: this is the YES/NO that decides whether Act may write pseudo-labels.
pub fn gate_canaries(accuracies: &[SourceAccuracy], canaries: &[CanarySpec]) -> GateResult {
    // TEMP: every CanarySpec must be present and within tolerance — no soft fail.
    for c in canaries {
        let Some(acc) = accuracies.iter().find(|a| a.source_name == c.source_name) else {
            // TEMP: LF missing from bundle ⇒ cannot trust the round.
            return GateResult {
                gate: "canary".into(),
                passed: false,
                detail: format!("missing canary source {}", c.source_name),
            };
        };
        // TEMP: |measured − expected| must be ≤ tolerance (e.g. 1.0 ± 0.05).
        if (acc.est_accuracy - c.known_accuracy).abs() > c.tolerance {
            return GateResult {
                gate: "canary".into(),
                passed: false,
                detail: format!(
                    "canary {}: est {:.3} != known {:.3} (tol {})",
                    c.source_name, acc.est_accuracy, c.known_accuracy, c.tolerance
                ),
            };
        }
    }
    GateResult {
        gate: "canary".into(),
        passed: true,
        detail: format!("{} canary source(s) recovered", canaries.len()),
    }
}

/// Emit all sources over all units; abstentions are persisted (never dropped).
/// TEMP: in-process path; canary smoke uses Python JSON instead.
pub fn emit_all_sources(
    units: &[InferenceUnit],
    sources: &[&dyn EvidenceSource],
    context: &EvidenceContext,
    run_id: RunId,
) -> Result<Vec<EvidenceEmission>, KernelError> {
    let mut emissions = Vec::new();
    for source in sources {
        for unit in units {
            let emission = match source.evaluate(unit, context) {
                Ok(e) => e,
                Err(e) => EvidenceEmission::abstain(unit.unit_id, *source, run_id, e.to_string()),
            };
            emission.validate()?;
            emissions.push(emission);
        }
    }
    Ok(emissions)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerticalSliceReport {
    pub created_at: DateTime<Utc>,
    pub n_units: usize,
    pub n_emissions: usize,
    pub n_abstained: usize,
    pub beliefs: Vec<AxisBelief>,
    pub accuracies: Vec<SourceAccuracy>,
    pub gate: GateResult,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::ArtifactId;
    use crate::types::implementation_hash;

    struct StemLf {
        spec: ProcessSpec,
        votes: IndexMap<Uuid, Option<String>>,
    }

    impl EvidenceSource for StemLf {
        fn name(&self) -> &str {
            "claimed_stem_lf"
        }
        fn source_family(&self) -> &str {
            "symbolic"
        }
        fn process_spec(&self) -> &ProcessSpec {
            &self.spec
        }
        fn evaluate(
            &self,
            unit: &InferenceUnit,
            _ctx: &EvidenceContext,
        ) -> Result<EvidenceEmission, KernelError> {
            match self.votes.get(&unit.unit_id) {
                Some(Some(v)) => Ok(EvidenceEmission {
                    emission_id: Uuid::new_v4(),
                    unit_id: unit.unit_id,
                    source_spec_id: self.spec.process_spec_id,
                    source_name: self.name().into(),
                    source_family: self.source_family().into(),
                    producer_run_id: unit.generated_by_run_id,
                    value: Some(serde_json::Value::String(v.clone())),
                    abstained: false,
                    native_score: NativeScore {
                        family: "symbolic".into(),
                        value: 1.0,
                        scale: "count".into(),
                        support: Some(1),
                    },
                    evidence_ids: vec![],
                }),
                _ => Ok(EvidenceEmission::abstain(
                    unit.unit_id,
                    self,
                    unit.generated_by_run_id,
                    "no claim",
                )),
            }
        }
    }

    fn spec() -> ProcessSpec {
        ProcessSpec {
            process_spec_id: Uuid::new_v4(),
            name: "claimed_stem_lf".into(),
            version: "0.1.0".into(),
            stage: "evidence".into(),
            source_code_artifact_id: ArtifactId::new(),
            code_commit: "test".into(),
            entrypoint: "claimed_stem_lf".into(),
            parameter_set_id: Uuid::new_v4(),
            environment_spec_id: Uuid::new_v4(),
            dependency_lock_artifact_id: ArtifactId::new(),
            model_artifact_ids: vec![],
            implementation_hash: implementation_hash("claimed_stem_lf"),
        }
    }

    #[test]
    fn majority_vote_and_canary_gate() {
        let run = RunId::new();
        let u1 = Uuid::new_v4();
        let u2 = Uuid::new_v4();
        let units = vec![
            InferenceUnit {
                unit_id: u1,
                set_id: "2nvzlh2k".into(),
                axis: Axis::Identity,
                subject: SubjectRef {
                    subject_type: "set-slot".into(),
                    subject_id: Uuid::new_v4(),
                },
                candidate: None,
                mix_region: None,
                generated_by_run_id: run,
                generation_parameters_hash: "x".into(),
                split: "train".into(),
            },
            InferenceUnit {
                unit_id: u2,
                set_id: "2nvzlh2k".into(),
                axis: Axis::Identity,
                subject: SubjectRef {
                    subject_type: "set-slot".into(),
                    subject_id: Uuid::new_v4(),
                },
                candidate: None,
                mix_region: None,
                generated_by_run_id: run,
                generation_parameters_hash: "x".into(),
                split: "train".into(),
            },
        ];
        let mut votes = IndexMap::new();
        votes.insert(u1, Some("acappella".into()));
        votes.insert(u2, None);
        let lf = StemLf {
            spec: spec(),
            votes,
        };
        let emissions = emit_all_sources(&units, &[&lf], &EvidenceContext::default(), run).unwrap();
        assert_eq!(emissions.len(), 2);
        assert!(emissions.iter().any(|e| e.abstained));
        assert!(emissions.iter().all(|e| e.validate().is_ok()));

        let beliefs = majority_vote_fit(
            &units,
            &emissions,
            run,
            ModelId::new(),
            &DecisionRuleSpec::default(),
        )
        .unwrap();
        assert_eq!(beliefs.len(), 2);

        let mut gold = IndexMap::new();
        gold.insert(u1, "acappella".into());
        let acc = estimate_source_accuracies(&emissions, &gold).unwrap();
        let gate = gate_canaries(
            &acc,
            &[CanarySpec {
                source_name: "claimed_stem_lf".into(),
                known_accuracy: 1.0,
                tolerance: 0.05,
            }],
        );
        assert!(gate.passed, "{}", gate.detail);
    }

    #[test]
    fn emission_rejects_abstain_with_value() {
        let e = EvidenceEmission {
            emission_id: Uuid::new_v4(),
            unit_id: Uuid::new_v4(),
            source_spec_id: Uuid::new_v4(),
            source_name: "x".into(),
            source_family: "symbolic".into(),
            producer_run_id: RunId::new(),
            value: Some(serde_json::json!("a")),
            abstained: true,
            native_score: NativeScore {
                family: "symbolic".into(),
                value: 0.0,
                scale: "abstain".into(),
                support: None,
            },
            evidence_ids: vec![],
        };
        assert!(e.validate().is_err());
    }
}
