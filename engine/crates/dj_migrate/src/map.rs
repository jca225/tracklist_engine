//! Map legacy inventory → observation / evidence rewrite plan.
//!
//! Does NOT mint [`dj_kernel::RecordingId`] from source keys. Emits evidence
//! bundles that may later feed `resolve_recording` once substantive (content)
//! atoms exist — locator-only kinds never mint (law `identity.no_locator`).
//!
//! Missing crawl fields become `status=abstained`, not `explicit_null`, so the
//! future LF matrix does not treat crawl gaps as positive null votes.

use crate::legacy::InventoryReport;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

fn simple_hash(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))[..16].to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectPlan {
    pub subject_type: String,
    pub provisional_id: String,
    pub legacy_keys: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationPlan {
    pub subject: SubjectPlan,
    pub predicate: String,
    pub value: Option<serde_json::Value>,
    pub status: String,
    pub legacy_source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceBundlePlan {
    /// Stable key for the evidence bundle — NOT a RecordingId.
    pub bundle_id: String,
    pub legacy_recording_id: Option<String>,
    pub legacy_track_id: Option<String>,
    pub evidence_kinds: Vec<String>,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanAssertionPlan {
    pub set_id: String,
    pub label: String,
    pub field: String,
    pub observed_value: Option<serde_json::Value>,
    pub uncertainty_family: String,
    pub uncertainty_params: serde_json::Value,
    pub legacy_track_id: Option<String>,
    pub abstained: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrosswalkEntry {
    pub kind: String,
    pub legacy: String,
    pub provisional: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapPlan {
    pub inventory_source: String,
    pub set_ids: Vec<String>,
    pub crosswalk: Vec<CrosswalkEntry>,
    pub observations: Vec<ObservationPlan>,
    pub evidence_bundles: Vec<EvidenceBundlePlan>,
    pub human_assertions: Vec<HumanAssertionPlan>,
    pub audio_staging: Vec<AudioStagingPlan>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioStagingPlan {
    pub legacy_path: Option<String>,
    pub sha256: Option<String>,
    pub recording_id: Option<String>,
    pub track_id: Option<String>,
    pub media_type: String,
    pub kind: String,
}

pub fn run(inventory_path: &Path) -> Result<MapPlan> {
    let text = fs::read_to_string(inventory_path)
        .with_context(|| format!("read {}", inventory_path.display()))?;
    let inv: InventoryReport = serde_json::from_str(&text)?;
    Ok(map_inventory(&inv))
}

pub fn map_inventory(inv: &InventoryReport) -> MapPlan {
    let mut plan = MapPlan {
        inventory_source: inv.source.clone(),
        set_ids: inv.set_ids.clone(),
        crosswalk: vec![],
        observations: vec![],
        evidence_bundles: vec![],
        human_assertions: vec![],
        audio_staging: vec![],
        notes: vec![
            "RecordingId is NOT minted here. evidence_bundles feed resolve_recording.".into(),
            "Legacy track_id / recording_id are locators, not canonical identity.".into(),
        ],
    };

    for set in &inv.sets {
        let provisional = format!("set:{}", set.set_id);
        plan.crosswalk.push(CrosswalkEntry {
            kind: "set".into(),
            legacy: set.set_id.clone(),
            provisional: provisional.clone(),
        });
    }

    for slot in &inv.slots {
        let provisional_slot = format!("slot:{}:{}", slot.set_id, slot.row_index);
        plan.crosswalk.push(CrosswalkEntry {
            kind: "slot".into(),
            legacy: format!("{}:{}", slot.set_id, slot.row_index),
            provisional: provisional_slot.clone(),
        });

        let subject = SubjectPlan {
            subject_type: "set-slot".into(),
            provisional_id: provisional_slot,
            legacy_keys: [
                Some(format!("set_id={}", slot.set_id)),
                Some(format!("row_index={}", slot.row_index)),
                slot.recording_id
                    .as_ref()
                    .map(|id| format!("recording_id={id}")),
                slot.track_id.as_ref().map(|id| format!("track_id={id}")),
            ]
            .into_iter()
            .flatten()
            .collect(),
        };

        let preds: [(&str, Option<serde_json::Value>); 8] = [
            (
                "source_track_key",
                slot.track_id
                    .as_ref()
                    .or(slot.recording_id.as_ref())
                    .map(|v| serde_json::Value::String(v.clone())),
            ),
            (
                "claimed_title",
                slot.claimed_title
                    .as_ref()
                    .map(|v| serde_json::Value::String(v.clone())),
            ),
            (
                "claimed_artists",
                slot.claimed_artists
                    .as_ref()
                    .map(|v| serde_json::Value::String(v.clone())),
            ),
            (
                "claimed_version",
                slot.claimed_version
                    .as_ref()
                    .map(|v| serde_json::Value::String(v.clone())),
            ),
            (
                "claimed_stem",
                slot.claimed_stem
                    .as_ref()
                    .map(|v| serde_json::Value::String(v.clone())),
            ),
            (
                "claimed_variant",
                slot.claimed_variant
                    .as_ref()
                    .map(|v| serde_json::Value::String(v.clone())),
            ),
            (
                "cue_seconds",
                slot.cue_seconds.map(|v| serde_json::json!(v)),
            ),
            (
                "slot_label",
                slot.slot_label
                    .as_ref()
                    .map(|v| serde_json::Value::String(v.clone())),
            ),
        ];

        for (predicate, value) in preds {
            // Missing crawl fields are abstentions, not asserted nulls (PWS matrix).
            let status = if value.is_none() {
                "abstained"
            } else {
                "observed"
            };
            plan.observations.push(ObservationPlan {
                subject: subject.clone(),
                predicate: predicate.into(),
                value,
                status: status.into(),
                legacy_source: "set_track_slots".into(),
            });
        }

        if slot.recording_id.is_some() || slot.track_id.is_some() {
            let stable = format!(
                "{}:{}:{}:{}",
                slot.set_id,
                slot.row_index,
                slot.recording_id.as_deref().unwrap_or(""),
                slot.track_id.as_deref().unwrap_or("")
            );
            plan.evidence_bundles.push(EvidenceBundlePlan {
                bundle_id: format!("bundle-{}", simple_hash(&stable)),
                legacy_recording_id: slot.recording_id.clone(),
                legacy_track_id: slot.track_id.clone(),
                evidence_kinds: vec!["legacy_recording_row".into(), "slot_claim".into()],
                note:
                    "Locator atoms only — resolve_recording requires substantive content evidence"
                        .into(),
            });
        }
    }

    for gt in &inv.ground_truth {
        let identity_abstain = gt
            .id_source
            .as_deref()
            .map(|s| s.eq_ignore_ascii_case("abstain"))
            .unwrap_or(false);
        let fields = [
            (
                "mix_start_s",
                gt.set_start_s.map(|v| serde_json::json!(v)),
                "student_t_error",
                false,
            ),
            (
                "mix_end_s",
                gt.set_end_s.map(|v| serde_json::json!(v)),
                "student_t_error",
                false,
            ),
            (
                "ref_start_s",
                gt.ref_start_s.map(|v| serde_json::json!(v)),
                "student_t_error",
                false,
            ),
            (
                "ref_end_s",
                gt.ref_end_s.map(|v| serde_json::json!(v)),
                "student_t_error",
                false,
            ),
            (
                "tempo_ratio",
                gt.tempo_ratio.map(|v| serde_json::json!(v)),
                "student_t_error",
                false,
            ),
            (
                "claimed_stem",
                gt.claimed_stem
                    .as_ref()
                    .map(|v| serde_json::Value::String(v.clone())),
                "categorical_prior",
                false,
            ),
            (
                "legacy_track_id",
                gt.track_id
                    .as_ref()
                    .map(|v| serde_json::Value::String(v.clone())),
                "categorical_prior",
                true, // identity field — respect id_source abstain
            ),
        ];
        for (field, value, family, is_identity) in fields {
            let abstained = (is_identity && identity_abstain) || value.is_none();
            let params = match family {
                "student_t_error" => serde_json::json!({"scale_s": 0.25, "df": 4}),
                _ => serde_json::json!({"mean": 0.99, "effective_sample_size": 20}),
            };
            plan.human_assertions.push(HumanAssertionPlan {
                set_id: gt.set_id.clone(),
                label: gt.label.clone(),
                field: field.into(),
                observed_value: if abstained { None } else { value },
                uncertainty_family: family.into(),
                uncertainty_params: params,
                legacy_track_id: gt.track_id.clone(),
                abstained,
            });
        }
    }

    for audio in &inv.audio {
        plan.audio_staging.push(AudioStagingPlan {
            legacy_path: audio.path.clone(),
            sha256: audio.sha256.clone(),
            recording_id: audio.recording_id.clone(),
            track_id: audio.track_id.clone(),
            media_type: "audio/mpeg".into(),
            kind: "audio".into(),
        });
        if audio.sha256.is_none() && audio.path.is_some() {
            plan.notes.push(format!(
                "audio missing sha256 (path-as-identity risk): {:?}",
                audio.path
            ));
        }
    }

    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legacy::{InventoryReport, LegacySlot};

    #[test]
    fn does_not_emit_recording_id_as_canonical() {
        let inv = InventoryReport {
            source: "test".into(),
            set_ids: vec!["2nvzlh2k".into()],
            sets: vec![],
            slots: vec![LegacySlot {
                set_id: "2nvzlh2k".into(),
                row_index: 1,
                recording_id: Some("abc".into()),
                track_id: Some("abc".into()),
                claimed_title: Some("Song".into()),
                claimed_artists: None,
                claimed_version: None,
                claimed_stem: Some("regular".into()),
                claimed_variant: None,
                cue_seconds: None,
                slot_label: Some("001".into()),
                layer_role: None,
            }],
            ground_truth: vec![],
            audio: vec![],
            notes: vec![],
        };
        let plan = map_inventory(&inv);
        assert!(!plan.evidence_bundles.is_empty());
        assert!(plan.notes.iter().any(|n| n.contains("NOT minted")));
        assert!(plan
            .observations
            .iter()
            .any(|o| o.predicate == "source_track_key"));
        assert!(plan
            .observations
            .iter()
            .any(|o| o.status == "abstained" && o.predicate == "claimed_artists"));
    }
}
