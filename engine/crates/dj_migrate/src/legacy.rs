//! Legacy row shapes from tracklist_engine (read-only).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacySet {
    pub set_id: String,
    pub title: Option<String>,
    pub artists: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacySlot {
    pub set_id: String,
    pub row_index: i64,
    pub recording_id: Option<String>,
    pub track_id: Option<String>,
    pub claimed_title: Option<String>,
    pub claimed_artists: Option<String>,
    pub claimed_version: Option<String>,
    pub claimed_stem: Option<String>,
    pub claimed_variant: Option<String>,
    pub cue_seconds: Option<f64>,
    pub slot_label: Option<String>,
    pub layer_role: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyGroundTruth {
    pub set_id: String,
    pub label: String,
    pub track_id: Option<String>,
    pub claimed_stem: Option<String>,
    pub set_start_s: Option<f64>,
    pub set_end_s: Option<f64>,
    pub ref_start_s: Option<f64>,
    pub ref_end_s: Option<f64>,
    pub tempo_ratio: Option<f64>,
    pub pitch_shift_semi: Option<f64>,
    pub source: Option<String>,
    pub id_source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyAudio {
    pub recording_id: Option<String>,
    pub track_id: Option<String>,
    pub platform: Option<String>,
    pub player_id: Option<String>,
    pub path: Option<String>,
    pub sha256: Option<String>,
    pub is_reference: Option<i64>,
    pub stem: Option<String>,
    pub variant: Option<String>,
    pub duration_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryReport {
    pub source: String,
    pub set_ids: Vec<String>,
    pub sets: Vec<LegacySet>,
    pub slots: Vec<LegacySlot>,
    pub ground_truth: Vec<LegacyGroundTruth>,
    pub audio: Vec<LegacyAudio>,
    pub notes: Vec<String>,
}
