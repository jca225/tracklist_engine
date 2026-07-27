//! Read-only inventory of legacy SQLite or gold fixtures.
//!
//! Never mutates `music_database.db`. Prefer `--fixtures-only` for CI; set
//! `LEGACY_DB_PATH` for a live snapshot on pi / SSH mount.
//!
//! TEMP: fixtures-only inventory is the place-in DB for migrate dry-run (BB11/BB12).

use crate::legacy::{InventoryReport, LegacyAudio, LegacyGroundTruth, LegacySet, LegacySlot};
use anyhow::{bail, Context, Result};
use rusqlite::{params_from_iter, Connection};
use serde::Deserialize;
use std::fs;
use std::path::Path;

pub fn run(
    db: Option<&Path>,
    fixtures: &Path,
    set_ids: &[String],
    fixtures_only: bool,
) -> Result<InventoryReport> {
    if fixtures_only || db.is_none() {
        return from_fixtures(fixtures, set_ids);
    }
    let db = db.expect("db checked");
    if !db.exists() {
        bail!(
            "LEGACY_DB_PATH / --db does not exist: {}. Use --fixtures-only or fixtures/gold.",
            db.display()
        );
    }
    from_sqlite(db, set_ids)
}

fn from_fixtures(fixtures: &Path, set_ids: &[String]) -> Result<InventoryReport> {
    let mut report = InventoryReport {
        source: format!("fixtures:{}", fixtures.display()),
        set_ids: set_ids.to_vec(),
        sets: vec![],
        slots: vec![],
        ground_truth: vec![],
        audio: vec![],
        notes: vec![
            "Fixture-backed inventory — no live DB. Set LEGACY_DB_PATH for full inventory.".into(),
        ],
    };

    let inventory_path = fixtures.join("inventory.json");
    if inventory_path.exists() {
        let text = fs::read_to_string(&inventory_path)
            .with_context(|| format!("read {}", inventory_path.display()))?;
        let mut loaded: InventoryReport = serde_json::from_str(&text)?;
        loaded.set_ids = set_ids.to_vec();
        loaded.source = format!("fixtures:{}", fixtures.display());
        filter_report(&mut loaded, set_ids);
        return Ok(loaded);
    }

    for set_id in set_ids {
        let yaml_path = fixtures.join(format!("{set_id}_ground_truth.yaml"));
        if yaml_path.exists() {
            let gt = load_gt_yaml(&yaml_path, set_id)?;
            report.sets.push(LegacySet {
                set_id: set_id.clone(),
                title: None,
                artists: None,
            });
            report.ground_truth.extend(gt);
        }
        let slots_path = fixtures.join(format!("{set_id}_slots.json"));
        if slots_path.exists() {
            let text = fs::read_to_string(&slots_path)?;
            let slots: Vec<LegacySlot> = serde_json::from_str(&text)?;
            report.slots.extend(slots);
        }
        let audio_path = fixtures.join(format!("{set_id}_audio.json"));
        if audio_path.exists() {
            let text = fs::read_to_string(&audio_path)?;
            let audio: Vec<LegacyAudio> = serde_json::from_str(&text)?;
            report.audio.extend(audio);
        }
    }

    if report.ground_truth.is_empty() && report.slots.is_empty() {
        bail!(
            "no gold fixtures found under {}. Expected inventory.json or {{set}}_ground_truth.yaml",
            fixtures.display()
        );
    }
    Ok(report)
}

#[derive(Debug, Deserialize)]
struct GtYaml {
    source: Option<String>,
    tracks: Vec<GtTrack>,
}

#[derive(Debug, Deserialize)]
struct GtTrack {
    track: Option<String>,
    slot_label: Option<String>,
    track_id: Option<String>,
    claimed_stem: Option<String>,
    set_start_s: Option<f64>,
    set_end_s: Option<f64>,
    ref_start_s: Option<f64>,
    ref_end_s: Option<f64>,
    tempo_ratio: Option<f64>,
    pitch_shift_semi: Option<f64>,
    id_source: Option<String>,
}

fn load_gt_yaml(path: &Path, set_id: &str) -> Result<Vec<LegacyGroundTruth>> {
    let text = fs::read_to_string(path)?;
    let doc: GtYaml = serde_yaml::from_str(&text)?;
    Ok(doc
        .tracks
        .into_iter()
        .map(|t| LegacyGroundTruth {
            set_id: set_id.to_string(),
            label: t
                .slot_label
                .clone()
                .or(t.track.clone())
                .unwrap_or_else(|| "unknown".into()),
            track_id: t.track_id,
            claimed_stem: t.claimed_stem,
            set_start_s: t.set_start_s,
            set_end_s: t.set_end_s,
            ref_start_s: t.ref_start_s,
            ref_end_s: t.ref_end_s,
            tempo_ratio: t.tempo_ratio,
            pitch_shift_semi: t.pitch_shift_semi,
            source: doc.source.clone(),
            id_source: t.id_source,
        })
        .collect())
}

fn filter_report(report: &mut InventoryReport, set_ids: &[String]) {
    let want: std::collections::HashSet<_> = set_ids.iter().cloned().collect();
    report.sets.retain(|s| want.contains(&s.set_id));
    report.slots.retain(|s| want.contains(&s.set_id));
    report.ground_truth.retain(|g| want.contains(&g.set_id));
}

fn from_sqlite(db: &Path, set_ids: &[String]) -> Result<InventoryReport> {
    let conn = Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("open {}", db.display()))?;

    let placeholders = set_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");

    let mut report = InventoryReport {
        source: format!("sqlite:{}", db.display()),
        set_ids: set_ids.to_vec(),
        sets: vec![],
        slots: vec![],
        ground_truth: vec![],
        audio: vec![],
        notes: vec!["Read-only inventory; no writes to legacy DB.".into()],
    };

    // dj_sets
    {
        let sql =
            format!("SELECT set_id, title, artists FROM dj_sets WHERE set_id IN ({placeholders})");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(set_ids.iter()), |row| {
            Ok(LegacySet {
                set_id: row.get(0)?,
                title: row.get(1)?,
                artists: row.get(2)?,
            })
        })?;
        for row in rows {
            report.sets.push(row?);
        }
    }

    // set_track_slots — columns may vary; select defensively
    if table_exists(&conn, "set_track_slots")? {
        let sql = format!(
            "SELECT set_id, row_index, recording_id, track_id,
                    claimed_version, claimed_stem, claimed_variant,
                    cue_seconds, slot_label, layer_role
             FROM set_track_slots WHERE set_id IN ({placeholders})"
        );
        match conn.prepare(&sql) {
            Ok(mut stmt) => {
                let rows = stmt.query_map(params_from_iter(set_ids.iter()), |row| {
                    Ok(LegacySlot {
                        set_id: row.get(0)?,
                        row_index: row.get(1)?,
                        recording_id: row.get(2)?,
                        track_id: row.get(3)?,
                        claimed_title: None,
                        claimed_artists: None,
                        claimed_version: row.get(4)?,
                        claimed_stem: row.get(5)?,
                        claimed_variant: row.get(6)?,
                        cue_seconds: row.get(7)?,
                        slot_label: row.get(8)?,
                        layer_role: row.get(9)?,
                    })
                })?;
                for row in rows {
                    report.slots.push(row?);
                }
            }
            Err(e) => report
                .notes
                .push(format!("set_track_slots query skipped: {e}")),
        }
    }

    if table_exists(&conn, "set_ground_truth")? {
        let sql = format!(
            "SELECT set_id, label, track_id, claimed_stem,
                    set_start_s, set_end_s, ref_start_s, ref_end_s,
                    tempo_ratio, pitch_shift_semi, source
             FROM set_ground_truth WHERE set_id IN ({placeholders})"
        );
        match conn.prepare(&sql) {
            Ok(mut stmt) => {
                let rows = stmt.query_map(params_from_iter(set_ids.iter()), |row| {
                    Ok(LegacyGroundTruth {
                        set_id: row.get(0)?,
                        label: row.get(1)?,
                        track_id: row.get(2)?,
                        claimed_stem: row.get(3)?,
                        set_start_s: row.get(4)?,
                        set_end_s: row.get(5)?,
                        ref_start_s: row.get(6)?,
                        ref_end_s: row.get(7)?,
                        tempo_ratio: row.get(8)?,
                        pitch_shift_semi: row.get(9)?,
                        source: row.get(10)?,
                        id_source: None,
                    })
                })?;
                for row in rows {
                    report.ground_truth.push(row?);
                }
            }
            Err(e) => report
                .notes
                .push(format!("set_ground_truth query skipped: {e}")),
        }
    }

    // Audio linked by recording_id / track_id from slots
    if table_exists(&conn, "track_audio")? {
        let ids: Vec<String> = report
            .slots
            .iter()
            .flat_map(|s| {
                [s.recording_id.clone(), s.track_id.clone()]
                    .into_iter()
                    .flatten()
            })
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        if !ids.is_empty() {
            let ph = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT recording_id, track_id, platform, player_id, path, sha256,
                        is_reference, stem, variant, duration_s
                 FROM track_audio
                 WHERE recording_id IN ({ph}) OR track_id IN ({ph})"
            );
            let mut params: Vec<String> = ids.clone();
            params.extend(ids);
            match conn.prepare(&sql) {
                Ok(mut stmt) => {
                    let rows = stmt.query_map(params_from_iter(params.iter()), |row| {
                        Ok(LegacyAudio {
                            recording_id: row.get(0)?,
                            track_id: row.get(1)?,
                            platform: row.get(2)?,
                            player_id: row.get(3)?,
                            path: row.get(4)?,
                            sha256: row.get(5)?,
                            is_reference: row.get(6)?,
                            stem: row.get(7)?,
                            variant: row.get(8)?,
                            duration_s: row.get(9)?,
                        })
                    })?;
                    for row in rows {
                        report.audio.push(row?);
                    }
                }
                Err(e) => report.notes.push(format!("track_audio query skipped: {e}")),
            }
        }
    }

    Ok(report)
}

fn table_exists(conn: &Connection, name: &str) -> Result<bool> {
    let n: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
        [name],
        |row| row.get(0),
    )?;
    Ok(n > 0)
}
