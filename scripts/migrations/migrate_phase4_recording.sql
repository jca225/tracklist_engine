-- Phase 4: work + recording layer (run AFTER migrate_identity_axes.sql).
-- Backs up music_database.db first.

PRAGMA foreign_keys = OFF;

CREATE TABLE IF NOT EXISTS work (
    work_id        TEXT PRIMARY KEY,
    title          TEXT,
    artists_json   TEXT,
    full_name      TEXT,
    updated_at     DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS recording (
    recording_id    TEXT PRIMARY KEY,
    work_id         TEXT NOT NULL,
    version         TEXT NOT NULL DEFAULT 'original',
    version_artist  TEXT,
    stem            TEXT NOT NULL DEFAULT 'regular',
    variant         TEXT NOT NULL DEFAULT 'regular',
    title           TEXT,
    full_name       TEXT,
    duration_seconds INTEGER,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (work_id) REFERENCES work(work_id) ON DELETE CASCADE,
    UNIQUE (work_id, version, version_artist, stem, variant)
);

-- track_fingerprints: normalize stem axis (recording_id already on newer DBs)
UPDATE track_fingerprints SET stem = 'regular' WHERE stem IN ('original', 'full');

-- track_audio: backfill recording_id + normalize stem (column added in partial runs)
UPDATE track_audio SET recording_id = track_id WHERE recording_id IS NULL;
UPDATE track_audio SET stem = 'regular' WHERE stem IN ('original', 'full');

-- set_track_slots: create on DBs that predate the table, else add claim columns
CREATE TABLE IF NOT EXISTS set_track_slots (
    set_id            TEXT NOT NULL,
    row_index         INTEGER NOT NULL,
    tlp_id            INTEGER,
    recording_id      TEXT,
    track_id          TEXT NOT NULL,
    source            TEXT DEFAULT 'scraped',
    slot_label        TEXT,
    is_concurrent     INTEGER DEFAULT 0,
    cue_seconds       INTEGER,
    cue_time_seconds  INTEGER,
    claimed_version   TEXT,
    claimed_stem      TEXT NOT NULL DEFAULT 'regular',
    claimed_variant   TEXT NOT NULL DEFAULT 'regular',
    full_name         TEXT,
    title             TEXT,
    artists_json      TEXT,
    duration_seconds  INTEGER,
    parsed_at         DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (set_id, row_index),
    FOREIGN KEY (set_id) REFERENCES dj_sets(set_id) ON DELETE CASCADE,
    FOREIGN KEY (recording_id) REFERENCES recording(recording_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_set_track_slots_recording ON set_track_slots(recording_id);
CREATE INDEX IF NOT EXISTS idx_set_track_slots_track ON set_track_slots(track_id);

-- Backfill work + recording 1:1 from track_metadata / track_audio
INSERT OR IGNORE INTO work (work_id, title, artists_json, full_name)
SELECT track_id, title, artists_json, full_name FROM track_metadata;

INSERT OR IGNORE INTO recording (
    recording_id, work_id, version, stem, variant, title, full_name, duration_seconds
)
SELECT
    ta.track_id,
    ta.track_id,
    COALESCE(tm.version, 'original'),
    ta.stem,
    ta.variant,
    tm.title,
    tm.full_name,
    tm.duration_seconds
FROM track_audio ta
LEFT JOIN track_metadata tm ON tm.track_id = ta.track_id
GROUP BY ta.track_id;

CREATE TABLE IF NOT EXISTS set_ground_truth (
    set_id          TEXT NOT NULL,
    label           TEXT NOT NULL,
    recording_id    TEXT,
    claimed_stem    TEXT,
    set_start_s     REAL NOT NULL,
    set_end_s       REAL NOT NULL,
    ref_start_s     REAL NOT NULL,
    ref_end_s       REAL,
    is_loop         INTEGER DEFAULT 0,
    ref_segments_json TEXT,
    media_links_json  TEXT,
    source          TEXT DEFAULT 'yaml',
    annotated_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (set_id, label),
    FOREIGN KEY (set_id) REFERENCES dj_sets(set_id) ON DELETE CASCADE
);

DROP VIEW IF EXISTS identity_mismatch;
CREATE VIEW identity_mismatch AS
SELECT
    s.set_id, s.row_index, s.full_name,
    s.claimed_stem, r.stem AS canonical_stem,
    s.claimed_version, r.version AS canonical_version,
    s.recording_id
FROM set_track_slots s
JOIN recording r ON r.recording_id = s.recording_id
WHERE (s.claimed_stem IS NOT NULL AND s.claimed_stem != r.stem)
   OR (s.claimed_version IS NOT NULL AND s.claimed_version != r.version);

PRAGMA foreign_keys = ON;
