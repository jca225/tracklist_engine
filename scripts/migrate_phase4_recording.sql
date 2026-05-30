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

-- track_fingerprints: align column name with recording layer (if still track_id)
-- Ignore error if already recording_id.
ALTER TABLE track_fingerprints RENAME COLUMN track_id TO recording_id;

-- track_audio: add recording_id if missing (post identity rename)
-- SQLite lacks IF NOT EXISTS for columns — ignore errors if already applied.
ALTER TABLE track_audio ADD COLUMN recording_id TEXT;
UPDATE track_audio SET recording_id = track_id WHERE recording_id IS NULL;

-- set_track_slots: claim columns + recording_id
ALTER TABLE set_track_slots ADD COLUMN recording_id TEXT;
ALTER TABLE set_track_slots ADD COLUMN claimed_version TEXT;
ALTER TABLE set_track_slots ADD COLUMN claimed_stem TEXT DEFAULT 'regular';
ALTER TABLE set_track_slots ADD COLUMN claimed_variant TEXT DEFAULT 'regular';
UPDATE set_track_slots SET recording_id = track_id WHERE recording_id IS NULL;
UPDATE set_track_slots SET claimed_stem = 'regular' WHERE claimed_stem IS NULL;
UPDATE set_track_slots SET claimed_variant = 'regular' WHERE claimed_variant IS NULL;

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
