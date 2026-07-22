-- Add the 'recording' axis + detach/relink actions to track_audio_correction.
-- SQLite cannot ALTER a CHECK constraint, so rebuild the table (no FKs to worry about).
--
-- ** APPLY EXACTLY ONCE. ** This script is not idempotent and not safe to
-- re-run: a second run renames the already-migrated table to
-- track_audio_correction_old and re-copies it through an INSERT column list
-- that OMITS old_recording_id/new_recording_id (those columns did not exist
-- pre-migration), silently NULLing out any recording-axis data written since
-- the first run. Check the table's CHECK constraint (or a saved run-log)
-- before applying — if `axis IN (...,'recording')` already appears, do NOT
-- run this again.
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;

ALTER TABLE track_audio_correction RENAME TO track_audio_correction_old;

CREATE TABLE track_audio_correction (
    correction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id              TEXT,
    position            TEXT,
    track_id            TEXT NOT NULL,
    axis                TEXT NOT NULL,
    action              TEXT NOT NULL,
    old_track_audio_id  INTEGER,
    old_platform        TEXT,
    old_player_id       TEXT,
    old_url             TEXT,
    new_track_audio_id  INTEGER,
    new_platform        TEXT,
    new_player_id       TEXT,
    new_url             TEXT,
    old_recording_id    TEXT,
    new_recording_id    TEXT,
    stem_value          TEXT,
    reason              TEXT,
    source              TEXT,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK (axis IN ('version','variant','stem','recording')),
    CHECK (action IN ('replace','add','relink','detach'))
);

INSERT INTO track_audio_correction
  (correction_id, set_id, position, track_id, axis, action,
   old_track_audio_id, old_platform, old_player_id, old_url,
   new_track_audio_id, new_platform, new_player_id, new_url,
   stem_value, reason, source, created_at)
SELECT
   correction_id, set_id, position, track_id, axis, action,
   old_track_audio_id, old_platform, old_player_id, old_url,
   new_track_audio_id, new_platform, new_player_id, new_url,
   stem_value, reason, source, created_at
FROM track_audio_correction_old;

DROP TABLE track_audio_correction_old;

CREATE INDEX IF NOT EXISTS idx_track_audio_correction_track ON track_audio_correction(track_id);
CREATE INDEX IF NOT EXISTS idx_track_audio_correction_set   ON track_audio_correction(set_id);

COMMIT;
PRAGMA foreign_keys=ON;
