-- Add the 'recording' axis + detach/relink actions to track_audio_correction.
-- SQLite cannot ALTER a CHECK constraint, so rebuild the table (no FKs to worry about).
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
