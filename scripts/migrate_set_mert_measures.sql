-- P4 / 6b: set-side MERT measure cache (mirror of track_mert_measures).
-- Beat grid comes from set_analysis.measure_times_json (beat_this on mix).
-- Apply on pi-storage once before set_mert_backfill_loop.py:
--   ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db < scripts/migrate_set_mert_measures.sql'

CREATE TABLE IF NOT EXISTS set_mert_measures (
    set_audio_id   INTEGER NOT NULL,
    measure_idx    INTEGER NOT NULL,
    start_s        REAL NOT NULL,
    end_s          REAL NOT NULL,
    dim            INTEGER NOT NULL,
    dtype          TEXT NOT NULL,             -- 'float16'
    embedding      BLOB NOT NULL,             -- (dim,) bytes, mean-pooled
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (set_audio_id, measure_idx),
    FOREIGN KEY (set_audio_id) REFERENCES set_audio(set_audio_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_set_mert_measures_set ON set_mert_measures(set_audio_id);
