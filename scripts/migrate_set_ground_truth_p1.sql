-- P1 ground-truth columns (idempotent on pi-storage).
-- Run once after deploy: sqlite3 /mnt/storage/data/db/music_database.db < scripts/migrate_set_ground_truth_p1.sql

ALTER TABLE set_ground_truth ADD COLUMN tempo_ratio REAL;
ALTER TABLE set_ground_truth ADD COLUMN pitch_shift_semi INTEGER;
ALTER TABLE set_ground_truth ADD COLUMN ref_source TEXT;
