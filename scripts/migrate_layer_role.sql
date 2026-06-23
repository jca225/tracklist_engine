-- Add layer_role and constituents_json to set_track_slots (pi-storage).
-- Backup DB first. Then: python -m tokenizer.materialize (after deploy).

ALTER TABLE set_track_slots ADD COLUMN layer_role TEXT;
ALTER TABLE set_track_slots ADD COLUMN constituents_json TEXT;
