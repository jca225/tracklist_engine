-- Read-only view: slot claims vs best track_audio match (pi-storage).
-- Apply: sqlite3 /mnt/storage/data/db/music_database.db < scripts/migrate_slot_satisfaction_view.sql

DROP VIEW IF EXISTS slot_inventory_view;
CREATE VIEW slot_inventory_view AS
SELECT
    s.set_id,
    s.slot_label,
    s.row_index,
    s.full_name,
    COALESCE(s.recording_id, s.track_id) AS recording_id,
    s.claimed_stem,
    s.claimed_variant,
    -- Mirrors core.slot_inventory.derive_layer_role (the gate's source of truth):
    -- a w1 overlay is the payload only when it's regular- or acappella-stemmed;
    -- an instrumental first-overlay is a (bed-like) constituent, not the payload.
    CASE
        WHEN s.slot_label GLOB '[0-9][0-9][0-9]' AND s.is_concurrent = 1 THEN 'bed'
        WHEN s.slot_label GLOB '[0-9][0-9][0-9]w*' AND s.claimed_stem = 'acappella' THEN 'payload'
        WHEN s.slot_label GLOB '[0-9][0-9][0-9]w1'
             AND COALESCE(s.claimed_stem, 'regular') IN ('regular', 'full', 'original') THEN 'payload'
        WHEN s.slot_label GLOB '[0-9][0-9][0-9]w*' THEN 'constituent'
        ELSE 'solo'
    END AS layer_role,
    s.is_concurrent,
    ta.track_audio_id,
    ta.stem AS asset_stem,
    ta.variant AS asset_variant,
    ta.platform,
    ta.is_reference,
    ta.path AS asset_path,
    im.claimed_version AS mismatch_claimed_version,
    im.canonical_version AS mismatch_canonical_version
FROM set_track_slots s
LEFT JOIN track_audio ta ON (
    ta.recording_id = COALESCE(s.recording_id, s.track_id)
    OR ta.track_id = COALESCE(s.recording_id, s.track_id)
)
LEFT JOIN identity_mismatch im ON (
    im.set_id = s.set_id AND im.row_index = s.row_index
)
WHERE s.slot_label IS NOT NULL;
