-- Identity-axis migration (run ONCE on pi-storage after code deploy).
-- Backs up music_database.db first. Then re-run tokenizer.materialize.
-- Requires SQLite 3.25+ (RENAME COLUMN).

PRAGMA foreign_keys = OFF;

-- ── track_audio: variant_tag→stem, edit_tag→variant ─────────────────────
ALTER TABLE track_audio RENAME COLUMN variant_tag TO stem;
ALTER TABLE track_audio RENAME COLUMN edit_tag TO variant;
UPDATE track_audio SET stem = 'regular' WHERE stem IN ('original', 'full');

-- ── track_metadata: version_tag→version ─────────────────────────────────
ALTER TABLE track_metadata RENAME COLUMN version_tag TO version;
UPDATE track_metadata SET version = lower(version) WHERE version IS NOT NULL;
UPDATE track_metadata SET version = 'original'
  WHERE version IS NULL OR version IN ('acappella', 'acapella');
UPDATE track_metadata SET version = 'remix' WHERE version = 'remix';
UPDATE track_metadata SET version = 'rework' WHERE version = 'rework';
UPDATE track_metadata SET version = 'altversion' WHERE version IN ('altversion', 'alt version');

-- ── set_track_slots: claims (add before rename if old schema) ───────────
-- If columns already exist from a partial run, skip the ADDs.
-- claimed_version / claimed_stem / claimed_variant populated by materialize.

-- ── fingerprints, cues, hits ────────────────────────────────────────────
ALTER TABLE track_fingerprints RENAME COLUMN variant_tag TO stem;
UPDATE track_fingerprints SET stem = 'regular' WHERE stem IN ('original', 'full');

ALTER TABLE canonical_track_cue_points RENAME COLUMN source_variant_tag TO source_stem;
UPDATE canonical_track_cue_points SET source_stem = 'regular'
  WHERE source_stem IN ('original', 'full');

ALTER TABLE set_fingerprint_hits RENAME COLUMN matched_variant TO matched_stem;
UPDATE set_fingerprint_hits SET matched_stem = 'regular'
  WHERE matched_stem IN ('original', 'full');

-- ── correction ledger ─────────────────────────────────────────────────────
ALTER TABLE track_audio_correction RENAME COLUMN variant_tag TO stem_value;
UPDATE track_audio_correction SET stem_value = 'regular'
  WHERE stem_value IN ('original', 'full');

PRAGMA foreign_keys = ON;
