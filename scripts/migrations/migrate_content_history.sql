-- content_history: append-only content-history hash ledger (Crush completeness
-- Phase B, spec 2026-07-22-gt-identity-binding-completeness-design.md v2.2/v4.6).
-- Back up music_database.db first. Apply-exactly-once in spirit, but written to
-- be re-run-safe: the CREATEs are IF NOT EXISTS and the backfill is guarded by
-- NOT EXISTS on track_audio_id, so a second run is a no-op (no duplicate
-- generation-0 rows).
--
-- The DDL below is kept byte-identical to ingest/content_history.py::SCHEMA and
-- web_crawler/database/schema.sql (single source, no drift).

PRAGMA foreign_keys = OFF;

CREATE TABLE IF NOT EXISTS content_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id    TEXT NOT NULL,
    version         TEXT,
    stem            TEXT NOT NULL,
    variant         TEXT NOT NULL,
    kind            TEXT NOT NULL,
    track_audio_id  INTEGER,
    content_sha256  TEXT,
    payload_sha256  TEXT,
    parent_content_sha256 TEXT,
    parent_payload_sha256 TEXT,
    op              TEXT,
    source          TEXT,
    generation      INTEGER NOT NULL DEFAULT 0,
    valid           INTEGER NOT NULL DEFAULT 1,
    ts              DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK (kind IN ('master','separated','derived')),
    CHECK (valid IN (0,1))
);
CREATE INDEX IF NOT EXISTS idx_content_history_chain
    ON content_history(recording_id, stem, variant, kind, generation);
CREATE INDEX IF NOT EXISTS idx_content_history_content_sha ON content_history(content_sha256);
CREATE INDEX IF NOT EXISTS idx_content_history_payload_sha ON content_history(payload_sha256);

-- Backfill: seed one generation-0 row per existing track_audio rip that has a
-- sha256. kind='master' (downloaded files); track_stems (separated) lineages
-- are seeded lazily by B2's hash-before-regenerate hook, not here. `version`
-- left NULL — it is a denormalized audit only; recording_id pins the identity
-- (v4.6). Caveat: sibling rips of the same (recording,stem,variant) each get
-- generation 0 (the ordinal is an audit index, not the bind key — binding is by
-- hash equality regardless of ordinal, so the collision is cosmetic).
--
-- `recording_id IS NOT NULL` guard: content_history.recording_id is NOT NULL
-- (the chain key pins identity by recording), but canonical carries a few
-- identity-incomplete rips (sha256 present, recording_id still NULL — the ingest
-- frontier). Without this filter the INSERT aborts on the first such row and
-- backfills nothing (found live: 1 row nulled the whole 19,608-row seed). Those
-- rows cannot participate in a recording-keyed chain anyway, so skipping them is
-- correct, not lossy.
INSERT INTO content_history
  (recording_id, stem, variant, kind, track_audio_id, content_sha256,
   generation, op, source)
SELECT
    ta.recording_id, ta.stem, ta.variant, 'master', ta.track_audio_id,
    ta.sha256, 0, 'fetch', 'backfill_migration'
FROM track_audio ta
WHERE ta.sha256 IS NOT NULL
  AND ta.recording_id IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM content_history ch
      WHERE ch.track_audio_id = ta.track_audio_id
  );
