-- C0: derivation lineage columns on content_history (Phase B continuation).
-- Additive ALTER for DBs whose CREATE already ran without parent_*.
-- Re-run note: raw ALTER fails with duplicate-column if columns already exist
-- (fresh installs get them from CREATE). ingest.content_history._ensure_schema
-- is the idempotent self-heal path for auto-pull.

PRAGMA foreign_keys = OFF;

ALTER TABLE content_history ADD COLUMN parent_content_sha256 TEXT;
ALTER TABLE content_history ADD COLUMN parent_payload_sha256 TEXT;
