-- WS-C: persist the community-correction + notice + ID-resolution layer.
-- Additive, non-rewriting. Run once against the canonical DB, then re-run
-- `python -m tokenizer.materialize` to backfill values.
ALTER TABLE track_suggestions ADD COLUMN data_type INTEGER;
ALTER TABLE track_suggestions ADD COLUMN cue_seconds INTEGER;
ALTER TABLE track_suggestions ADD COLUMN play_cue_seconds INTEGER;
ALTER TABLE track_suggestions ADD COLUMN suggester_guest_id INTEGER;
ALTER TABLE track_suggestions ADD COLUMN suggester_kind TEXT;
ALTER TABLE track_suggestions ADD COLUMN track_page_path TEXT;
ALTER TABLE track_suggestions ADD COLUMN track_id_numeric INTEGER;
ALTER TABLE track_suggestions ADD COLUMN is_id_remix INTEGER;
ALTER TABLE track_suggestions ADD COLUMN has_apple INTEGER;
ALTER TABLE track_suggestions ADD COLUMN has_affiliate INTEGER;
ALTER TABLE track_suggestions ADD COLUMN has_live_video INTEGER;
ALTER TABLE track_suggestions ADD COLUMN poll_correct INTEGER;
ALTER TABLE track_suggestions ADD COLUMN poll_not_correct INTEGER;
ALTER TABLE track_suggestions ADD COLUMN poll_unsure INTEGER;
ALTER TABLE track_suggestions ADD COLUMN labels_json TEXT;
ALTER TABLE track_suggestions ADD COLUMN google_search_url TEXT;

CREATE TABLE IF NOT EXISTS set_notices (
    set_id TEXT NOT NULL, row_index INTEGER NOT NULL,
    row_type TEXT, text TEXT, links_json TEXT, icons_json TEXT,
    parsed_json TEXT, parsed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (set_id, row_index)
);
CREATE INDEX IF NOT EXISTS idx_set_notices_set ON set_notices(set_id);

CREATE TABLE IF NOT EXISTS set_slot_id_meta (
    set_id TEXT NOT NULL, row_index INTEGER NOT NULL, tlp_id INTEGER,
    is_id INTEGER DEFAULT 0, protected INTEGER DEFAULT 0, rbcst INTEGER DEFAULT 0,
    watchers INTEGER, presave_count INTEGER,
    parsed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (set_id, row_index)
);
CREATE INDEX IF NOT EXISTS idx_set_slot_id_meta_set ON set_slot_id_meta(set_id);
-- track_id_links already exists (web_crawler/database/schema.sql).
