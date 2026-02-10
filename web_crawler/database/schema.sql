
PRAGMA foreign_keys = ON;

-- =============================================================================
-- CANONICAL: DJ SETS
-- =============================================================================
-- This is your "best current belief" about each DJ set. It is safe for downstream
-- analytics, training data extraction, etc.
CREATE TABLE IF NOT EXISTS dj_sets (
    set_id TEXT PRIMARY KEY,              -- 1001tracklists tracklist_id
    set_url TEXT,                         -- canonical URL for this set page

    title TEXT DEFAULT '',
    date_played TEXT DEFAULT '',          -- ISO string if available (or '')
    artists TEXT,                         -- raw string or JSON string

    creator_name TEXT,
    creator_url TEXT,

    views INTEGER,
    ided_tracks INTEGER,
    total_tracks INTEGER,
    likes INTEGER,

    play_time TEXT,
    styles TEXT,                          -- raw string or JSON string

    scraped_at DATETIME DEFAULT CURRENT_TIMESTAMP
);


-- =============================================================================
-- EVIDENCE: SET CRAWLS (snapshots of fetched pages)
-- =============================================================================
-- Store what you actually fetched. This allows reproducibility and re-parsing.
CREATE TABLE IF NOT EXISTS dj_set_crawls (
    crawl_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id       TEXT NOT NULL,
    set_url      TEXT NOT NULL,

    fetched_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    http_status  INTEGER,
    etag         TEXT,
    last_modified TEXT,

    html_sha256  TEXT,             -- hash for dedupe / change detection
    html_path     TEXT,             -- optional; can store NULL if saving elsewhere

    FOREIGN KEY (set_id) REFERENCES dj_sets(set_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_dj_set_crawls_set_id ON dj_set_crawls(set_id);
CREATE INDEX IF NOT EXISTS idx_dj_set_crawls_sha    ON dj_set_crawls(html_sha256);



-- =============================================================================
-- SET-LEVEL MEDIA LINKS
-- =============================================================================
CREATE TABLE IF NOT EXISTS dj_set_media_links (
    media_link_id INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id        TEXT NOT NULL,
    platform      TEXT NOT NULL,   -- youtube/soundcloud/spotify/apple/mixcloud/hearthis/other
    url           TEXT,            -- resolved URL if available
    id_item       TEXT,            -- data-idmedia
    id_source     TEXT,            -- data-idsource
    scraped_at    DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (set_id) REFERENCES dj_sets(set_id) ON DELETE CASCADE
);


-- =============================================================================
-- RAW ROWS (direct children of #tlTab)
-- =============================================================================
CREATE TABLE IF NOT EXISTS dj_set_rows (
    row_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id      TEXT NOT NULL,
    row_index   INTEGER NOT NULL, -- delete?
    element_id  TEXT, -- delete?
    classes     TEXT, -- delete?
    data_attrs_json TEXT, -- delete?
    text_excerpt TEXT, -- delete?
    raw_html    TEXT,
    scraped_at  DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (set_id) REFERENCES dj_sets(set_id) ON DELETE CASCADE
);


-- =============================================================================
-- SCRAPE FAILURES (set-level + track-level)
-- =============================================================================
CREATE TABLE IF NOT EXISTS scrape_failures (
    failure_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id      TEXT NOT NULL,
    set_url     TEXT,
    stage       TEXT NOT NULL,   -- page_load | scrape | ajax | db

    track_title TEXT,
    track_id    TEXT,
    tlp_id      TEXT,
    params_json TEXT,

    error       TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);


-- =============================================================================
-- TRACK-LEVEL MEDIA LINKS (AJAX media viewer)
-- =============================================================================
CREATE TABLE IF NOT EXISTS dj_set_track_media_links (
    track_media_id INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id         TEXT NOT NULL,
    tlp_id         TEXT,            -- data-id from tlp row
    track_id       TEXT,            -- data-trackid from mediaRow / tlp row

    platform       TEXT NOT NULL,   -- youtube/soundcloud/spotify/apple/beatport/traxsource/affiliate/other
    player_id      TEXT,            -- raw playerId from AJAX response

    id_object      TEXT,            -- request params for re-fetch
    id_item        TEXT,
    id_source      TEXT,
    view_source    TEXT,
    view_item      TEXT,

    scraped_at     DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (set_id) REFERENCES dj_sets(set_id) ON DELETE CASCADE
);
