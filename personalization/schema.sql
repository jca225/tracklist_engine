-- Taste prior warehouse (pi-worker local). NOT canonical music_database.db.
-- Rebuildable from JSONL under data/taste/raw/.

CREATE TABLE IF NOT EXISTS listeners (
    user_id                 TEXT PRIMARY KEY,
    platform                TEXT NOT NULL,
    handle                  TEXT NOT NULL,
    mix_id                  TEXT NOT NULL,
    sc_user_id              INTEGER,
    first_seen_at           TEXT NOT NULL,
    source_evidence_json    TEXT NOT NULL DEFAULT '{}',
    username                TEXT,
    followers_count         INTEGER,
    followings_count        INTEGER,
    verified                INTEGER DEFAULT 0,
    city                    TEXT,
    country_code            TEXT
);

CREATE INDEX IF NOT EXISTS idx_listeners_mix ON listeners(mix_id);
CREATE INDEX IF NOT EXISTS idx_listeners_sc ON listeners(sc_user_id);

CREATE TABLE IF NOT EXISTS sc_likes (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                 TEXT NOT NULL,
    mix_id                  TEXT NOT NULL,
    sc_user_id              INTEGER NOT NULL,
    liked_at                TEXT NOT NULL,
    track_id                INTEGER NOT NULL,
    track_title             TEXT,
    track_permalink         TEXT,
    track_artist_username   TEXT,
    track_genre             TEXT,
    raw_json                TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES listeners(user_id) ON DELETE CASCADE,
    UNIQUE(sc_user_id, track_id, liked_at)
);

CREATE INDEX IF NOT EXISTS idx_sc_likes_user ON sc_likes(user_id);
CREATE INDEX IF NOT EXISTS idx_sc_likes_mix ON sc_likes(mix_id);

-- Comments on the mix upload itself. track_position_ms = playhead in the mix when
-- they commented (primary timestep for structure / engagement alignment).
CREATE TABLE IF NOT EXISTS sc_mix_comments (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                 TEXT NOT NULL,
    mix_id                  TEXT NOT NULL,
    sc_user_id              INTEGER NOT NULL,
    sc_track_id             INTEGER NOT NULL,
    comment_id              INTEGER NOT NULL,
    commented_at            TEXT NOT NULL,
    mix_position_ms         INTEGER,
    body                    TEXT NOT NULL,
    raw_json                TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES listeners(user_id) ON DELETE CASCADE,
    UNIQUE(comment_id)
);

CREATE INDEX IF NOT EXISTS idx_sc_mix_comments_mix ON sc_mix_comments(mix_id);
CREATE INDEX IF NOT EXISTS idx_sc_mix_comments_position ON sc_mix_comments(mix_id, mix_position_ms);

CREATE TABLE IF NOT EXISTS sc_playlists (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                 TEXT NOT NULL,
    mix_id                  TEXT NOT NULL,
    sc_user_id              INTEGER NOT NULL,
    playlist_id             INTEGER NOT NULL,
    title                   TEXT,
    track_count             INTEGER,
    track_ids_json          TEXT NOT NULL,
    created_at              TEXT,
    last_modified           TEXT,
    raw_json                TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES listeners(user_id) ON DELETE CASCADE,
    UNIQUE(sc_user_id, playlist_id)
);

CREATE INDEX IF NOT EXISTS idx_sc_playlists_user ON sc_playlists(sc_user_id);
CREATE INDEX IF NOT EXISTS idx_sc_playlists_mix ON sc_playlists(mix_id);

-- Heuristic bot / low-quality listener scores (recomputed from warehouse).
CREATE TABLE IF NOT EXISTS listener_bot_scores (
    user_id                 TEXT PRIMARY KEY,
    mix_id                  TEXT NOT NULL,
    sc_user_id              INTEGER,
    bot_score               REAL NOT NULL,
    is_bot                  INTEGER NOT NULL DEFAULT 0,
    reasons_json            TEXT NOT NULL DEFAULT '[]',
    computed_at             TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES listeners(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_listener_bot_mix ON listener_bot_scores(mix_id);

-- Cached MERT layer-6 summary vectors for SoundCloud tracks (user-prior building blocks).
CREATE TABLE IF NOT EXISTS sc_track_mert (
    sc_track_id             INTEGER NOT NULL,
    mert_version            TEXT NOT NULL,
    dim                     INTEGER NOT NULL DEFAULT 1024,
    embedding               BLOB NOT NULL,
    source_url              TEXT,
    embedded_at             TEXT NOT NULL,
    PRIMARY KEY (sc_track_id, mert_version)
);

CREATE TABLE IF NOT EXISTS user_prior_vectors (
    user_id                 TEXT PRIMARY KEY,
    mix_id                  TEXT NOT NULL,
    sc_user_id              INTEGER,
    mert_version            TEXT NOT NULL,
    dim                     INTEGER NOT NULL DEFAULT 1024,
    n_tracks_used           INTEGER NOT NULL,
    embedding               BLOB NOT NULL,
    computed_at             TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES listeners(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_prior_mix ON user_prior_vectors(mix_id);

CREATE TABLE IF NOT EXISTS taste_clusters (
    user_id                 TEXT NOT NULL,
    mix_id                  TEXT NOT NULL,
    cluster_id              INTEGER NOT NULL,
    algorithm               TEXT NOT NULL,
    computed_at             TEXT NOT NULL,
    PRIMARY KEY (user_id, mix_id, algorithm),
    FOREIGN KEY (user_id) REFERENCES listeners(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_taste_clusters_mix ON taste_clusters(mix_id, cluster_id);

CREATE TABLE IF NOT EXISTS scrape_checkpoints (
    mix_id                  TEXT NOT NULL,
    phase                   TEXT NOT NULL,
    checkpoint_json         TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    PRIMARY KEY (mix_id, phase)
);

CREATE TABLE IF NOT EXISTS scrape_runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    phase                   TEXT NOT NULL,
    mix_id                  TEXT NOT NULL,
    started_at              TEXT NOT NULL,
    finished_at             TEXT NOT NULL,
    output_rows             INTEGER NOT NULL,
    params_json             TEXT NOT NULL DEFAULT '{}'
);
