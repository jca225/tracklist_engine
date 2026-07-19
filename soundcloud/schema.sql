-- soundcloud/schema.sql
-- SoundCloud data lake (pi-storage: /mnt/storage/data/soundcloud/sc_lake.db).
-- Off-canonical: NOT music_database.db, NOT on the alignment DAG.
-- Rebuildable from raw/{entity}/{id}/{fetched_at}.jsonl.

CREATE TABLE IF NOT EXISTS sc_users (
    sc_user_id        INTEGER PRIMARY KEY,
    permalink         TEXT,
    username          TEXT,
    followers_count   INTEGER,
    followings_count  INTEGER,
    verified          INTEGER NOT NULL DEFAULT 0,
    city              TEXT,
    country           TEXT,
    description       TEXT,
    fetched_at        TEXT NOT NULL,
    raw_ref           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sc_tracks (
    sc_track_id        INTEGER PRIMARY KEY,
    title              TEXT,
    owner_sc_user_id   INTEGER,
    genre              TEXT,
    tag_list           TEXT,
    duration_ms        INTEGER,
    playback_count     INTEGER,
    likes_count        INTEGER,
    created_at         TEXT,
    permalink          TEXT,
    fetched_at         TEXT NOT NULL,
    raw_ref            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sc_playlists (
    sc_playlist_id     INTEGER PRIMARY KEY,
    title              TEXT,
    owner_sc_user_id   INTEGER,
    track_count        INTEGER,
    is_album           INTEGER NOT NULL DEFAULT 0,
    fetched_at         TEXT NOT NULL,
    raw_ref            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sc_likes (
    sc_user_id   INTEGER NOT NULL,
    sc_track_id  INTEGER NOT NULL,
    created_at   TEXT,
    PRIMARY KEY (sc_user_id, sc_track_id)
);

CREATE TABLE IF NOT EXISTS sc_reposts (
    sc_user_id    INTEGER NOT NULL,
    item_kind     TEXT NOT NULL,          -- 'track' | 'playlist'
    item_id       INTEGER NOT NULL,
    created_at    TEXT,
    PRIMARY KEY (sc_user_id, item_kind, item_id)
);

CREATE TABLE IF NOT EXISTS sc_follows (
    follower_sc_user_id  INTEGER NOT NULL,
    followee_sc_user_id  INTEGER NOT NULL,
    PRIMARY KEY (follower_sc_user_id, followee_sc_user_id)
);

CREATE TABLE IF NOT EXISTS sc_playlist_tracks (
    sc_playlist_id  INTEGER NOT NULL,
    sc_track_id     INTEGER NOT NULL,
    position        INTEGER NOT NULL,
    PRIMARY KEY (sc_playlist_id, sc_track_id)
);

-- Identity seam (stub): populated by the future audio-seed / discovery consumers.
CREATE TABLE IF NOT EXISTS sc_recording_map (
    sc_track_id   INTEGER PRIMARY KEY,
    recording_id  TEXT,
    method        TEXT,
    confidence    REAL
);

CREATE TABLE IF NOT EXISTS crawl_checkpoints (
    sc_user_id   INTEGER NOT NULL,
    phase        TEXT NOT NULL,           -- entity type, e.g. 'likes'
    done         INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (sc_user_id, phase)
);

CREATE INDEX IF NOT EXISTS idx_sc_tracks_owner ON sc_tracks(owner_sc_user_id);
CREATE INDEX IF NOT EXISTS idx_sc_likes_track ON sc_likes(sc_track_id);
CREATE INDEX IF NOT EXISTS idx_sc_follows_followee ON sc_follows(followee_sc_user_id);
