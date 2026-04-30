
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
    retries     INTEGER DEFAULT 0,
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


-- =============================================================================
-- AUDIO PIPELINE
-- =============================================================================
-- Downloaded audio per (canonical track_id, platform). A track may have
-- multiple rips; one is promoted to "reference" for MERT/analysis.
CREATE TABLE IF NOT EXISTS track_audio (
    track_audio_id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id       TEXT NOT NULL,            -- 1001tracklists data-trackid
    platform       TEXT NOT NULL,            -- youtube | soundcloud
    source_url     TEXT NOT NULL,
    player_id      TEXT,                     -- platform-specific id used for fetch
    path           TEXT NOT NULL,            -- absolute or repo-relative path
    sha256         TEXT,
    duration_s     REAL,
    sample_rate    INTEGER,
    codec          TEXT,
    bitrate_kbps   INTEGER,
    is_reference   INTEGER DEFAULT 0,        -- 1 = chosen reference for this track
    -- Which version of the song this audio is: 'original' (full song, the
    -- canonical cue-detr target), 'acappella' (vocals-only from the
    -- mashup source), 'instrumental' (instrumental-only), 'remix' (remixed
    -- variant). Assigned by parsing the scraped tracklist row text.
    variant_tag    TEXT NOT NULL DEFAULT 'original',
    downloaded_at  DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(track_id, platform, player_id)
);

CREATE INDEX IF NOT EXISTS idx_track_audio_track_id ON track_audio(track_id);
CREATE INDEX IF NOT EXISTS idx_track_audio_reference ON track_audio(track_id, is_reference);


-- Demucs-separated stems for a specific audio asset.
CREATE TABLE IF NOT EXISTS track_stems (
    track_stem_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    track_audio_id INTEGER NOT NULL,
    stem_name      TEXT NOT NULL,            -- vocals | drums | bass | other
    path           TEXT NOT NULL,
    codec          TEXT,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(track_audio_id, stem_name),
    FOREIGN KEY (track_audio_id) REFERENCES track_audio(track_audio_id) ON DELETE CASCADE
);


-- Scalar MIR features (key/bpm/loudness/mood) derived from a specific audio asset.
-- `source` names the extractor (essentia / beat_this / musicbrainz / inferred).
CREATE TABLE IF NOT EXISTS track_audio_features (
    track_audio_id    INTEGER NOT NULL,
    source            TEXT NOT NULL,
    key_pc            INTEGER,                -- 0..11 pitch class, NULL if unknown
    key_mode          TEXT,                   -- 'major' | 'minor' | NULL
    bpm               REAL,
    time_sig_num      INTEGER,
    time_sig_den      INTEGER,
    lufs              REAL,
    danceability      REAL,
    energy            REAL,
    valence           REAL,
    confidence_json   TEXT,                   -- per-field confidences as JSON
    analyzed_at       DATETIME DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (track_audio_id, source),
    FOREIGN KEY (track_audio_id) REFERENCES track_audio(track_audio_id) ON DELETE CASCADE
);


-- Time-axis analysis: measure grid and cue-detr cue points.
CREATE TABLE IF NOT EXISTS track_analysis (
    track_audio_id         INTEGER PRIMARY KEY,
    beat_times_json        TEXT,              -- JSON array of beat timestamps (s)
    downbeat_times_json    TEXT,              -- JSON array of downbeat timestamps (s)
    measure_times_json     TEXT,              -- JSON array of measure-start timestamps (s)
    cue_points_json        TEXT,              -- JSON array of cue-detr cue timestamps (s)
    analyzer_versions_json TEXT,              -- {"beat_this": "x.y", "cue_detr": "..."} for reproducibility
    analyzed_at            DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (track_audio_id) REFERENCES track_audio(track_audio_id) ON DELETE CASCADE
);


-- MERT embeddings stored per cue-delimited section, not per measure.
-- embedding is raw float16 bytes; shape is (n_frames, dim) — store dims for reloading.
CREATE TABLE IF NOT EXISTS track_mert_sections (
    track_audio_id INTEGER NOT NULL,
    section_idx    INTEGER NOT NULL,
    start_s        REAL NOT NULL,
    end_s          REAL NOT NULL,
    n_frames       INTEGER NOT NULL,
    dim            INTEGER NOT NULL,
    dtype          TEXT NOT NULL,             -- e.g. 'float16'
    embedding      BLOB NOT NULL,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (track_audio_id, section_idx),
    FOREIGN KEY (track_audio_id) REFERENCES track_audio(track_audio_id) ON DELETE CASCADE
);


-- Canonical identity resolution: AcoustID fingerprint → MusicBrainz recording.
CREATE TABLE IF NOT EXISTS track_identity (
    track_id                 TEXT PRIMARY KEY,
    acoustid_fingerprint     TEXT,
    acoustid_id              TEXT,
    musicbrainz_recording_id TEXT,
    isrc                     TEXT,
    release_date             TEXT,
    release_label            TEXT,
    resolved_at              DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_track_identity_mbid ON track_identity(musicbrainz_recording_id);
CREATE INDEX IF NOT EXISTS idx_track_identity_isrc ON track_identity(isrc);


-- Full-mix audio for a DJ set itself (as posted on YT/SC/Mixcloud etc.).
-- Used for manual review, gap-filling, and alignment of cue-delimited sections
-- in the played mix back to per-track reference audio.
CREATE TABLE IF NOT EXISTS set_audio (
    set_audio_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id         TEXT NOT NULL,
    platform       TEXT NOT NULL,            -- youtube | soundcloud | mixcloud | other
    source_url     TEXT NOT NULL,            -- normalized URL we actually fetched
    path           TEXT NOT NULL,
    sha256         TEXT,
    duration_s     REAL,
    sample_rate    INTEGER,
    codec          TEXT,
    bitrate_kbps   INTEGER,
    is_reference   INTEGER DEFAULT 0,        -- 1 = the one we treat as canonical for alignment
    downloaded_at  DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(set_id, platform, source_url),
    FOREIGN KEY (set_id) REFERENCES dj_sets(set_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_set_audio_set_id ON set_audio(set_id);


-- Sidecar timeline: ordered tokenized rows with their cue-section anchor,
-- persisted alongside set_audio so analysts / the UI can open one JSON and see
-- "which track is supposed to be playing at T seconds". Schema is {set_id,
-- set_audio_id, payload} where payload is the serialized list of segments.
CREATE TABLE IF NOT EXISTS set_timeline (
    set_id         TEXT PRIMARY KEY,
    set_audio_id   INTEGER,
    payload_json   TEXT NOT NULL,
    built_at       DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (set_id)       REFERENCES dj_sets(set_id) ON DELETE CASCADE,
    FOREIGN KEY (set_audio_id) REFERENCES set_audio(set_audio_id) ON DELETE SET NULL
);


-- Set-level beat/downbeat/measure grid from beat_this on the full DJ mix.
-- Used as the mix-axis measure grid for stage-5 measure refinement.
CREATE TABLE IF NOT EXISTS set_analysis (
    set_audio_id           INTEGER PRIMARY KEY,
    beat_times_json        TEXT,              -- JSON array of beat timestamps (s)
    downbeat_times_json    TEXT,              -- JSON array of downbeat timestamps (s)
    measure_times_json     TEXT,              -- JSON array of measure-start timestamps (s)
    analyzer_versions_json TEXT,
    analyzed_at            DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (set_audio_id) REFERENCES set_audio(set_audio_id) ON DELETE CASCADE
);


-- Demucs 4-stem split of the full DJ mix. Used by the stem-mask classifier
-- (stage 4) to decide which stems of which refs are audible in each aligned
-- measure of the played mix.
CREATE TABLE IF NOT EXISTS set_stems (
    set_stem_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    set_audio_id   INTEGER NOT NULL,
    stem_name      TEXT NOT NULL,             -- vocals | drums | bass | other
    path           TEXT NOT NULL,
    codec          TEXT,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(set_audio_id, stem_name),
    FOREIGN KEY (set_audio_id) REFERENCES set_audio(set_audio_id) ON DELETE CASCADE
);


-- =============================================================================
-- MEASURE-LEVEL STRUCTURE (first-class, populated from track_analysis /
-- set_analysis JSON blobs during the revamp — see `audio_pipeline/adapters/
-- measure_adapter.py`).
-- =============================================================================

-- Per-track measure grid with per-measure BPM + (optional) key.
-- Populated from `track_analysis.measure_times_json` + `track_audio_features`.
CREATE TABLE IF NOT EXISTS track_measures (
    track_audio_id INTEGER NOT NULL,
    measure_idx    INTEGER NOT NULL,
    start_s        REAL NOT NULL,
    end_s          REAL NOT NULL,
    bpm            REAL,
    key_pc         INTEGER,                  -- 0..11
    key_mode       TEXT,                     -- 'major' | 'minor' | NULL
    PRIMARY KEY (track_audio_id, measure_idx),
    FOREIGN KEY (track_audio_id) REFERENCES track_audio(track_audio_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS set_measures (
    set_audio_id   INTEGER NOT NULL,
    measure_idx    INTEGER NOT NULL,
    start_s        REAL NOT NULL,
    end_s          REAL NOT NULL,
    bpm            REAL,
    PRIMARY KEY (set_audio_id, measure_idx),
    FOREIGN KEY (set_audio_id) REFERENCES set_audio(set_audio_id) ON DELETE CASCADE
);


-- Measure-level alignment: for each played mix measure, which ref measure
-- of which track is sounding (multiple rows per mix measure for mashups),
-- at what pitch shift, tempo ratio, and stem mask. This is the atomic
-- "playback script" — rendering the ref stems per row reconstructs the mix.
CREATE TABLE IF NOT EXISTS measure_alignment (
    set_id            TEXT NOT NULL,
    set_measure_idx   INTEGER NOT NULL,
    ref_track_id      TEXT NOT NULL,
    ref_measure_idx   INTEGER NOT NULL,
    pitch_shift_semi  INTEGER NOT NULL DEFAULT 0,    -- -6..+6
    tempo_ratio       REAL NOT NULL DEFAULT 1.0,     -- ref_dur / mix_dur; >1 = DJ sped ref up
    stem_mask_json    TEXT NOT NULL,                 -- JSON list e.g. ["vocals"] | ["drums","bass","other"] | ["full"]
    gain_db           REAL,
    confidence        REAL,
    aligned_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (set_id, set_measure_idx, ref_track_id),
    FOREIGN KEY (set_id) REFERENCES dj_sets(set_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_measure_alignment_set ON measure_alignment(set_id);
CREATE INDEX IF NOT EXISTS idx_measure_alignment_ref ON measure_alignment(ref_track_id);


-- One-per-set rendered playback score (measure_alignment rolled up as JSON)
-- + self-consistency metric against the actual mix audio.
CREATE TABLE IF NOT EXISTS set_playback_score (
    set_id                         TEXT PRIMARY KEY,
    score_json                     TEXT NOT NULL,
    reconstruction_mfcc_distance   REAL,      -- mean cosine distance per 100ms frame, 0 = perfect
    reconstruction_method          TEXT,      -- 'pyrubberband_v1' | future versions
    rendered_at                    DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (set_id) REFERENCES dj_sets(set_id) ON DELETE CASCADE
);


-- =============================================================================
-- IDENTITY (chromaprint / acoustid)
-- =============================================================================
-- Chromaprint raw fingerprint per (track, variant). 'variant_tag' supports
-- storing fingerprints for derived audio too (vocals-only, instrumental
-- sum) so stage-1b identity can distinguish acappella from full plays.
CREATE TABLE IF NOT EXISTS track_fingerprints (
    track_id      TEXT NOT NULL,
    variant_tag   TEXT NOT NULL DEFAULT 'original',
    fingerprint   BLOB NOT NULL,
    duration_s    REAL NOT NULL,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (track_id, variant_tag)
);

CREATE INDEX IF NOT EXISTS idx_track_fingerprints_track ON track_fingerprints(track_id);


-- =============================================================================
-- Canonical cue points per song (NOT per audio variant). Cue-detr runs
-- poorly on vocals-only or instrumental-only audio because it was trained
-- on EDM with drops; running it on the full/original version produces
-- dense, reliable cue points that are then shared across all variants of
-- the same track_id. Populated by audio_pipeline.analysis.canonical_cues.
CREATE TABLE IF NOT EXISTS canonical_track_cue_points (
    track_id               TEXT PRIMARY KEY,
    cue_points_json        TEXT NOT NULL,    -- list[float] in seconds on the source variant's timeline
    source_track_audio_id  INTEGER,          -- which track_audio row's audio was analysed
    source_variant_tag     TEXT,             -- 'original' / 'instrumental' / ... whichever was used
    cue_detr_sensitivity   REAL,             -- cue-detr threshold used at inference
    computed_at            DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_track_audio_id) REFERENCES track_audio(track_audio_id) ON DELETE SET NULL
);


-- Raw fingerprint hits found by sliding chromaprint over the full mix.
-- Consumed by stage-3 MAP inference as an identity likelihood. Kept as
-- raw hits (before row-resolution) so multiple rows can claim evidence
-- from the same hit block.
CREATE TABLE IF NOT EXISTS set_fingerprint_hits (
    set_id            TEXT NOT NULL,
    mix_start_s       REAL NOT NULL,
    mix_end_s         REAL NOT NULL,
    matched_track_id  TEXT NOT NULL,
    matched_variant   TEXT NOT NULL DEFAULT 'original',
    score             REAL NOT NULL,         -- 0..1 chromaprint similarity
    detected_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (set_id, mix_start_s, matched_track_id, matched_variant),
    FOREIGN KEY (set_id) REFERENCES dj_sets(set_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_set_fp_hits_set   ON set_fingerprint_hits(set_id);
CREATE INDEX IF NOT EXISTS idx_set_fp_hits_track ON set_fingerprint_hits(matched_track_id);


-- =============================================================================
-- SECTION STRUCTURE (alias view over track_mert_sections for clarity;
-- exposes section boundaries without forcing callers to touch the
-- embedding BLOB). Populated identically by analysis pipeline.
-- =============================================================================
CREATE TABLE IF NOT EXISTS track_sections (
    track_audio_id INTEGER NOT NULL,
    section_idx    INTEGER NOT NULL,
    start_s        REAL NOT NULL,
    end_s          REAL NOT NULL,
    kind           TEXT,                      -- 'intro' | 'verse' | 'chorus' | 'bridge' | 'drop' | NULL (not yet labeled)
    PRIMARY KEY (track_audio_id, section_idx),
    FOREIGN KEY (track_audio_id) REFERENCES track_audio(track_audio_id) ON DELETE CASCADE
);


-- DJ-set section alignment: maps each played section back onto a reference track
-- with an optional "cutup plan" describing measure-level rearrangements.
CREATE TABLE IF NOT EXISTS set_section_alignment (
    set_id                  TEXT NOT NULL,
    section_idx             INTEGER NOT NULL, -- section ordinal within the set
    set_start_s             REAL NOT NULL,
    set_end_s               REAL NOT NULL,
    ref_track_id            TEXT,              -- canonical track_id we aligned to
    ref_start_s             REAL,              -- where in the ref the match starts (seconds). CCC path populates this; DTW leaves NULL.
    ref_end_s               REAL,              -- where in the ref the match ends
    ref_section_idx         INTEGER,           -- which track_mert_sections row the match primarily covers
    transposition_semitones INTEGER,           -- -3..+3 or NULL if unknown
    bpm_ratio               REAL,              -- played_bpm / ref_bpm
    cutup_plan_json         TEXT,              -- list[Segment] as JSON
    confidence              REAL,
    stem_match_rates_json   TEXT,              -- {'full': r, 'vocals': r, 'drums': r, 'bass': r, 'other': r}
    confidence_source       TEXT DEFAULT 'legacy',  -- 'legacy' | 'indicators_sota_v1' | ... — provenance tag
    aligned_at              DATETIME DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (set_id, section_idx),
    FOREIGN KEY (set_id) REFERENCES dj_sets(set_id) ON DELETE CASCADE
);
