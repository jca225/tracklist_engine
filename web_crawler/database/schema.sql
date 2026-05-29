
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
    -- Edit length (Variant axis): 'regular' (radio/album cut) | 'extended'
    -- (extended mix, typically the DJ club edit). Independent of variant_tag
    -- (which is the Stem axis). See [[audio-identity-taxonomy]].
    edit_tag       TEXT NOT NULL DEFAULT 'regular',
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
    key_strength      REAL,                   -- 0..1 KeyExtractor correlation peak vs runner-up
    bpm               REAL,
    time_sig_num      INTEGER,
    time_sig_den      INTEGER,
    lufs              REAL,
    danceability      REAL,
    energy            REAL,
    valence           REAL,
    acousticness      REAL,                   -- 0..1, P(acoustic) from Essentia mood_acoustic
    instrumentalness  REAL,                   -- 0..1, 1 - P(voice) from voice_instrumental
    speechiness       REAL,                   -- 0..1, max P(speech | conversation) from YAMNet
    liveness          REAL,                   -- 0..1, peak P(applause | cheering | crowd) from YAMNet
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


-- Per-measure mean-pooled MERT embeddings. One row per beat_this-derived
-- measure of the original full-mix audio. embedding is float16 (dim,) bytes.
-- This is the raw cache; the BPE cue-point optimizer (Phase 8b) reaggregates
-- ranges of these into post-BPE section embeddings without rerunning MERT.
CREATE TABLE IF NOT EXISTS track_mert_measures (
    track_audio_id INTEGER NOT NULL,
    measure_idx    INTEGER NOT NULL,
    start_s        REAL NOT NULL,
    end_s          REAL NOT NULL,
    dim            INTEGER NOT NULL,
    dtype          TEXT NOT NULL,             -- 'float16'
    embedding      BLOB NOT NULL,             -- (dim,) bytes, mean-pooled
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (track_audio_id, measure_idx),
    FOREIGN KEY (track_audio_id) REFERENCES track_audio(track_audio_id) ON DELETE CASCADE
);


-- MERT embeddings stored per cue-delimited section. Populated post-BPE (the
-- cue-point optimizer in Phase 8b decides section boundaries, then reads
-- track_mert_measures and writes mean-pooled section embeddings here).
-- Pre-BPE, this table is empty. embedding is float16 bytes; shape is
-- (n_frames, dim) for legacy raw storage or (dim,) when n_frames=1 mean-pool.
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
-- RESERVED — no live writer: the upsert_timeline/load_set_timeline accessors
-- (and the SetTimeline model) were removed with the Viterbi aligner (2026-05).
-- Table kept for the future alignment write-back; see core/CLAUDE.md.
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
-- RESERVED — no live writer: upsert_measure_alignment_rows was removed with the
-- Viterbi aligner (2026-05). Table kept for the future alignment write-back.
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
-- RESERVED — no live writer (never had one): output target for the future
-- alignment write-back, kept alongside measure_alignment.
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
-- RESERVED — no live writer: upsert_section_alignment was removed with the
-- Viterbi aligner (2026-05). Table kept for the future alignment write-back.
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
    confidence_source       TEXT DEFAULT 'legacy',  -- provenance tag (e.g. 'legacy', 'human_label')
    aligned_at              DATETIME DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (set_id, section_idx),
    FOREIGN KEY (set_id) REFERENCES dj_sets(set_id) ON DELETE CASCADE
);


-- =============================================================================
-- TOKEN MATERIALIZATION
-- Output of tokenizer/materialize.py running over dj_set_rows.raw_html.
-- All three tables are rebuildable from scratch (re-running materialize
-- drops + repopulates).
-- =============================================================================

-- Confirmed track identity from track_tokenizer.TrackRow where is_ided=True.
-- One row per track_id, aggregated across every set the track appears in.
CREATE TABLE IF NOT EXISTS track_metadata (
    track_id          TEXT PRIMARY KEY,
    title             TEXT,
    artists_json      TEXT,                 -- JSON array of artist names
    full_name         TEXT,                 -- meta itemprop=name, e.g. "Artist - Title"
    genre             TEXT,
    duration_seconds  INTEGER,
    is_remixish       INTEGER DEFAULT 0,
    version_tag       TEXT,                 -- Remix | Rework | Acappella | AltVersion | NULL
    has_youtube       INTEGER DEFAULT 0,
    has_soundcloud    INTEGER DEFAULT 0,
    has_spotify       INTEGER DEFAULT 0,
    has_apple         INTEGER DEFAULT 0,
    plays_total       INTEGER,              -- max plays observed across sets
    set_count         INTEGER,              -- distinct sets this track appears in
    artwork_url       TEXT,
    updated_at        DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_track_metadata_title ON track_metadata(title);


-- Per-set ordered slot spine — the symbolic X for the alignment model (the
-- per-track sibling track_metadata is the download view; this is the aligner
-- view). One row per played slot, in 1001tracklists order, preserving the
-- distinctions the aligner conditions on: version-as-played, instrumental
-- qualifier, cue claim, and w/ adjacency. track_id is the data-trackid when
-- present, else a synthetic 'tlp{tlp_id}' (source='synthetic') so sided rows
-- with no global track id (the "Rvmor gap") still appear. full_name is verbatim
-- (carries the (Instrumental)/(Acappella) qualifier the download query strips).
CREATE TABLE IF NOT EXISTS set_track_slots (
    set_id           TEXT NOT NULL,
    row_index        INTEGER NOT NULL,     -- order within the set (from dj_set_rows)
    tlp_id           INTEGER,              -- data-id on the row
    track_id         TEXT,                 -- data-trackid OR synthetic 'tlp{tlp_id}'
    source           TEXT DEFAULT 'scraped', -- 'scraped' | 'synthetic'
    slot_label       TEXT,                 -- '030', '030w1' (section + w/ layering)
    is_concurrent    INTEGER DEFAULT 0,    -- w/ adjacency (layered under a primary)
    cue_seconds      INTEGER,              -- hidden-input cue offset
    cue_time_seconds INTEGER,              -- parsed from the displayed timecode
    version_tag      TEXT,                 -- Remix | Rework | Acappella | AltVersion | NULL
    is_instrumental  INTEGER DEFAULT 0,    -- (Instrumental) qualifier seen as-played
    full_name        TEXT,                 -- verbatim "Artist - Title (Remixer Remix) (Instrumental)"
    title            TEXT,
    artists_json     TEXT,
    duration_seconds INTEGER,
    parsed_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (set_id, row_index),
    FOREIGN KEY (set_id) REFERENCES dj_sets(set_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_set_track_slots_track ON set_track_slots(track_id);


-- User-contributed track identity guesses from suggestion_tokenizer.SuggestionRow.
-- Many suggestions can target the same track slot (set_id, tlp_id).
CREATE TABLE IF NOT EXISTS track_suggestions (
    sug_id                INTEGER PRIMARY KEY,
    set_id                TEXT NOT NULL,
    tlp_id                INTEGER,
    pos                   INTEGER,
    track_slug            TEXT,             -- when suggestion points at an existing track_id
    track_display         TEXT,             -- raw "Artist - Title (Remix)"
    artist_title          TEXT,             -- cleaned
    suggester_user_id     INTEGER,
    suggester_name        TEXT,
    suggestion_timestamp  TEXT,
    is_remix              INTEGER,
    has_youtube           INTEGER,
    has_soundcloud        INTEGER,
    has_spotify           INTEGER,
    parsed_at             DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_track_sug_set  ON track_suggestions(set_id);
CREATE INDEX IF NOT EXISTS idx_track_sug_tlp  ON track_suggestions(tlp_id);
CREATE INDEX IF NOT EXISTS idx_track_sug_slug ON track_suggestions(track_slug);


-- Cross-tracklist linkage hints from id_tokenizer.IDTrack.linked_items.
-- For an unidentified track slot in set X, users link to set Y where the
-- same track *is* identified — useful to resolve IDs by transitivity.
CREATE TABLE IF NOT EXISTS track_id_links (
    set_id                  TEXT NOT NULL,
    tlp_id                  INTEGER NOT NULL,
    linker_user_name        TEXT,
    linker_user_href        TEXT,
    linker_user_followers   TEXT,
    linked_tracklist_href   TEXT,
    linked_tracklist_text   TEXT,
    parsed_at               DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (set_id, tlp_id, linker_user_name, linked_tracklist_href)
);
CREATE INDEX IF NOT EXISTS idx_track_id_links_target ON track_id_links(linked_tracklist_href);


-- Append-only correction ledger. One row each time a track's downloaded audio
-- is replaced or a variant added because the auto-acquired version was the
-- wrong identity along one of the three axes:
--   axis='version'  wrong arrangement (got the original, wanted the remix)
--   axis='variant'  wrong edit length (got radio/regular, wanted extended)
--   axis='stem'     wrong/poor component (acappella vs instrumental vs full)
-- These rows are the training signal for the future acquisition gates. NO
-- foreign keys on purpose: a correction must OUTLIVE the track_audio rows it
-- references (a 'replace' deletes the old row), so it snapshots the old/new
-- identity inline rather than pointing at rows that may vanish.
CREATE TABLE IF NOT EXISTS track_audio_correction (
    correction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id              TEXT,              -- set where the mistake was noticed (nullable)
    position            TEXT,              -- published section no. / slot label (free text)
    track_id            TEXT NOT NULL,     -- 1001tracklists data-trackid being corrected
    axis                TEXT NOT NULL,     -- version | variant | stem
    action              TEXT NOT NULL,     -- replace (destructive) | add (additive)
    old_track_audio_id  INTEGER,           -- retired row id (NULL for a pure add)
    old_platform        TEXT,
    old_player_id       TEXT,
    old_url             TEXT,
    new_track_audio_id  INTEGER,           -- inserted row id
    new_platform        TEXT,
    new_player_id       TEXT,
    new_url             TEXT,
    variant_tag         TEXT,              -- stem-axis value (acappella|instrumental|original)
    reason              TEXT,              -- free-text why it was wrong
    source              TEXT,              -- replace_track_audio | acquire_variant | manual
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK (axis IN ('version','variant','stem')),
    CHECK (action IN ('replace','add'))
);
CREATE INDEX IF NOT EXISTS idx_track_audio_correction_track ON track_audio_correction(track_id);
CREATE INDEX IF NOT EXISTS idx_track_audio_correction_set   ON track_audio_correction(set_id);
