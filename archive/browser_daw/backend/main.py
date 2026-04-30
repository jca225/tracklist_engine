from __future__ import annotations

import json
import sqlite3
import urllib.parse
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "db" / "music_database.db"
DEFAULT_PROJECT_NAME = "Default Project"
CACHE_DIR = REPO_ROOT / "browser_daw" / "cache"
TRACK_META_CACHE_PATH = CACHE_DIR / "track_metadata_cache.json"
AUDIO_DRIVE_TRACKS_DIR = Path("/Users/johnnycabrahams/Desktop/tracklist_audio_drive/tracks")

_FOLDER_TO_VARIANT: dict[str, str] = {
    "canonical_originals": "original",
    "2nvzlh2k": "acappella",
    "1d9zwh49": "instrumental",
}

app = FastAPI(title="Browser DAW API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateProjectBody(BaseModel):
    name: str = Field(default=DEFAULT_PROJECT_NAME, min_length=1, max_length=120)
    master_bpm: float = Field(default=128.0, ge=60.0, le=220.0)
    master_camelot: str | None = Field(default=None)


class UpdateProjectBody(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    master_bpm: float | None = Field(default=None, ge=60.0, le=220.0)
    master_camelot: str | None = Field(default=None)


class AddClipBody(BaseModel):
    track_id: str
    variant_tag: str | None = None
    lane_idx: int = Field(default=1, ge=1, le=256)
    timeline_start_s: float = Field(default=0.0, ge=0.0)
    src_start_s: float = Field(default=0.0, ge=0.0)
    src_end_s: float = Field(default=32.0, gt=0.0)
    auto_sync: bool = True


class UpdateClipBody(BaseModel):
    lane_idx: int | None = Field(default=None, ge=1, le=256)
    timeline_start_s: float | None = Field(default=None, ge=0.0)
    src_start_s: float | None = Field(default=None, ge=0.0)
    src_end_s: float | None = Field(default=None, gt=0.0)
    tempo_ratio: float | None = Field(default=None, gt=0.0)
    pitch_shift_semi: int | None = Field(default=None, ge=-12, le=12)
    variant_tag: str | None = None


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


_CAMELOT_TO_KEY: dict[str, tuple[int, str]] = {
    "1A": (8, "minor"), "2A": (3, "minor"), "3A": (10, "minor"), "4A": (5, "minor"),
    "5A": (0, "minor"), "6A": (7, "minor"), "7A": (2, "minor"), "8A": (9, "minor"),
    "9A": (4, "minor"), "10A": (11, "minor"), "11A": (6, "minor"), "12A": (1, "minor"),
    "1B": (11, "major"), "2B": (6, "major"), "3B": (1, "major"), "4B": (8, "major"),
    "5B": (3, "major"), "6B": (10, "major"), "7B": (5, "major"), "8B": (0, "major"),
    "9B": (7, "major"), "10B": (2, "major"), "11B": (9, "major"), "12B": (4, "major"),
}
_KEY_TO_CAMELOT: dict[tuple[int, str], str] = {v: k for k, v in _CAMELOT_TO_KEY.items()}


def _normalize_camelot(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().upper()
    if text in _CAMELOT_TO_KEY:
        return text
    return None


def _camelot_to_key(value: str | None) -> tuple[int | None, str | None]:
    key = _normalize_camelot(value)
    if key is None:
        return (None, None)
    pc, mode = _CAMELOT_TO_KEY[key]
    return (pc, mode)


def _key_to_camelot(key_pc: int | None, key_mode: str | None) -> str | None:
    if key_pc is None or key_mode is None:
        return None
    mode = str(key_mode).lower()
    if mode not in {"major", "minor"}:
        return None
    return _KEY_TO_CAMELOT.get((int(key_pc), mode))


def _nearest_pitch_shift(from_key_pc: int | None, to_key_pc: int | None) -> int:
    if from_key_pc is None or to_key_pc is None:
        return 0
    forward = (to_key_pc - from_key_pc) % 12
    backward = forward - 12
    return backward if abs(backward) < abs(forward) else forward


def _ensure_browser_tables() -> None:
    sql = """
    CREATE TABLE IF NOT EXISTS browser_daw_projects (
        project_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        master_bpm REAL NOT NULL DEFAULT 128.0,
        master_key_pc INTEGER,
        master_key_mode TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS browser_daw_clips (
        clip_id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        track_id TEXT NOT NULL,
        lane_idx INTEGER NOT NULL DEFAULT 1,
        timeline_start_s REAL NOT NULL DEFAULT 0.0,
        src_start_s REAL NOT NULL DEFAULT 0.0,
        src_end_s REAL NOT NULL DEFAULT 32.0,
        tempo_ratio REAL NOT NULL DEFAULT 1.0,
        pitch_shift_semi INTEGER NOT NULL DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES browser_daw_projects(project_id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_browser_daw_clips_project
        ON browser_daw_clips(project_id);
    """
    with _connect() as conn:
        conn.executescript(sql)
        cols = conn.execute("PRAGMA table_info(browser_daw_clips)").fetchall()
        col_names = {str(c["name"]) for c in cols}
        if "variant_tag" not in col_names:
            conn.execute("ALTER TABLE browser_daw_clips ADD COLUMN variant_tag TEXT NOT NULL DEFAULT 'original'")
        proj_cols = conn.execute("PRAGMA table_info(browser_daw_projects)").fetchall()
        proj_col_names = {str(c["name"]) for c in proj_cols}
        if "master_camelot" not in proj_col_names:
            conn.execute("ALTER TABLE browser_daw_projects ADD COLUMN master_camelot TEXT")
        conn.commit()


def _ensure_default_project() -> int:
    _ensure_browser_tables()
    with _connect() as conn:
        row = conn.execute(
            "SELECT project_id FROM browser_daw_projects ORDER BY project_id LIMIT 1"
        ).fetchone()
        if row is not None:
            return int(row["project_id"])
        cur = conn.execute(
            """
            INSERT INTO browser_daw_projects(name, master_bpm, master_key_pc, master_key_mode, master_camelot)
            VALUES (?, 128.0, NULL, NULL, NULL)
            """,
            (DEFAULT_PROJECT_NAME,),
        )
        conn.commit()
        return int(cur.lastrowid)


def _track_sync_info(conn: sqlite3.Connection, track_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT
            ta.track_id,
            taf.bpm,
            taf.key_pc,
            taf.key_mode
        FROM track_audio ta
        LEFT JOIN track_audio_features taf
            ON taf.track_audio_id = ta.track_audio_id
            AND taf.source = 'audio_pipeline_v1'
        WHERE ta.track_id = ?
        ORDER BY ta.is_reference DESC, ta.downloaded_at DESC
        LIMIT 1
        """,
        (track_id,),
    ).fetchone()


def _preferred_track_audio_row(conn: sqlite3.Connection, track_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT track_id, path, platform, player_id, codec, variant_tag
        FROM track_audio
        WHERE track_id = ?
        ORDER BY is_reference DESC, downloaded_at DESC
        LIMIT 1
        """,
        (track_id,),
    ).fetchone()


def _pick_track_audio_row(
    conn: sqlite3.Connection, track_id: str, variant_tag: str | None
) -> sqlite3.Row | None:
    if variant_tag:
        row = conn.execute(
            """
            SELECT track_id, path, platform, player_id, codec, variant_tag
            FROM track_audio
            WHERE track_id = ? AND variant_tag = ?
            ORDER BY is_reference DESC, downloaded_at DESC
            LIMIT 1
            """,
            (track_id, variant_tag),
        ).fetchone()
        if row is not None:
            return row
    return _preferred_track_audio_row(conn, track_id)


def _preferred_track_audio_id(conn: sqlite3.Connection, track_id: str) -> int | None:
    row = conn.execute(
        """
        SELECT track_audio_id
        FROM track_audio
        WHERE track_id = ?
        ORDER BY is_reference DESC, downloaded_at DESC
        LIMIT 1
        """,
        (track_id,),
    ).fetchone()
    if row is None:
        return None
    return int(row["track_audio_id"])


def _track_stem_paths(conn: sqlite3.Connection, track_id: str) -> dict[str, Path]:
    audio_id = _preferred_track_audio_id(conn, track_id)
    if audio_id is None:
        return {}
    rows = conn.execute(
        """
        SELECT stem_name, path
        FROM track_stems
        WHERE track_audio_id = ?
        """,
        (audio_id,),
    ).fetchall()
    out: dict[str, Path] = {}
    for row in rows:
        name = str(row["stem_name"])
        out[name] = Path(row["path"])
    return out


def _filesystem_variant_files(track_id: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not AUDIO_DRIVE_TRACKS_DIR.exists():
        return out
    pattern = f"{track_id}__*"
    for folder in AUDIO_DRIVE_TRACKS_DIR.iterdir():
        if not folder.is_dir():
            continue
        matches = sorted(folder.glob(pattern))
        if not matches:
            continue
        variant = _FOLDER_TO_VARIANT.get(folder.name, folder.name)
        out.setdefault(variant, matches[0])
    return out


def _parse_track_title(raw_title: str | None) -> tuple[str | None, str | None]:
    if not raw_title:
        return (None, None)
    text = raw_title.strip()
    if " - " in text:
        artist_part, song_part = text.split(" - ", 1)
        return (song_part.strip() or None, artist_part.strip() or None)
    return (text, None)


def _load_track_meta_cache() -> dict[str, dict]:
    if not TRACK_META_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(TRACK_META_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_track_meta_cache(cache: dict[str, dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TRACK_META_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _extract_name_from_rows(conn: sqlite3.Connection, track_id: str) -> tuple[str | None, str | None]:
    rows = conn.execute(
        """
        SELECT r.text_excerpt, r.raw_html
        FROM dj_set_track_media_links m
        JOIN dj_set_rows r
          ON r.set_id = m.set_id
         AND (r.element_id = m.tlp_id OR r.raw_html LIKE '%' || m.tlp_id || '%')
        WHERE m.track_id = ?
          AND r.raw_html IS NOT NULL
        ORDER BY r.row_index
        LIMIT 40
        """,
        (track_id,),
    ).fetchall()
    for row in rows:
        excerpt = (row["text_excerpt"] or "").strip()
        if excerpt:
            song_name, artist_name = _parse_track_title(excerpt)
            if song_name is not None:
                return (song_name, artist_name)
        raw_html = row["raw_html"] or ""
        marker = 'title="'
        if marker in raw_html:
            start = raw_html.find(marker) + len(marker)
            end = raw_html.find('"', start)
            if end > start:
                song_name, artist_name = _parse_track_title(raw_html[start:end].strip())
                if song_name is not None:
                    return (song_name, artist_name)
    return (None, None)


def _youtube_oembed_title(player_id: str | None) -> str | None:
    if not player_id:
        return None
    watch_url = f"https://www.youtube.com/watch?v={player_id}"
    endpoint = "https://www.youtube.com/oembed?url={}&format=json".format(urllib.parse.quote(watch_url, safe=":/?=&"))
    try:
        with urllib.request.urlopen(endpoint, timeout=2.5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        title = payload.get("title")
        return str(title).strip() if title else None
    except Exception:
        return None


def _resolve_track_metadata_dynamic(
    conn: sqlite3.Connection,
    *,
    track_id: str,
    platform: str | None,
    player_id: str | None,
    scraped_track_title: str | None,
) -> tuple[str | None, str | None, str]:
    cache = _load_track_meta_cache()
    cached = cache.get(track_id)
    if cached is not None:
        return (cached.get("song_name"), cached.get("artist_name"), cached.get("source", "cache"))

    # 1) Existing scraped failure titles.
    if scraped_track_title:
        song_name, artist_name = _parse_track_title(scraped_track_title)
        if song_name:
            cache[track_id] = {"song_name": song_name, "artist_name": artist_name, "source": "scrape_failures"}
            _save_track_meta_cache(cache)
            return (song_name, artist_name, "scrape_failures")

    # 2) Parse matching track rows from scraped HTML.
    song_name, artist_name = _extract_name_from_rows(conn, track_id)
    if song_name:
        cache[track_id] = {"song_name": song_name, "artist_name": artist_name, "source": "dj_set_rows"}
        _save_track_meta_cache(cache)
        return (song_name, artist_name, "dj_set_rows")

    # 3) Lightweight provider lookup for YouTube.
    if (platform or "").lower() == "youtube":
        yt_title = _youtube_oembed_title(player_id)
        if yt_title:
            song_name, artist_name = _parse_track_title(yt_title)
            song_name = song_name or yt_title
            cache[track_id] = {"song_name": song_name, "artist_name": artist_name, "source": "youtube_oembed"}
            _save_track_meta_cache(cache)
            return (song_name, artist_name, "youtube_oembed")

    return (None, None, "unresolved")


@app.get("/health")
def health() -> dict:
    project_id = _ensure_default_project()
    return {
        "ok": True,
        "db_path": str(DB_PATH),
        "db_exists": DB_PATH.exists(),
        "default_project_id": project_id,
    }


@app.get("/tracks")
def tracks(limit: int = Query(default=100, ge=1, le=1000)) -> dict:
    sql = """
        SELECT
            ta.track_id,
            ta.path,
            ta.player_id,
            taf.bpm,
            taf.key_pc,
            taf.key_mode,
            ta.platform,
            ta.variant_tag,
            ta.duration_s,
            ta.codec,
            ta.bitrate_kbps,
            ti.musicbrainz_recording_id,
            ti.isrc,
            ti.release_label,
            sf.track_title AS scraped_track_title
        FROM track_audio ta
        LEFT JOIN track_audio_features taf
            ON taf.track_audio_id = ta.track_audio_id
            AND taf.source = 'audio_pipeline_v1'
        LEFT JOIN track_identity ti
            ON ti.track_id = ta.track_id
        LEFT JOIN (
            SELECT track_id, MAX(failure_id) AS max_failure_id
            FROM scrape_failures
            WHERE track_id IS NOT NULL AND track_id != '' AND track_title IS NOT NULL AND track_title != ''
            GROUP BY track_id
        ) sf_latest
            ON sf_latest.track_id = ta.track_id
        LEFT JOIN scrape_failures sf
            ON sf.failure_id = sf_latest.max_failure_id
        ORDER BY ta.downloaded_at DESC
        LIMIT ?
    """
    with _connect() as conn:
        variants_rows = conn.execute(
            """
            SELECT track_id, variant_tag
            FROM track_audio
            WHERE variant_tag IS NOT NULL AND variant_tag != ''
            """
        ).fetchall()
        variants_by_track: dict[str, list[str]] = {}
        for r in variants_rows:
            variants_by_track.setdefault(r["track_id"], [])
            if r["variant_tag"] not in variants_by_track[r["track_id"]]:
                variants_by_track[r["track_id"]].append(r["variant_tag"])

        rows = conn.execute(sql, (limit,)).fetchall()
        items: list[dict] = []
        seen_track_ids: set[str] = set()
        for row in rows:
            track_id = row["track_id"]
            if track_id in seen_track_ids:
                continue
            seen_track_ids.add(track_id)
            song_name, artist_name, meta_source = _resolve_track_metadata_dynamic(
                conn=conn,
                track_id=track_id,
                platform=row["platform"],
                player_id=row["player_id"],
                scraped_track_title=row["scraped_track_title"],
            )
            bpm_int = int(round(float(row["bpm"]))) if row["bpm"] is not None else None
            available_variants = variants_by_track.get(track_id, [])
            fs_variants = _filesystem_variant_files(track_id)
            for v in fs_variants.keys():
                if v not in available_variants:
                    available_variants.append(v)
            stem_paths = _track_stem_paths(conn, track_id)
            stem_names = set(stem_paths.keys())
            if "vocals" in stem_names and "acappella" not in available_variants:
                available_variants.append("acappella")
            if {"drums", "bass", "other"}.issubset(stem_names) and "instrumental" not in available_variants:
                available_variants.append("instrumental")
            if "original" in available_variants:
                available_variants = ["original"] + [v for v in available_variants if v != "original"]
            items.append(
                {
                    "track_id": track_id,
                    "song_name": song_name,
                    "artist_name": artist_name,
                    "audio_path": row["path"],
                    "bpm": bpm_int,
                    "key_pc": row["key_pc"],
                    "key_mode": row["key_mode"],
                    "platform": row["platform"],
                    "variant_tag": row["variant_tag"],
                    "duration_s": row["duration_s"],
                    "codec": row["codec"],
                    "bitrate_kbps": row["bitrate_kbps"],
                    "musicbrainz_recording_id": row["musicbrainz_recording_id"],
                    "isrc": row["isrc"],
                    "release_label": row["release_label"],
                    "scraped_track_title": row["scraped_track_title"],
                    "metadata_source": meta_source,
                    "available_variants": available_variants,
                    "filesystem_variants": sorted(fs_variants.keys()),
                    "available_stems": sorted(stem_names),
                }
            )
    return {"items": items}


@app.post("/tracks/metadata-cache/clear")
def clear_metadata_cache() -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TRACK_META_CACHE_PATH.write_text("{}", encoding="utf-8")
    return {"ok": True}


@app.get("/tracks/{track_id}/analysis")
def track_analysis(track_id: str) -> dict:
    sql = """
        SELECT ta.track_id, tan.measure_times_json, tan.cue_points_json
        FROM track_audio ta
        JOIN track_analysis tan
            ON tan.track_audio_id = ta.track_audio_id
        WHERE ta.track_id = ?
        ORDER BY ta.is_reference DESC, ta.downloaded_at DESC
        LIMIT 1
    """
    with _connect() as conn:
        row = conn.execute(sql, (track_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No analysis found for track_id={track_id}")

    try:
        measure_times = json.loads(row["measure_times_json"] or "[]")
        cue_points = json.loads(row["cue_points_json"] or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in analysis rows: {exc}") from exc

    return {
        "track_id": row["track_id"],
        "measure_times": measure_times,
        "cue_points": cue_points,
    }


@app.get("/tracks/{track_id}/audio")
def track_audio_file(track_id: str, variant_tag: str | None = Query(default=None)):
    with _connect() as conn:
        row = _pick_track_audio_row(conn, track_id, variant_tag)
    path: Path | None = None
    if row is not None:
        path = Path(row["path"])
    if path is None or not path.exists() or not path.is_file():
        fs_variants = _filesystem_variant_files(track_id)
        requested = (variant_tag or "").strip().lower()
        if requested and requested in fs_variants:
            path = fs_variants[requested]
        elif "original" in fs_variants:
            path = fs_variants["original"]
        elif fs_variants:
            path = next(iter(fs_variants.values()))
    if path is None:
        raise HTTPException(status_code=404, detail=f"No audio row or filesystem file found for track_id={track_id}")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file missing on disk for track_id={track_id}")

    suffix = path.suffix.lower()
    media_type = "application/octet-stream"
    if suffix in {".m4a", ".mp4"}:
        media_type = "audio/mp4"
    elif suffix == ".mp3":
        media_type = "audio/mpeg"
    elif suffix == ".wav":
        media_type = "audio/wav"
    elif suffix == ".flac":
        media_type = "audio/flac"

    return FileResponse(path=path, media_type=media_type, filename=path.name)


@app.get("/tracks/{track_id}/stems")
def track_stems(track_id: str) -> dict:
    with _connect() as conn:
        paths = _track_stem_paths(conn, track_id)
    items = []
    for stem_name, path in sorted(paths.items()):
        items.append(
            {
                "stem_name": stem_name,
                "exists": bool(path.exists() and path.is_file()),
                "url": f"/tracks/{track_id}/stems/{stem_name}/audio",
            }
        )
    return {"track_id": track_id, "items": items}


@app.get("/tracks/{track_id}/stems/{stem_name}/audio")
def track_stem_audio_file(track_id: str, stem_name: str):
    with _connect() as conn:
        paths = _track_stem_paths(conn, track_id)
    path = paths.get(stem_name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Stem {stem_name} not found for track_id={track_id}")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Stem file missing on disk for track_id={track_id}, stem={stem_name}")

    suffix = path.suffix.lower()
    media_type = "application/octet-stream"
    if suffix in {".m4a", ".mp4"}:
        media_type = "audio/mp4"
    elif suffix == ".mp3":
        media_type = "audio/mpeg"
    elif suffix == ".wav":
        media_type = "audio/wav"
    elif suffix == ".flac":
        media_type = "audio/flac"
    return FileResponse(path=path, media_type=media_type, filename=path.name)


@app.get("/projects")
def list_projects() -> dict:
    _ensure_default_project()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT project_id, name, master_bpm, master_camelot
            FROM browser_daw_projects
            ORDER BY project_id
            """
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["master_camelot"] = _normalize_camelot(item.get("master_camelot"))
        items.append(item)
    return {"items": items}


@app.post("/projects")
def create_project(body: CreateProjectBody) -> dict:
    camelot = _normalize_camelot(body.master_camelot)
    key_pc, key_mode = _camelot_to_key(camelot)
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO browser_daw_projects(name, master_bpm, master_key_pc, master_key_mode, master_camelot)
            VALUES (?, ?, ?, ?, ?)
            """,
            (body.name.strip(), float(body.master_bpm), key_pc, key_mode, camelot),
        )
        conn.commit()
        row = conn.execute(
            """
            SELECT project_id, name, master_bpm, master_camelot
            FROM browser_daw_projects
            WHERE project_id = ?
            """,
            (int(cur.lastrowid),),
        ).fetchone()
    return {"item": dict(row)}


@app.patch("/projects/{project_id}")
def update_project(project_id: int, body: UpdateProjectBody) -> dict:
    with _connect() as conn:
        current = conn.execute(
            """
            SELECT project_id, name, master_bpm, master_camelot
            FROM browser_daw_projects
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
        if current is None:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
        name = body.name.strip() if body.name is not None else current["name"]
        master_bpm = float(body.master_bpm) if body.master_bpm is not None else float(current["master_bpm"])
        master_camelot = _normalize_camelot(body.master_camelot) if body.master_camelot is not None else _normalize_camelot(current["master_camelot"])
        master_key_pc, master_key_mode = _camelot_to_key(master_camelot)
        conn.execute(
            """
            UPDATE browser_daw_projects
            SET name = ?, master_bpm = ?, master_key_pc = ?, master_key_mode = ?, master_camelot = ?, updated_at = CURRENT_TIMESTAMP
            WHERE project_id = ?
            """,
            (name, master_bpm, master_key_pc, master_key_mode, master_camelot, project_id),
        )
        conn.commit()
        updated = conn.execute(
            """
            SELECT project_id, name, master_bpm, master_camelot
            FROM browser_daw_projects
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
    item = dict(updated)
    item["master_camelot"] = _normalize_camelot(item.get("master_camelot"))
    return {"item": item}


@app.get("/projects/{project_id}/clips")
def list_project_clips(project_id: int) -> dict:
    with _connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM browser_daw_projects WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        if exists is None:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
        rows = conn.execute(
            """
            SELECT clip_id, project_id, track_id, lane_idx, timeline_start_s,
                   src_start_s, src_end_s, tempo_ratio, pitch_shift_semi, variant_tag
            FROM browser_daw_clips
            WHERE project_id = ?
            ORDER BY lane_idx, timeline_start_s, clip_id
            """,
            (project_id,),
        ).fetchall()
    return {"items": [dict(row) for row in rows]}


@app.post("/projects/{project_id}/clips")
def add_clip(project_id: int, body: AddClipBody) -> dict:
    if body.src_end_s <= body.src_start_s:
        raise HTTPException(status_code=400, detail="src_end_s must be > src_start_s")
    with _connect() as conn:
        project = conn.execute(
            """
            SELECT project_id, master_bpm, master_key_pc, master_key_mode, master_camelot
            FROM browser_daw_projects
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
        if project is None:
            raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
        track = _track_sync_info(conn, body.track_id)
        if track is None:
            raise HTTPException(status_code=404, detail=f"Track not found: {body.track_id}")

        tempo_ratio = 1.0
        pitch_shift = 0
        if body.auto_sync:
            t_bpm = track["bpm"]
            if t_bpm is not None and float(t_bpm) > 0:
                tempo_ratio = float(project["master_bpm"]) / float(t_bpm)
            project_key_pc = project["master_key_pc"]
            if project_key_pc is None:
                project_key_pc, _ = _camelot_to_key(project["master_camelot"])
            track_key_pc = track["key_pc"]
            pitch_shift = _nearest_pitch_shift(track_key_pc, project_key_pc)

        resolved_variant = (body.variant_tag or "original").strip().lower()
        chosen_audio_row = _pick_track_audio_row(conn, body.track_id, resolved_variant)
        if chosen_audio_row is not None:
            resolved_variant = str(chosen_audio_row["variant_tag"])
        else:
            fs_variants = _filesystem_variant_files(body.track_id)
            if resolved_variant not in fs_variants:
                resolved_variant = "original" if "original" in fs_variants else resolved_variant

        cur = conn.execute(
            """
            INSERT INTO browser_daw_clips(
                project_id, track_id, lane_idx, timeline_start_s, src_start_s, src_end_s, tempo_ratio, pitch_shift_semi, variant_tag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                body.track_id,
                body.lane_idx,
                body.timeline_start_s,
                body.src_start_s,
                body.src_end_s,
                tempo_ratio,
                pitch_shift,
                resolved_variant,
            ),
        )
        conn.commit()
        row = conn.execute(
            """
            SELECT clip_id, project_id, track_id, lane_idx, timeline_start_s,
                   src_start_s, src_end_s, tempo_ratio, pitch_shift_semi, variant_tag
            FROM browser_daw_clips WHERE clip_id = ?
            """,
            (int(cur.lastrowid),),
        ).fetchone()
    return {"item": dict(row)}


@app.patch("/clips/{clip_id}")
def update_clip(clip_id: int, body: UpdateClipBody) -> dict:
    with _connect() as conn:
        current = conn.execute(
            """
            SELECT clip_id, project_id, track_id, lane_idx, timeline_start_s,
                   src_start_s, src_end_s, tempo_ratio, pitch_shift_semi, variant_tag
            FROM browser_daw_clips
            WHERE clip_id = ?
            """,
            (clip_id,),
        ).fetchone()
        if current is None:
            raise HTTPException(status_code=404, detail=f"Clip not found: {clip_id}")
        lane_idx = body.lane_idx if body.lane_idx is not None else current["lane_idx"]
        timeline_start_s = body.timeline_start_s if body.timeline_start_s is not None else current["timeline_start_s"]
        src_start_s = body.src_start_s if body.src_start_s is not None else current["src_start_s"]
        src_end_s = body.src_end_s if body.src_end_s is not None else current["src_end_s"]
        tempo_ratio = body.tempo_ratio if body.tempo_ratio is not None else current["tempo_ratio"]
        pitch_shift_semi = body.pitch_shift_semi if body.pitch_shift_semi is not None else current["pitch_shift_semi"]
        variant_tag = body.variant_tag if body.variant_tag is not None else current["variant_tag"]
        if src_end_s <= src_start_s:
            raise HTTPException(status_code=400, detail="src_end_s must be > src_start_s")
        conn.execute(
            """
            UPDATE browser_daw_clips
            SET lane_idx = ?, timeline_start_s = ?, src_start_s = ?, src_end_s = ?,
                tempo_ratio = ?, pitch_shift_semi = ?, variant_tag = ?, updated_at = CURRENT_TIMESTAMP
            WHERE clip_id = ?
            """,
            (lane_idx, timeline_start_s, src_start_s, src_end_s, tempo_ratio, pitch_shift_semi, variant_tag, clip_id),
        )
        conn.commit()
        row = conn.execute(
            """
            SELECT clip_id, project_id, track_id, lane_idx, timeline_start_s,
                   src_start_s, src_end_s, tempo_ratio, pitch_shift_semi, variant_tag
            FROM browser_daw_clips WHERE clip_id = ?
            """,
            (clip_id,),
        ).fetchone()
    return {"item": dict(row)}


@app.post("/clips/{clip_id}/split")
def split_clip(clip_id: int, split_at_src_s: float = Query(..., gt=0.0)) -> dict:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT clip_id, project_id, track_id, lane_idx, timeline_start_s,
                   src_start_s, src_end_s, tempo_ratio, pitch_shift_semi, variant_tag
            FROM browser_daw_clips WHERE clip_id = ?
            """,
            (clip_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Clip not found: {clip_id}")
        if split_at_src_s <= float(row["src_start_s"]) or split_at_src_s >= float(row["src_end_s"]):
            raise HTTPException(status_code=400, detail="split_at_src_s must be inside clip range")

        old_start = float(row["src_start_s"])
        old_end = float(row["src_end_s"])
        ratio = float(row["tempo_ratio"])
        left_duration_timeline = (split_at_src_s - old_start) / ratio
        right_timeline_start = float(row["timeline_start_s"]) + left_duration_timeline

        conn.execute(
            """
            UPDATE browser_daw_clips
            SET src_end_s = ?, updated_at = CURRENT_TIMESTAMP
            WHERE clip_id = ?
            """,
            (split_at_src_s, clip_id),
        )
        cur = conn.execute(
            """
            INSERT INTO browser_daw_clips(
                project_id, track_id, lane_idx, timeline_start_s, src_start_s, src_end_s, tempo_ratio, pitch_shift_semi, variant_tag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(row["project_id"]),
                row["track_id"],
                int(row["lane_idx"]),
                right_timeline_start,
                split_at_src_s,
                old_end,
                ratio,
                int(row["pitch_shift_semi"]),
                row["variant_tag"],
            ),
        )
        conn.commit()
        left = conn.execute(
            """
            SELECT clip_id, project_id, track_id, lane_idx, timeline_start_s,
                   src_start_s, src_end_s, tempo_ratio, pitch_shift_semi, variant_tag
            FROM browser_daw_clips WHERE clip_id = ?
            """,
            (clip_id,),
        ).fetchone()
        right = conn.execute(
            """
            SELECT clip_id, project_id, track_id, lane_idx, timeline_start_s,
                   src_start_s, src_end_s, tempo_ratio, pitch_shift_semi, variant_tag
            FROM browser_daw_clips WHERE clip_id = ?
            """,
            (int(cur.lastrowid),),
        ).fetchone()
    return {"left": dict(left), "right": dict(right)}


@app.delete("/clips/{clip_id}")
def delete_clip(clip_id: int) -> dict:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM browser_daw_clips WHERE clip_id = ?", (clip_id,))
        conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail=f"Clip not found: {clip_id}")
    return {"ok": True}
