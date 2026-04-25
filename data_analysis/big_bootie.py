"""Big Bootie data access + tokenization helpers.

Pilot dataset: Two Friends' Big Bootie Mix series (30 sets, ~5,300 rows).
This module loads rows, runs the HTML tokenizer, and flattens the results
into a tidy DataFrame suitable for EDA, audio-pipeline joining, and the
`set_section_alignment` work downstream.
"""
from __future__ import annotations

import re
import sqlite3
import sys
from collections import OrderedDict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

# `row_tokens/` lives under `web_crawler/` (it parses scraped 1001tracklists row
# HTML). Named `row_tokens` rather than `tokenizers` to avoid colliding with the
# HuggingFace `tokenizers` package that may be on the Python path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from web_crawler.row_tokens import classify_row, tokenize_row


BIG_BOOTIE_TITLE_LIKE = "%Big Bootie%"


# ---------- data access ------------------------------------------------------

def load_big_bootie_sets(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        f"""
        SELECT set_id, set_url, title, date_played, views, ided_tracks,
               total_tracks, likes, play_time, styles
        FROM dj_sets
        WHERE title LIKE ?
        ORDER BY date_played
        """,
        conn, params=(BIG_BOOTIE_TITLE_LIKE,),
    )
    # Use float (not pd.Int64) so matplotlib handles missing values natively —
    # Int64 NA propagates as pandas.NA which plotting backends can't coerce.
    df["volume"] = pd.to_numeric(
        df["title"].str.extract(r"(?i)(?:Vol\.?|Volume|Episode)\s*(\d+)", expand=False),
        errors="coerce",
    )
    return df


def load_big_bootie_rows(conn: sqlite3.Connection) -> pd.DataFrame:
    """All raw rows for every Big Bootie set, ordered by (set, row_index)."""
    return pd.read_sql_query(
        f"""
        SELECT r.*
        FROM dj_set_rows r
        JOIN dj_sets s USING(set_id)
        WHERE s.title LIKE ?
        ORDER BY r.set_id, r.row_index
        """,
        conn, params=(BIG_BOOTIE_TITLE_LIKE,),
    )


def load_big_bootie_track_media_links(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        f"""
        SELECT m.*
        FROM dj_set_track_media_links m
        JOIN dj_sets s USING(set_id)
        WHERE s.title LIKE ?
        """,
        conn, params=(BIG_BOOTIE_TITLE_LIKE,),
    )


# ---------- tokenization over a row batch ------------------------------------

_TRACK_COLS = (
    "row_dom_id", "data_id", "data_trno", "is_ided", "is_concurrent",
    "is_remixish", "version_tag", "track_number_raw",
    "cue_seconds", "cue_timecode", "cue_time_seconds",
    "track_key", "track_page_href",
    "title", "full_name", "genre",
    "duration_iso", "duration_seconds",
    "plays", "media_track_numeric_id", "is_remix_row",
    "spotify_cta_text", "spotify_cta_count",
    "artwork_url",
)


def _extract_artists(tok) -> str | None:
    v = getattr(tok, "artists", None)
    if not v:
        return None
    return "|".join(v) if isinstance(v, list) else str(v)


def _extract_labels(tok) -> str | None:
    v = getattr(tok, "publisher_labels", None)
    if not v:
        return None
    return "|".join(getattr(lbl, "name", str(lbl)) for lbl in v)


def _extract_media_flags(tok) -> dict[str, bool]:
    flags = getattr(tok, "media_flags", None)
    if flags is None:
        return {"yt": False, "sc": False, "sp": False, "ap": False, "aff": False}
    return {
        "yt": bool(getattr(flags, "youtube", False)),
        "sc": bool(getattr(flags, "soundcloud", False)),
        "sp": bool(getattr(flags, "spotify", False)),
        "ap": bool(getattr(flags, "apple", False)),
        "aff": bool(getattr(flags, "affiliate", False)),
    }


def _resolve_cue_sections(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `cue_seconds_section` column with the section-anchor cue time.

    Tracklist HTML only populates `cue_seconds` on the first row of a
    mashup layer; concurrent sub-rows (`w/`) leave it at 0 or None. To count
    audio sections correctly we ffill the last non-zero cue_seconds within
    each set, ordered by row_index.
    """
    df = df.sort_values(["set_id", "row_index"]).reset_index(drop=True)
    is_track = df["row_kind"] == "track"
    anchor = df["cue_seconds"].where(is_track & (df["cue_seconds"] > 0))
    df["cue_seconds_section"] = anchor.groupby(df["set_id"]).ffill()
    return df


def tokenize_rows(rows_df: pd.DataFrame) -> pd.DataFrame:
    """Run the HTML tokenizer on every row; return a flat DataFrame.

    Always returns one row per input with columns:
      - set_id, row_index
      - row_kind ('track' | 'suggestion' | 'text' | 'player_widget' | 'save_footer' | 'unknown')
      - token_type (dataclass/pydantic class name or 'None')
      - track-row-specific columns (None for non-track rows)
      - has_yt/has_sc/has_sp/has_ap flags (from media_flags on track rows)
    """
    out_rows: list[dict] = []
    for row in rows_df.itertuples(index=False):
        raw = row.raw_html or ""
        kind = classify_row(raw)
        tok = tokenize_row(raw) if kind in {"track", "suggestion", "text"} else None
        rec: dict[str, object] = {
            "set_id": row.set_id,
            "row_index": int(row.row_index),
            "row_kind": kind,
            "token_type": type(tok).__name__ if tok is not None else "None",
        }
        if kind == "track" and tok is not None:
            for c in _TRACK_COLS:
                rec[c] = getattr(tok, c, None)
            rec["artists"] = _extract_artists(tok)
            rec["labels"] = _extract_labels(tok)
            rec.update({f"has_{k}": v for k, v in _extract_media_flags(tok).items()})
        else:
            for c in _TRACK_COLS:
                rec[c] = None
            rec["artists"] = None
            rec["labels"] = None
            for k in ("yt", "sc", "sp", "ap", "aff"):
                rec[f"has_{k}"] = None
        out_rows.append(rec)
    return _resolve_cue_sections(pd.DataFrame(out_rows))


# ---------- convenience ------------------------------------------------------

def attach_volume(sets_df: pd.DataFrame, *dfs: pd.DataFrame) -> list[pd.DataFrame]:
    """Add a `volume` column to each DF by joining on set_id."""
    lookup = sets_df.set_index("set_id")["volume"]
    return [d.assign(volume=d["set_id"].map(lookup)) for d in dfs]


def group_by_volume(rows_df: pd.DataFrame, sets_df: pd.DataFrame) -> "OrderedDict[int, pd.DataFrame]":
    """{volume_int: df_of_rows_sorted_by_row_index} for volumes with known vol number."""
    set_to_vol = dict(zip(sets_df["set_id"], sets_df["volume"]))
    grouped = {
        sid: g.sort_values("row_index").reset_index(drop=True)
        for sid, g in rows_df.groupby("set_id")
    }
    out = OrderedDict()
    for sid, g in grouped.items():
        v = set_to_vol.get(sid)
        if pd.isna(v):
            continue
        out[int(v)] = g
    return OrderedDict(sorted(out.items()))


CUE_BLOCK_SECONDS = re.compile(r"cueValuesEntry\.seconds\s*=\s*(\d+);")
CUE_BLOCK_IDS = re.compile(r"cueValuesEntry\.ids\[\d+\]\s*=\s*'([^']+)';")


def extract_cue_points_from_html(html: str) -> list[dict]:
    """Parse the cueValueData JS array 1001tracklists emits at page load time.

    Returns [{time_seconds, html_ids}, ...]. These map the audio playhead
    (a DJ-set-wide time axis) to which track's DOM element is active.
    """
    out: list[dict] = []
    blocks = html.split("cueValuesEntry = {};")
    for block in blocks[1:]:
        m = CUE_BLOCK_SECONDS.search(block)
        if not m:
            continue
        out.append({
            "time_seconds": int(m.group(1)),
            "html_ids": CUE_BLOCK_IDS.findall(block),
        })
    return out
