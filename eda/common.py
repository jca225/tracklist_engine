import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import sqlite3
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
from collections.abc import Iterator
from typing import Iterator
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from datetime import time

from typing import Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


def get_dj_metadata_df(json_dir: Path) -> pd.DataFrame:
    """Reads all JSON files in the specified directory, extract DJ metadata, and returns it as a DataFrame.
       Each JSON file is expected to contain a list of DJ metadata dicts, each item in the list becomes a row in the set.
       A new column is made to indicate which artist the metadata is for, derived from the filename (without extension)."""
    pattern = os.path.join(json_dir, "*.json")

    all_items = []

    for path in glob.glob(pattern):
        artist_name = os.path.splitext(os.path.basename(path))[0]

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Data is a list of dicts
        if isinstance(data, dict):
            data = [data]  # tolerate single object files
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON array (list) or object (dict), got {type(data)}")

        for item in data:
            if not isinstance(item, dict):
                continue  # or: item = {"value": item}
            all_items.append({**item, "artist_name": artist_name})

    return pd.DataFrame(all_items)


def get_dj_sets_with_at_least_one_dj_set_media_link(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get DJ Sets that have at least one DJ Set Media Link."""
    sets_with_links_query = """
    SELECT DISTINCT s.*
    FROM dj_sets s
    INNER JOIN dj_set_media_links l
        ON s.set_id = l.set_id;
    """
    return pd.read_sql_query(sets_with_links_query, conn)


def get_dj_set_rows_with_at_least_one_dj_set_media_link(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get DJ Set Rows belonging to DJ Sets with at least one DJ Set Media Link."""
    set_rows_with_links_query = """
    SELECT r.*
    FROM dj_set_rows r
    WHERE r.set_id IN (
        SELECT DISTINCT set_id
        FROM dj_set_media_links
    );
    """
    return pd.read_sql_query(set_rows_with_links_query, conn)

def get_dj_set_row_media_links_with_dj_set_media_links(conn):
    """Select DJ Set Row Media Links for DJ Sets with at least one DJ Set Media Link"""
    set_media_links_with_dj_links_query = """
    SELECT m.*
    FROM dj_set_track_media_links m
    WHERE m.set_id IN (
        SELECT DISTINCT set_id
        FROM dj_set_media_links
    );
    """
    return pd.read_sql_query(set_media_links_with_dj_links_query, conn)


def get_dj_sets_with_media_links_without_suggestions(conn: sqlite3.Connection) -> pd.DataFrame:
    """Select DJ Sets that have media links but no suggestion rows."""
    sets_query = """
    SELECT s.*
    FROM dj_sets s
    WHERE s.set_id IN (
        SELECT DISTINCT set_id 
        FROM dj_set_media_links
        WHERE set_id NOT IN (
            SELECT DISTINCT set_id 
            FROM dj_set_rows 
            WHERE element_id LIKE 'sug%'
        )
    );
    """
    return pd.read_sql_query(sets_query, conn)


def get_dj_sets_without_suggestions(conn: sqlite3.Connection) -> pd.DataFrame:
    """Select DJ Sets that have media links but no suggestion rows."""
    
    query = """
    SELECT s.*
    FROM dj_sets s
    JOIN (
        -- 1. Get all sets that DO have media links
        SELECT set_id 
        FROM dj_set_media_links
        
        EXCEPT
        
        -- 2. Subtract all sets that DO have suggestion rows
        SELECT set_id 
        FROM dj_set_rows 
        WHERE element_id LIKE 'sug%'
    ) valid_sets ON s.set_id = valid_sets.set_id;
    """
    
    return pd.read_sql_query(query, conn)

def get_dj_set_rows_without_suggestions(conn: sqlite3.Connection) -> pd.DataFrame:
    """Select DJ Set Rows belonging to DJ Sets that have media links but no suggestion rows."""
    query = """
    SELECT r.*
    FROM dj_set_rows r
    WHERE r.set_id IN (
        SELECT DISTINCT set_id 
        FROM dj_set_media_links
        WHERE set_id NOT IN (
            SELECT DISTINCT set_id 
            FROM dj_set_rows 
            WHERE element_id LIKE 'sug%'
        )
    );
    """
    return pd.read_sql_query(query, conn)



def get_dj_set_row_media_links_without_suggestions(conn: sqlite3.Connection) -> pd.DataFrame:
    """Select DJ Set Row Media Links for DJ Sets that have media links but no suggestion rows."""
    query = """
    SELECT m.*
    FROM dj_set_track_media_links m
    WHERE m.set_id IN (
        SELECT DISTINCT set_id
        FROM dj_set_media_links
        WHERE set_id NOT IN (
            SELECT DISTINCT set_id
            FROM dj_set_rows
            WHERE element_id LIKE 'sug%'
        )
    );
    """
    return pd.read_sql_query(query, conn)


# ---------- generation-oriented EDA helpers ----------------------------------
# Below: helpers used by `set_structure.ipynb`. Goal is to characterize the
# corpus the way a DJ-set generator would consume it — set length, track
# reuse, missingness, mashup density, style distribution.

_PLAY_TIME_HMS = re.compile(r"(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?", re.I)


def parse_play_time_minutes(s: object) -> float:
    """Parse 1001tracklists `play_time` strings ('1h 28m', '59m') into minutes.
    Returns NaN for missing/unparseable values so pandas plotting handles them."""
    if not isinstance(s, str) or not s.strip():
        return float("nan")
    m = _PLAY_TIME_HMS.fullmatch(s.strip())
    if not m or not (m.group(1) or m.group(2)):
        return float("nan")
    h = int(m.group(1) or 0)
    mm = int(m.group(2) or 0)
    return float(h * 60 + mm)


def get_dj_sets_full(conn: sqlite3.Connection) -> pd.DataFrame:
    """All sets with computed `play_minutes` and a string year column."""
    df = pd.read_sql_query(
        "SELECT set_id, title, date_played, artists, creator_name, "
        "views, likes, ided_tracks, total_tracks, play_time, styles "
        "FROM dj_sets",
        conn,
    )
    df["play_minutes"] = df["play_time"].map(parse_play_time_minutes)
    df["year"] = pd.to_datetime(df["date_played"], errors="coerce").dt.year
    df["ided_ratio"] = df["ided_tracks"] / df["total_tracks"].where(df["total_tracks"] > 0)
    return df


def get_set_row_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Per-set row inventory split by element_id family.

    Columns: set_id, n_rows, n_track_rows, n_suggestion_rows, n_other_rows.
    `track_rows` = element_id starts with 'tlp_' (real track entries).
    `suggestion_rows` = 'sug_' (1001tracklists user-suggested IDs).
    `other_rows` = playerWidget/tl_save/None (UI scaffolding)."""
    return pd.read_sql_query(
        """
        SELECT set_id,
               COUNT(*)                                                 AS n_rows,
               SUM(CASE WHEN element_id LIKE 'tlp_%'  THEN 1 ELSE 0 END) AS n_track_rows,
               SUM(CASE WHEN element_id LIKE 'sug_%'  THEN 1 ELSE 0 END) AS n_suggestion_rows,
               SUM(CASE WHEN element_id IS NULL OR element_id NOT LIKE 'tlp_%'
                              AND element_id NOT LIKE 'sug_%' THEN 1 ELSE 0 END) AS n_other_rows
        FROM dj_set_rows
        GROUP BY set_id
        """,
        conn,
    )


def get_set_media_link_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Per-set media-link coverage: total links + per-platform booleans."""
    df = pd.read_sql_query(
        """
        SELECT set_id, platform, COUNT(*) AS n
        FROM dj_set_media_links
        GROUP BY set_id, platform
        """,
        conn,
    )
    pivot = df.pivot_table(index="set_id", columns="platform", values="n",
                           aggfunc="sum", fill_value=0)
    pivot["n_links_total"] = pivot.sum(axis=1)
    return pivot.reset_index()


def get_set_track_link_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Per-set per-track-row media link counts and distinct-platform breadth."""
    return pd.read_sql_query(
        """
        SELECT set_id,
               COUNT(*)                                                AS n_track_links,
               COUNT(DISTINCT track_id)                                AS n_distinct_tracks,
               COUNT(DISTINCT platform)                                AS n_platforms,
               SUM(CASE WHEN platform='youtube'    THEN 1 ELSE 0 END)  AS n_youtube,
               SUM(CASE WHEN platform='spotify'    THEN 1 ELSE 0 END)  AS n_spotify,
               SUM(CASE WHEN platform='soundcloud' THEN 1 ELSE 0 END)  AS n_soundcloud,
               SUM(CASE WHEN platform='apple'      THEN 1 ELSE 0 END)  AS n_apple
        FROM dj_set_track_media_links
        WHERE track_id IS NOT NULL AND track_id != ''
        GROUP BY set_id
        """,
        conn,
    )


def get_set_failure_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Per-set scrape-failure counts split by stage (page_load/scrape/ajax/db)."""
    df = pd.read_sql_query(
        "SELECT set_id, stage, COUNT(*) AS n FROM scrape_failures GROUP BY set_id, stage",
        conn,
    )
    return df.pivot_table(index="set_id", columns="stage", values="n",
                          aggfunc="sum", fill_value=0).reset_index()


def explode_styles(sets_df: pd.DataFrame) -> pd.DataFrame:
    """Long-format (set_id, style) frame from comma-separated `styles`."""
    s = (sets_df[["set_id", "styles"]]
         .dropna(subset=["styles"])
         .assign(style=lambda d: d["styles"].str.split(r"\s*,\s*"))
         .explode("style"))
    s["style"] = s["style"].str.strip()
    return s[s["style"] != ""].reset_index(drop=True)


def get_track_play_frequency(conn: sqlite3.Connection,
                             min_plays: int = 1) -> pd.DataFrame:
    """How many distinct sets each canonical track_id appears in.

    Joins via `dj_set_track_media_links.track_id`. Used for Pareto / catalog
    reuse plots. Returns columns: track_id, n_sets, n_links."""
    return pd.read_sql_query(
        """
        SELECT track_id,
               COUNT(DISTINCT set_id) AS n_sets,
               COUNT(*)               AS n_links
        FROM dj_set_track_media_links
        WHERE track_id IS NOT NULL AND track_id != ''
        GROUP BY track_id
        HAVING n_sets >= ?
        ORDER BY n_sets DESC
        """,
        conn, params=(min_plays,),
    )


def cue_section_durations_for_set(conn: sqlite3.Connection,
                                  set_id: str) -> np.ndarray:
    """Gap (seconds) between consecutive cue_seconds anchor points in a set.

    Computed off the tokenized track rows so suggestions/concurrent sub-rows
    are filtered out. Useful for sizing the typical mix-section length —
    a generator needs to know how long each played segment lasts."""
    from eda.big_bootie import tokenize_rows  # local import to avoid cycle
    rows = pd.read_sql_query(
        "SELECT * FROM dj_set_rows WHERE set_id = ? ORDER BY row_index",
        conn, params=(set_id,),
    )
    tok = tokenize_rows(rows)
    cues = (tok.loc[tok["row_kind"] == "track", "cue_seconds_section"]
              .dropna().drop_duplicates().sort_values().to_numpy())
    if cues.size < 2:
        return np.array([])
    return np.diff(cues)

