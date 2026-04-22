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

