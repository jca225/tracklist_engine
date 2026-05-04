"""Streamlit app for ground-truth alignment labeling (Phase 8a).

Walks measure-by-measure through a chosen set's mix audio. For each
measure window, you listen to the mix, pick which reference track is
currently playing, and pick where in that track. Optional pitch / tempo
shifts. Saves to `measure_alignment` with `confidence=1.0` and
`confidence_source='human_label'` so the BPE optimizer (Phase 8b) and
the algorithm trainer (Phase 8c) can filter ground truth from
algorithm-generated rows.

Run on the same machine that has audio files mounted/local:

    venvs/audio/bin/streamlit run audio_pipeline/labeling/app.py -- \\
        --db data/db/music_database.db \\
        --audio-root /mnt/storage

Or on pi-storage with port-forward:

    ssh -L 8501:localhost:8501 pi-storage \\
      'cd tracklist_engine && venvs/audio/bin/streamlit run audio_pipeline/labeling/app.py'

Schema requirement: `measure_alignment.confidence_source` column. Created
inline at startup if absent (idempotent ALTER).
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import streamlit as st


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path,
                   default=Path(os.environ.get("TRACKLIST_DB",
                                               "/mnt/storage/data/db/music_database.db")))
    p.add_argument("--audio-root", type=Path,
                   default=Path(os.environ.get("TRACKLIST_AUDIO_ROOT", "/mnt/storage")))
    # Streamlit owns argv[0..]; everything after `--` is ours.
    args, _ = p.parse_known_args(sys.argv[1:])
    return args


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    finally:
        conn.close()


def _ensure_schema(db_path: Path) -> None:
    """Idempotently add `confidence_source` column to measure_alignment."""
    try:
        with _connect(db_path) as conn:
            conn.execute(
                "ALTER TABLE measure_alignment ADD COLUMN confidence_source TEXT "
                "DEFAULT 'algorithm'"
            )
            conn.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e).lower():
            raise


def _list_labelable_sets(db_path: Path) -> list[dict]:
    """Sets with mix audio downloaded AND >=1 track audio file present."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT s.set_id, s.title, s.date_played, s.total_tracks,
                   sa.path AS mix_path, sa.duration_s AS mix_duration_s,
                   (SELECT COUNT(DISTINCT track_id) FROM track_audio
                    WHERE track_id IN (SELECT track_id FROM dj_set_track_media_links
                                       WHERE set_id = s.set_id)) AS tracks_downloaded
            FROM dj_sets s
            JOIN set_audio sa ON sa.set_id = s.set_id
            ORDER BY s.date_played DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def _list_set_tracks(db_path: Path, set_id: str) -> list[dict]:
    """Reference tracks for this set that have downloaded audio + metadata."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT ta.track_id, ta.path, ta.duration_s,
                   tm.title AS track_title, tm.artists_json AS artists_json,
                   tm.full_name AS full_name
            FROM track_audio ta
            JOIN dj_set_track_media_links m ON m.track_id = ta.track_id
            LEFT JOIN track_metadata tm ON tm.track_id = ta.track_id
            WHERE m.set_id = ?
            ORDER BY tm.title NULLS LAST
            """,
            (set_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _load_existing_labels(db_path: Path, set_id: str) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT set_measure_idx, ref_track_id, ref_measure_idx,
                   pitch_shift_semi, tempo_ratio, stem_mask_json,
                   confidence, confidence_source, aligned_at
            FROM measure_alignment
            WHERE set_id = ? AND confidence_source = 'human_label'
            ORDER BY set_measure_idx
            """,
            (set_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _save_label(
    db_path: Path,
    set_id: str,
    set_measure_idx: int,
    ref_track_id: str,
    ref_measure_idx: int,
    pitch_shift_semi: int,
    tempo_ratio: float,
    stem_mask: list[str],
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO measure_alignment
              (set_id, set_measure_idx, ref_track_id, ref_measure_idx,
               pitch_shift_semi, tempo_ratio, stem_mask_json,
               confidence, confidence_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1.0, 'human_label')
            ON CONFLICT(set_id, set_measure_idx, ref_track_id) DO UPDATE SET
              ref_measure_idx   = excluded.ref_measure_idx,
              pitch_shift_semi  = excluded.pitch_shift_semi,
              tempo_ratio       = excluded.tempo_ratio,
              stem_mask_json    = excluded.stem_mask_json,
              confidence        = 1.0,
              confidence_source = 'human_label',
              aligned_at        = CURRENT_TIMESTAMP
            """,
            (
                set_id, set_measure_idx, ref_track_id, ref_measure_idx,
                pitch_shift_semi, float(tempo_ratio), json.dumps(stem_mask),
            ),
        )
        conn.commit()


def _delete_label(db_path: Path, set_id: str, set_measure_idx: int, ref_track_id: str) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            DELETE FROM measure_alignment
            WHERE set_id = ? AND set_measure_idx = ? AND ref_track_id = ?
              AND confidence_source = 'human_label'
            """,
            (set_id, set_measure_idx, ref_track_id),
        )
        conn.commit()


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title="Tracklist alignment labeler", layout="wide")
    _ensure_schema(args.db)

    st.title("Phase 8a — alignment labeling")
    st.caption(
        f"DB: `{args.db}` · audio root: `{args.audio_root}`. "
        "Labels save to `measure_alignment` with `confidence_source='human_label'`."
    )

    sets = _list_labelable_sets(args.db)
    if not sets:
        st.warning(
            "No labelable sets yet. Need at least one set with `set_audio` downloaded "
            "(`audio_pipeline.main --with-mixes`) plus some track refs in `track_audio`."
        )
        return

    set_options = [
        f"{s['set_id']} · {s['title'][:60]} · {s['tracks_downloaded']}/{s['total_tracks']} refs"
        for s in sets
    ]
    chosen_idx = st.selectbox("Set to label", range(len(sets)),
                              format_func=lambda i: set_options[i], key="set_picker")
    chosen = sets[chosen_idx]
    set_id = chosen["set_id"]
    mix_path = Path(chosen["mix_path"])

    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.subheader(chosen["title"])
        if mix_path.exists():
            st.audio(str(mix_path))
        else:
            st.error(f"mix file not on this machine: `{mix_path}`")
    with col_b:
        st.metric("date", chosen["date_played"])
        st.metric("duration", f"{chosen['mix_duration_s'] or 0:.0f}s")
        st.metric("refs ready", f"{chosen['tracks_downloaded']}/{chosen['total_tracks']}")

    tracks = _list_set_tracks(args.db, set_id)
    if not tracks:
        st.warning("No downloaded reference tracks for this set yet.")
        return

    st.divider()
    st.subheader("Add / update label")
    track_options = [
        f"{t['track_id']} · {(t['full_name'] or t['track_title'] or '?')[:70]}"
        for t in tracks
    ]
    track_idx = st.selectbox("Reference track", range(len(tracks)),
                             format_func=lambda i: track_options[i], key="track_picker")
    track = tracks[track_idx]
    track_path = Path(track["path"])
    if track_path.exists():
        st.audio(str(track_path))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        set_measure_idx = st.number_input("set_measure_idx", min_value=0, value=0, step=1)
    with c2:
        ref_measure_idx = st.number_input("ref_measure_idx", min_value=0, value=0, step=1)
    with c3:
        pitch = st.number_input("pitch_shift_semi", min_value=-12, max_value=12, value=0, step=1)
    with c4:
        tempo = st.number_input("tempo_ratio", min_value=0.5, max_value=2.0, value=1.0, step=0.01,
                                format="%.2f")
    stem_mask = st.multiselect(
        "stem_mask (which stems are audible in the mix)",
        options=["full", "vocals", "drums", "bass", "other", "instrumental"],
        default=["full"],
    )

    col_save, col_del = st.columns(2)
    with col_save:
        if st.button("Save label", type="primary", use_container_width=True):
            _save_label(args.db, set_id, int(set_measure_idx), track["track_id"],
                        int(ref_measure_idx), int(pitch), float(tempo), stem_mask)
            st.success(f"saved set_measure_idx={set_measure_idx} → {track['track_id']} "
                       f"measure {ref_measure_idx}")
            st.rerun()
    with col_del:
        if st.button("Delete this row", use_container_width=True):
            _delete_label(args.db, set_id, int(set_measure_idx), track["track_id"])
            st.info("deleted")
            st.rerun()

    st.divider()
    st.subheader(f"Existing labels for `{set_id}`")
    labels = _load_existing_labels(args.db, set_id)
    if not labels:
        st.write("(none yet)")
    else:
        st.dataframe(labels, use_container_width=True)


if __name__ == "__main__":
    main()
