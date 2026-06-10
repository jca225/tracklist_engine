#!/usr/bin/env python3
"""Export mix + ref MERT probe vectors for one set to a compressed .npz."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

DB = "/mnt/storage/data/db/music_database.db"


def probe(blob: bytes, dim: int, layer: int) -> np.ndarray:
    n_layers = len(blob) // 2 // dim
    stack = np.frombuffer(blob, dtype=np.float16).reshape(n_layers, dim)
    return stack[layer].astype(np.float32)


def main() -> int:
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} SET_ID OUT.npz LAYER", file=sys.stderr)
        return 2
    set_id, out_s, layer_s = sys.argv[1], sys.argv[2], sys.argv[3]
    layer = int(layer_s)
    out = Path(out_s)

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT set_audio_id FROM set_audio WHERE set_id=? "
        "ORDER BY is_reference DESC, set_audio_id LIMIT 1",
        (set_id,),
    )
    row = cur.fetchone()
    if not row:
        print(f"no set_audio for {set_id}", file=sys.stderr)
        return 1
    set_audio_id = int(row[0])

    cur.execute(
        "SELECT measure_idx, start_s, end_s, dim, embedding "
        "FROM set_mert_measures WHERE set_audio_id=? ORDER BY measure_idx",
        (set_audio_id,),
    )
    mix_rows = cur.fetchall()
    mix_start = np.array([r[1] for r in mix_rows], dtype=np.float64)
    mix_end = np.array([r[2] for r in mix_rows], dtype=np.float64)
    mix_vec = np.stack([probe(r[4], r[3], layer) for r in mix_rows], axis=0)

    # One audio row per recording, preferring is_reference but falling back to
    # any row that has MERT — requiring is_reference=1 dropped 57/135 BB12 GT
    # recordings whose embeddings already existed. Keyed by recording_id (the
    # GT/pool key), not legacy track_id.
    cur.execute(
        """
        SELECT pick.recording_id, tmm.measure_idx, tmm.start_s, tmm.end_s, tmm.dim, tmm.embedding
        FROM (
            SELECT DISTINCT recording_id FROM set_track_slots WHERE set_id = ?
        ) sts
        JOIN (
            SELECT ta.recording_id, ta.track_audio_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY ta.recording_id
                       ORDER BY ta.is_reference DESC, ta.track_audio_id
                   ) AS rn
            FROM track_audio ta
            JOIN (SELECT DISTINCT track_audio_id FROM track_mert_measures) m
              ON m.track_audio_id = ta.track_audio_id
        ) pick ON pick.recording_id = sts.recording_id AND pick.rn = 1
        JOIN track_mert_measures tmm ON tmm.track_audio_id = pick.track_audio_id
        ORDER BY pick.recording_id, tmm.measure_idx
        """,
        (set_id,),
    )
    ref_rows = cur.fetchall()
    by_tid: dict[str, list[tuple]] = {}
    for tid, _idx, s, e, dim, blob in ref_rows:
        by_tid.setdefault(tid, []).append((s, e, dim, blob))

    ref_ids = sorted(by_tid)
    ref_payload: dict[str, np.ndarray] = {}
    for tid in ref_ids:
        rows = by_tid[tid]
        ref_payload[f"ref_{tid}_start"] = np.array([r[0] for r in rows], dtype=np.float64)
        ref_payload[f"ref_{tid}_end"] = np.array([r[1] for r in rows], dtype=np.float64)
        ref_payload[f"ref_{tid}_vec"] = np.stack([probe(r[3], r[2], layer) for r in rows], axis=0)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        set_audio_id=np.array(set_audio_id),
        mix_start=mix_start,
        mix_end=mix_end,
        mix_vec=mix_vec,
        ref_ids=np.array(json.dumps(ref_ids)),
        **ref_payload,
    )
    print(f"exported mix={len(mix_rows)} refs={len(ref_ids)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
