"""Embed the recurring TAIL tracks (the personalization zone) with all-layer MERT.

The v0 ID-CF taste model beat popularity 6.4x in the tail — but only for tracks with
dense co-occurrence. MERT embeddings generalize there (and to cold-start). This
downloads each tail target from SoundCloud and embeds it all-layer (25 x 1024,
whole-track mean per layer — free since the forward pass already computes every
layer; float32 accumulation avoids the float16 overflow).

  venvs/audio/bin/python -m personalization.embed_tail --limit 3     # smoke
  venvs/audio/bin/python -m personalization.embed_tail --min-likers 50
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np

from analysis.adapters import audio_io, mert_adapter
from personalization.prior_mert import download_track, sc_track_url

DB = Path("data/taste/taste_warehouse.db")
OUT = Path("data/taste/tail_track_embeds.pkl")


def tail_targets(conn: sqlite3.Connection, min_likers: int, head: int = 500) -> list[tuple[int, str]]:
    rows = conn.execute(
        """
        WITH pop AS (SELECT track_id, COUNT(DISTINCT user_id) k FROM sc_likes GROUP BY track_id),
        ranked AS (SELECT track_id, k, ROW_NUMBER() OVER (ORDER BY k DESC) rk FROM pop)
        SELECT r.track_id,
               (SELECT raw_json FROM sc_likes WHERE track_id=r.track_id LIMIT 1) rj
        FROM ranked r WHERE r.rk > ? AND r.k >= ? ORDER BY r.k DESC
        """,
        (head, min_likers),
    ).fetchall()
    out = []
    for tid, rj in rows:
        url = sc_track_url(str(rj)) if rj else None
        if url:
            out.append((int(tid), url))
    return out


def embed_all_layers(h, samples_24k: np.ndarray) -> np.ndarray | None:
    """(n_layers, dim) float16, whole-track mean per layer. Float32 frame-sum accumulation."""
    import torch
    cs = int(mert_adapter.MERT_CHUNK_S * mert_adapter.MERT_SR)
    sums = None
    n = 0
    for i in range(0, samples_24k.size, cs):
        chunk = samples_24k[i:i + cs]
        if chunk.size < mert_adapter.MERT_SR // 10:
            continue
        inputs = h._processor(chunk, sampling_rate=mert_adapter.MERT_SR, return_tensors="pt")
        inputs = {k: v.to(h.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = h._model(**inputs, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 0).squeeze(1)        # (n_layers, T, dim)
        s = hs.float().sum(dim=1).cpu().numpy()                  # (n_layers, dim) float32
        sums = s if sums is None else sums + s
        n += hs.shape[1]
    if sums is None or n == 0:
        return None
    return (sums / n).astype(np.float16)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-likers", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    conn = sqlite3.connect(DB)
    targets = tail_targets(conn, args.min_likers)
    if args.limit:
        targets = targets[:args.limit]
    cache = pickle.loads(OUT.read_bytes()) if OUT.is_file() else {}
    todo = [(tid, url) for tid, url in targets if tid not in cache]
    print(f"tail targets: {len(targets)} | already embedded: {len(cache)} | to do: {len(todo)}")
    if not todo:
        return 0

    h = mert_adapter.load().value
    ok = fail = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        for i, (tid, url) in enumerate(todo, 1):
            p = None
            for attempt in range(3):                       # survive transient WiFi/throttle blips
                p = download_track(url, tmpd / str(tid))
                if p is not None:
                    break
                time.sleep(2 * (attempt + 1))              # backoff before retry
            time.sleep(0.4)                                # gentle pacing between tracks
            if p is None:
                fail += 1; continue
            wf = audio_io.load_mono(p, target_sr=mert_adapter.MERT_SR)
            p.unlink(missing_ok=True)
            if not wf.is_ok():
                fail += 1; continue
            vec = embed_all_layers(h, wf.value.samples)
            if vec is None:
                fail += 1; continue
            cache[tid] = vec
            ok += 1
            if ok % 25 == 0:
                OUT.write_bytes(pickle.dumps(cache))
                print(f"  {ok} embedded ({fail} failed) / {len(todo)}")
    OUT.write_bytes(pickle.dumps(cache))
    print(f"done: embedded {ok}, failed {fail}, total cached {len(cache)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
