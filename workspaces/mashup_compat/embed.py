"""Embed BB12 Roformer stems with the existing all-layer MERT-330M embedder.

For each (recording, role) the proof needs, pull the matching Roformer stem from
pi-storage (bed -> instrumental, payload -> vocals), run `embed_track_per_measure`
over the WHOLE stem (one measure = [0, duration]) to get a (n_layers, dim) =
(25, 1024) whole-song token keeping ALL layers, and cache it.

I/O at the edges (ssh resolve + rsync pull); compute in the middle (MERT). Resumable:
already-cached stems are skipped, so a crash mid-batch just re-runs the remainder.

  venvs/audio/bin/python -m workspaces.mashup_compat.embed --limit 2   # smoke test
  venvs/audio/bin/python -m workspaces.mashup_compat.embed             # full 150
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

from analysis.adapters import audio_io, mert_adapter
from workspaces.mashup_compat.pairs import Stem, extract_pairs, needed_stems

PI = "pi-storage"
PI_DB = "/mnt/storage/data/db/music_database.db"
PI_STEMS = "/mnt/storage/stems"
CACHE_DIR = Path("data/mashup_compat")
LOCAL_STEMS = CACHE_DIR / "stems"
EMB_PATH = CACHE_DIR / "bb12_stem_embeds.pkl"


def _resolve_track_audio_ids(recording_ids: list[str]) -> dict[str, int]:
    """recording_id -> reference track_audio_id (the regular audio that separation ran on)."""
    ids = ",".join(f"'{r}'" for r in sorted(set(recording_ids)))
    sql = (
        "SELECT recording_id, track_audio_id FROM track_audio "
        f"WHERE recording_id IN ({ids}) AND stem='regular' "
        "ORDER BY is_reference DESC, track_audio_id;"
    )
    out = subprocess.run(["ssh", PI, f"sqlite3 {PI_DB} \"{sql}\""],
                         capture_output=True, text=True, check=True).stdout
    m: dict[str, int] = {}
    for line in out.strip().splitlines():
        rid, taid = line.split("|")
        m.setdefault(rid, int(taid))        # first row wins = is_reference
    return m


def _pull_stem(track_audio_id: int, stem_file: str) -> Path | None:
    """rsync /mnt/storage/stems/<taid>/<vocals|instrumental>.* -> local; return local path."""
    LOCAL_STEMS.mkdir(parents=True, exist_ok=True)
    dest = LOCAL_STEMS / str(track_audio_id)
    dest.mkdir(exist_ok=True)
    remote = f"{PI}:{PI_STEMS}/{track_audio_id}/{stem_file}.*"
    subprocess.run(["rsync", "-q", "--ignore-existing", remote, f"{dest}/"],
                   check=False)
    hits = sorted(dest.glob(f"{stem_file}.*"))
    return hits[0] if hits else None


GRID_S = 1.0   # window size for per-window pooling; small enough that the float16
               # frame-sum inside the embedder can't overflow (whole-track = 1 window does)


def _decode(m) -> np.ndarray:
    n_layers = len(m.embedding_bytes) // (2 * m.dim)
    return np.frombuffer(m.embedding_bytes, dtype=np.float16).reshape(n_layers, m.dim).astype(np.float32)


def _whole_song_token(h: mert_adapter.MertHandle, stem_path: Path) -> np.ndarray | None:
    wf = audio_io.load_mono(stem_path, target_sr=mert_adapter.MERT_SR)
    if not wf.is_ok():
        print(f"  load_mono failed: {wf.error.detail}", file=sys.stderr)
        return None
    dur = wf.value.samples.size / mert_adapter.MERT_SR
    # ~1s grid → many small measures (no float16 overflow), then pool in float32.
    grid = tuple(float(x) for x in np.arange(0.0, dur, GRID_S)) + (float(dur),)
    if len(grid) < 2:
        grid = (0.0, float(dur))
    emb = mert_adapter.embed_track_per_measure(
        h, wf.value.samples, track_audio_id=0, measure_times=grid)
    if not emb.is_ok() or not emb.value:
        print(f"  embed failed: {getattr(emb, 'error', 'empty')}", file=sys.stderr)
        return None
    windows = np.stack([_decode(m) for m in emb.value], axis=0)   # (n_win, n_layers, dim) f32
    return np.mean(windows, axis=0).astype(np.float16)            # (n_layers, dim) whole-song token


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="labeling/fixtures/bb12_ground_truth.yaml")
    ap.add_argument("--limit", type=int, default=0, help="embed only first N stems (smoke test)")
    args = ap.parse_args(argv)

    stems = needed_stems(extract_pairs(args.gt))
    if args.limit:
        stems = stems[:args.limit]
    print(f"need {len(stems)} stem tokens")

    cache: dict[tuple[str, str], np.ndarray] = {}
    if EMB_PATH.is_file():
        cache = pickle.loads(EMB_PATH.read_bytes())
        print(f"resuming: {len(cache)} already cached")

    todo = [s for s in stems if (s.track_id, s.role) not in cache]
    if not todo:
        print("all cached, nothing to embed")
        return 0

    taids = _resolve_track_audio_ids([s.track_id for s in todo])
    h_r = mert_adapter.load()
    if not h_r.is_ok():
        print(f"MERT load failed: {h_r.error.detail}", file=sys.stderr)
        return 1
    h = h_r.value

    ok = miss = 0
    for i, s in enumerate(todo, 1):
        taid = taids.get(s.track_id)
        if taid is None:
            print(f"[{i}/{len(todo)}] {s.label} ({s.role}): no track_audio_id"); miss += 1; continue
        sp = _pull_stem(taid, s.stem_file)
        if sp is None:
            print(f"[{i}/{len(todo)}] {s.label} ({s.role}): no {s.stem_file} stem on pi"); miss += 1; continue
        vec = _whole_song_token(h, sp)
        if vec is None:
            miss += 1; continue
        cache[(s.track_id, s.role)] = vec
        ok += 1
        print(f"[{i}/{len(todo)}] {s.label} ({s.role}) -> {vec.shape}")
        if ok % 10 == 0:                      # periodic checkpoint
            EMB_PATH.write_bytes(pickle.dumps(cache))

    EMB_PATH.write_bytes(pickle.dumps(cache))
    print(f"done: embedded {ok}, missing {miss}, total cached {len(cache)} -> {EMB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
