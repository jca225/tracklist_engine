"""Build a bar-synchronous mix MERT artifact from audio + measure boundaries."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from analysis.adapters import audio_io, mert_adapter
from eda.alignment.artifacts import save_mix_mert_artifact
from eda.alignment.mert_vectors import probe_vector


def _load_measure_times(path: Path) -> np.ndarray:
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return np.asarray(raw, dtype=np.float64)
    if isinstance(raw, dict) and "measure_times" in raw:
        return np.asarray(raw["measure_times"], dtype=np.float64)
    raise ValueError(f"{path}: expected JSON list or {{measure_times: [...]}}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 0: mix audio → per-bar MERT .npz")
    p.add_argument("--set-id", required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument(
        "--measure-times-json",
        type=Path,
        required=True,
        help="JSON array of measure boundary times in seconds (from beat_this / set_analysis)",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--layer", type=int, default=mert_adapter.MERT_DEFAULT_LAYER)
    args = p.parse_args(argv)

    if not args.audio.is_file():
        print(f"audio not found: {args.audio}", file=sys.stderr)
        return 1

    measure_times = _load_measure_times(args.measure_times_json)
    if len(measure_times) < 2:
        print("need >= 2 measure boundaries", file=sys.stderr)
        return 1

    wf_r = audio_io.load_mono(args.audio, target_sr=mert_adapter.MERT_SR)
    if not wf_r.is_ok():
        print(f"load_mono failed: {wf_r.error.detail}", file=sys.stderr)
        return 1

    h_r = mert_adapter.load()
    if not h_r.is_ok():
        print(f"MERT load failed: {h_r.error.detail}", file=sys.stderr)
        return 1
    h = h_r.value

    emb_r = mert_adapter.embed_track_per_measure(
        h,
        wf_r.value.samples,
        track_audio_id=0,
        measure_times=tuple(float(t) for t in measure_times),
    )
    if not emb_r.is_ok():
        print(f"embed failed: {emb_r.error.detail}", file=sys.stderr)
        return 1

    measures = emb_r.value
    bar_start = np.array([m.start_s for m in measures], dtype=np.float64)
    bar_end = np.array([m.end_s for m in measures], dtype=np.float64)
    vecs = np.stack(
        [probe_vector(m.embedding_bytes, m.dim, layer=args.layer) for m in measures],
        axis=0,
    )

    save_mix_mert_artifact(
        args.out,
        set_id=args.set_id,
        bar_start_s=bar_start,
        bar_end_s=bar_end,
        mert=vecs,
        mert_layer=args.layer,
        mert_model=h.version,
    )
    print(f"wrote {args.out} ({vecs.shape[0]} bars, dim={vecs.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
