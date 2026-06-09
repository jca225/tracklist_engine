"""Mix structure probe — dual-stream information-dynamics + boundary scoring."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.result import Ok
from eda.alignment.artifacts import MixMertArtifact, load_mix_mert_artifact
from eda.alignment.boundaries import (
    gt_section_starts_s,
    score_boundaries,
    seconds_to_bar_indices,
)
from eda.alignment.chroma_tokens import chroma_token_labels
from eda.alignment.plot_trace import save_trace_plot
from eda.alignment.stream_probe import probe_token_stream
from eda.alignment.tokenize import fit_vq_kmeans
from labeling.ground_truth.schema import GroundTruthSet, load


def _sanitize_mert(mert: np.ndarray) -> np.ndarray:
    out = mert.astype(np.float32, copy=True)
    bad = ~np.isfinite(out).all(axis=1)
    if np.any(bad):
        out[bad] = 0.0
    return out


def _synthetic_artifact(
    *,
    n_sections: int = 8,
    bars_per_section: int = 32,
    dim: int = 64,
    seed: int = 0,
) -> MixMertArtifact:
    rng = np.random.default_rng(seed)
    n_bars = n_sections * bars_per_section
    bar_dur = 2.0
    bar_start = np.arange(n_bars, dtype=np.float64) * bar_dur
    bar_end = bar_start + bar_dur
    mert = np.zeros((n_bars, dim), dtype=np.float32)
    for s in range(n_sections):
        lo = s * bars_per_section
        hi = lo + bars_per_section
        center = rng.normal(size=dim).astype(np.float32)
        center /= np.linalg.norm(center) + 1e-8
        mert[lo:hi] = center + rng.normal(scale=0.05, size=(bars_per_section, dim)).astype(
            np.float32
        )
    return MixMertArtifact(
        set_id="synthetic",
        bar_start_s=bar_start,
        bar_end_s=bar_end,
        mert=mert,
        mert_layer=6,
        mert_model="synthetic",
    )


def run_probe(
    artifact: MixMertArtifact,
    gt: GroundTruthSet | None,
    *,
    audio_path: Path | None = None,
    n_tokens: int = 24,
    n_chroma_tokens: int = 16,
    peak_percentile: float = 75.0,
    peak_min_distance: int = 4,
    local_window: int = 32,
    local_z: float = 1.5,
    tolerance_bars: int = 2,
    chroma: bool = True,
) -> tuple[dict, dict[str, object]]:
    mert = _sanitize_mert(artifact.mert)
    gt_bars: tuple[int, ...] | None = None
    if gt is not None:
        gt_starts = gt_section_starts_s(gt)
        gt_bars = seconds_to_bar_indices(gt_starts, artifact.bar_start_s)

    _, mert_labels = fit_vq_kmeans(mert, n_tokens)
    mert_stream, mert_trace = probe_token_stream(
        artifact,
        mert_labels,
        gt_bars,
        stream_name="mert_vq",
        n_tokens=n_tokens,
        peak_percentile=peak_percentile,
        peak_min_distance=peak_min_distance,
        local_window=local_window,
        local_z=local_z,
        tolerance_bars=tolerance_bars,
    )

    streams: dict[str, dict] = {"mert_vq": mert_stream}
    traces: dict[str, object] = {"mert_vq": mert_trace}

    if chroma and audio_path is not None:
        chroma_labels = chroma_token_labels(audio_path, artifact, n_chroma_tokens)
        chroma_stream, chroma_trace = probe_token_stream(
            artifact,
            chroma_labels,
            gt_bars,
            stream_name="chroma_vq",
            n_tokens=n_chroma_tokens,
            peak_percentile=peak_percentile,
            peak_min_distance=peak_min_distance,
            local_window=local_window,
            local_z=local_z,
            tolerance_bars=tolerance_bars,
        )
        streams["chroma_vq"] = chroma_stream
        traces["chroma_vq"] = chroma_trace

        if gt_bars is not None:
            union_local = tuple(sorted(set(mert_stream["peaks"]["mir_local"]) | set(chroma_stream["peaks"]["mir_local"])))
            streams["combined_mir_local"] = {
                "stream": "combined_mir_local",
                "peaks": {"mir_local": list(union_local)},
                "scores": {
                    "mir_local": asdict(
                        score_boundaries(union_local, gt_bars, tolerance_bars=tolerance_bars)
                    ),
                },
            }

    result: dict = {
        "set_id": artifact.set_id,
        "n_bars": artifact.n_bars,
        "n_tokens_mert": n_tokens,
        "n_tokens_chroma": n_chroma_tokens if chroma and audio_path else None,
        "streams": streams,
    }
    if gt_bars is not None:
        result["n_gt_section_starts"] = len(gt_bars)
        result["gt_boundary_bars"] = list(gt_bars)

    return result, traces


def _persist_aux(result: dict, aux_db: Path) -> None:
    aux_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(aux_db)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study TEXT NOT NULL,
            key TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO analysis_results (study, key, payload_json, created_at) VALUES (?, ?, ?, ?)",
        (
            "mix_structure_probe_v2",
            result["set_id"],
            json.dumps(result),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Mix structure analysis probe (dual-stream)")
    p.add_argument("--artifact", type=Path, help="Mix MERT .npz")
    p.add_argument("--audio", type=Path, help="Mix audio for chroma stream")
    p.add_argument("--gt", type=Path, help="Ground-truth YAML")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--no-chroma", action="store_true")
    p.add_argument("--n-tokens", type=int, default=24)
    p.add_argument("--n-chroma-tokens", type=int, default=16)
    p.add_argument("--tolerance-bars", type=int, default=2)
    p.add_argument("--peak-percentile", type=float, default=75.0)
    p.add_argument("--peak-min-distance", type=int, default=4)
    p.add_argument("--local-window", type=int, default=32)
    p.add_argument("--local-z", type=float, default=1.5)
    p.add_argument("--out", type=Path)
    p.add_argument("--plot", type=Path, help="Plot MERT stream only")
    p.add_argument("--persist", action="store_true")
    args = p.parse_args(argv)

    if args.synthetic:
        artifact = _synthetic_artifact()
        gt = None
        audio = None
    elif args.artifact is not None:
        artifact = load_mix_mert_artifact(args.artifact)
        gt = None
        audio = args.audio
        if args.gt is not None:
            match load(args.gt):
                case Ok(g):
                    gt = g
                case err:
                    print(f"GT load failed: {err.error.detail}", file=sys.stderr)
                    return 1
    else:
        p.error("provide --artifact or --synthetic")

    result, traces = run_probe(
        artifact,
        gt,
        audio_path=audio,
        n_tokens=args.n_tokens,
        n_chroma_tokens=args.n_chroma_tokens,
        peak_percentile=args.peak_percentile,
        peak_min_distance=args.peak_min_distance,
        local_window=args.local_window,
        local_z=args.local_z,
        tolerance_bars=args.tolerance_bars,
        chroma=not args.no_chroma and audio is not None,
    )

    print(json.dumps(result, indent=2))
    if args.plot is not None and "mert_vq" in traces:
        gt_bars = tuple(result.get("gt_boundary_bars") or ())
        try:
            save_trace_plot(args.plot, artifact, traces["mert_vq"], gt_boundary_bars=gt_bars or None)
        except ImportError as e:
            print(f"plot skipped: {e}", file=sys.stderr)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    if args.persist:
        _persist_aux(result, Path("data/analysis/aux.db"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
