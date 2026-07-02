#!/usr/bin/env python3
"""Reconstruction-margin placement refiner for HOST (regular) spans.

Step-2 v1 (docs/reconstruction_supervision_plan.md): the learned model does identity
only; placement is a non-learned DP. Reconstruction was validated as a placement signal
for the host/regular channel (79% peak-at-0). This module uses it as a POST-INFER
refiner: for each regular span, slide the predicted mix window over a band and pick the
set_start that best RECONSTRUCTS the mix from the (fixed) predicted ref content —
GATED so it only overrides the pipeline when the reconstruction gain is confident.

Operates on a predicted-timeline JSON, writes a refined one. New module — does NOT touch
infer.py's decode (the parallel-aligner surface). Acappella/instrumental spans are left
untouched (reconstruction is a per-window vocal floor there — see the plan doc).

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.recon_rerank \\
        --set-id 1fsnxchk [--band-s 30 --gate 0.02]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.recon_probe import (  # noqa: E402
    MIN_DUR_S,
    SR,
    find_aligning_dir,
    load_audio,
    match_score,
    slice_s,
)

OUT_DIR = Path(__file__).resolve().parent / "out"


def rerank_timeline(
    set_id: str,
    in_path: Path,
    out_path: Path,
    *,
    band_s: float = 30.0,
    step_s: float = 1.0,
    gate: float = 0.08,
) -> dict:
    """Refine host-span set_start by reconstruction match; return a summary dict."""
    aligning = find_aligning_dir(set_id)
    if aligning is None:
        raise SystemExit(f"no aligning dir for {set_id}")
    manifest = json.loads((aligning / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)
    mix_path = manifest.get("mix_local_path") or str(aligning / "mix.m4a")
    print(f"loading mix {Path(mix_path).name} ...")
    mix = load_audio(mix_path)
    if mix is None:
        raise SystemExit("mix load failed")
    mix_dur = len(mix) / SR

    tl = json.loads(Path(in_path).read_text())
    spans = tl["spans"]
    n_host = n_moved = 0
    shifts: list[float] = []
    for s in spans:
        if (s.get("claimed_stem") or "regular") != "regular":
            continue
        try:
            ss, se = float(s["set_start_s"]), float(s["set_end_s"])
            rs = float(s["ref_start_s"])
        except (KeyError, TypeError, ValueError):
            continue
        dur = se - ss
        if dur < MIN_DUR_S:
            continue
        t = by_tid.get(str(s.get("recording_id") or ""))
        if t is None:
            continue
        ref = load_audio(t.get("local_path"))
        if ref is None:
            continue
        n_host += 1
        ref_seg = slice_s(ref, rs, dur)  # fixed predicted ref content
        base = match_score(slice_s(mix, ss, dur), ref_seg)
        if base is None:
            continue
        # slide the mix window over a band around the predicted set_start
        lo = max(0.0, ss - band_s)
        hi = min(mix_dur - dur, ss + band_s)
        best_c, best_sc = ss, base
        c = lo
        while c <= hi:
            sc = match_score(slice_s(mix, c, dur), ref_seg)
            if sc is not None and sc > best_sc:
                best_sc, best_c = sc, c
            c += step_s
        # GATE: only override when the reconstruction gain over the pipeline is confident
        if best_c != ss and (best_sc - base) > gate:
            delta = best_c - ss
            s["set_start_s"] = best_c
            s["set_end_s"] = se + delta
            for seg in s.get("ref_segments") or []:
                if "mix_start_s" in seg:
                    seg["mix_start_s"] = float(seg["mix_start_s"]) + delta
            s["recon_refined"] = True
            s["recon_gain"] = round(best_sc - base, 4)
            n_moved += 1
            shifts.append(abs(delta))

    tl["recon_rerank"] = {
        "band_s": band_s,
        "step_s": step_s,
        "gate": gate,
        "n_host": n_host,
        "n_moved": n_moved,
    }
    Path(out_path).write_text(json.dumps(tl, indent=2))
    med_shift = float(np.median(shifts)) if shifts else 0.0
    print(
        f"host spans: {n_host} | recon-refined: {n_moved} "
        f"({n_moved / max(1, n_host):.0%}) | median |shift| {med_shift:.1f}s"
    )
    print(f"wrote {out_path}")
    return {"n_host": n_host, "n_moved": n_moved, "median_shift_s": med_shift}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-id", default="1fsnxchk")
    ap.add_argument("--in", dest="in_path", type=Path, default=None)
    ap.add_argument("--out", dest="out_path", type=Path, default=None)
    ap.add_argument("--band-s", type=float, default=30.0)
    ap.add_argument("--step-s", type=float, default=1.0)
    ap.add_argument("--gate", type=float, default=0.08)
    args = ap.parse_args(argv)
    in_path = args.in_path or (OUT_DIR / f"{args.set_id}_predicted_timeline.json")
    out_path = args.out_path or (OUT_DIR / f"{args.set_id}_recon_refined_timeline.json")
    rerank_timeline(
        args.set_id,
        in_path,
        out_path,
        band_s=args.band_s,
        step_s=args.step_s,
        gate=args.gate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
