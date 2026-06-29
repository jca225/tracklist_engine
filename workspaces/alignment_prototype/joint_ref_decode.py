#!/usr/bin/env python3
"""Joint per-span ref decode: continuity + loops via the piecewise-linear path
decoder, feature-routed per stem.

Single-window matched filtering (refine_ref_offsets) finds the right CONTENT but
coin-flips among self-similar repeats and cannot represent loops/cuts at all.
This module does what the human annotator does: look at the whole span and require
the song to move forward sensibly, emitting a SEGMENT LIST (loops, section-jumps,
warps) — the GT schema's shape.

Core decode = ``path_decode.decode_path`` (Viterbi over a windowed matched-filter:
stay-on-diagonal free, jump costs ``lam``; a backward jump is a LOOP, a forward
jump a DJ edit/cut). Feature is routed per AXIS (nuisance-invariance): acappella →
HuBERT (phonetic, key-invariant — chroma is weak on vocals), regular/instrumental →
chroma. Reference and mix features are stem-routed (vocals/instrumental stem).

Reads ``out/<set_id>_predicted_timeline.json`` (post set_start placement) and
updates it in place: adds ``ref_segments`` ([{mix_start_s (ABS), ref_start_s,
ref_end_s}], the GT/decode_path schema) + ``ref_path_conf``; rewrites scalar
``ref_start_s``/``ref_end_s`` from the first/last segment.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.joint_ref_decode \\
        --set-id 1fsnxchk [--lam 0.15] [--window-s 12] [--hop-s 2]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.path_decode import (
    FPS,
    _ensure_feat,
    _job,
    _stretch_band,
)
from workspaces.alignment_prototype.refine_ref_offsets import (
    _MIX_SOURCE,
    _STEM_FILE,
    find_aligning_dir,
)

OUT_DIR = Path(__file__).resolve().parent / "out"

# stretch band fallback when the MERT beat grid is unavailable (no octave anchor)
_FALLBACK_STRETCHES = (0.5, 0.51, 0.96, 0.98, 1.0, 1.02, 1.04, 1.96, 2.0, 2.04)


def _feature_for(stem: str) -> str:
    """Axis-routed matched-filter feature: vocals → HuBERT, else chroma."""
    return "hubert" if stem == "acappella" else "chroma"


def _ref_audio_for(span: dict, track: dict) -> str | None:
    """Stem-routed reference audio (vocals/instrumental stem or full track)."""
    sk = _STEM_FILE.get(span.get("claimed_stem") or "regular")
    if sk:
        sp = (track.get("stems") or {}).get(sk)
        if sp and Path(sp).is_file():
            return sp
    lp = track.get("local_path")
    return lp if lp and Path(lp).is_file() else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--lam", type=float, default=0.15, help="forward-jump penalty")
    p.add_argument(
        "--lam-back",
        type=float,
        default=None,
        help="backward-jump penalty (default = --lam; repeat-ambiguity is better "
        "handled by fiber-aware scoring than by chasing a specific instance)",
    )
    p.add_argument("--window-s", type=float, default=12.0, help="matched-filter window")
    p.add_argument("--hop-s", type=float, default=2.0, help="window hop")
    p.add_argument("--hubert-layer", type=int, default=9)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    timeline_path = OUT_DIR / f"{args.set_id}_predicted_timeline.json"
    timeline = json.loads(timeline_path.read_text())
    spans = timeline["spans"]

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    # beat grid for the octave-folded stretch band (fallback if MERT missing)
    mix_series = ref_series = None
    try:
        from core.result import Ok
        from workspaces.alignment_prototype.mert_store import load_bb12_mert

        match load_bb12_mert(args.set_id):
            case Ok((_sid, ms, rs)):
                mix_series, ref_series = ms, rs
    except Exception as e:  # noqa: BLE001 — grid is an optimization, not required
        print(f"(no beat grid: {e}; using fallback stretch band)", file=sys.stderr)

    # precompute mix features per stem in the PARENT (MPS not fork-safe for hubert)
    mix_npy: dict[str, Path] = {}
    for stem, (fname, _) in _MIX_SOURCE.items():
        f = set_dir / fname
        if not f.is_file():
            continue
        feat = _feature_for(stem)
        print(f"{feat}({fname}) …", file=sys.stderr)
        mix_npy[stem] = _ensure_feat(
            f, f"{args.set_id}_{stem}", feat, args.hubert_layer
        )

    wlen, hop = int(args.window_s * FPS), int(args.hop_s * FPS)
    lam_back = args.lam if args.lam_back is None else args.lam_back
    jobs, skipped = [], 0
    for idx, s in enumerate(spans):
        t = by_tid.get(s["recording_id"])
        if t is None:
            skipped += 1
            continue
        stem = s.get("claimed_stem") or "regular"
        ref_path = _ref_audio_for(s, t)
        mnpy = mix_npy.get(stem) or mix_npy.get("regular")
        if ref_path is None or mnpy is None:
            skipped += 1
            continue
        a = int(s["set_start_s"] * FPS)
        n = int(max(0.0, s["set_end_s"] - s["set_start_s"]) * FPS)
        if n < 4:
            skipped += 1
            continue
        feat = _feature_for(stem)
        ref_npy = _ensure_feat(
            ref_path, ref_path, feat, args.hubert_layer
        )  # parent-side
        if mix_series is not None and ref_series is not None:
            shim = SimpleNamespace(
                recording_id=s["recording_id"], set_start_s=s["set_start_s"]
            )
            stretches = _stretch_band(shim, mix_series, ref_series)
        else:
            stretches = _FALLBACK_STRETCHES
        jobs.append(
            (
                idx,
                str(mnpy),
                a,
                n,
                str(ref_npy),
                stretches,
                args.lam,
                wlen,
                hop,
                lam_back,
            )
        )

    print(
        f"joint decode: {len(jobs)} spans ({skipped} no-audio), "
        f"win={args.window_s:.0f}s hop={args.hop_s:.0f}s lam={args.lam}, "
        f"{args.workers} workers…"
    )
    res: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, r in enumerate(ex.map(_job, jobs, chunksize=2)):
            res[r["idx"]] = r
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{len(jobs)}")

    n_multi, updated = 0, 0
    for idx, s in enumerate(spans):
        r = res.get(idx)
        if not r or not r["segs"]:
            continue
        s0 = s["set_start_s"]
        segs = [
            {
                "mix_start_s": round(s0 + ms, 2),
                "ref_start_s": round(rs, 2),
                "ref_end_s": round(re, 2),
            }
            for (ms, rs, re) in r["segs"]
        ]
        s["ref_segments"] = segs
        s["ref_path_conf"] = r["score"]
        s.setdefault("ref_start_detect", s["ref_start_s"])
        s["ref_start_s"] = segs[0]["ref_start_s"]
        s["ref_end_s"] = segs[-1]["ref_end_s"]
        updated += 1
        if len(segs) > 1:
            n_multi += 1

    timeline["ref_decode"] = (
        "path_decode jump-Viterbi, feature-routed (joint_ref_decode)"
    )
    timeline_path.write_text(json.dumps(timeline, indent=2))
    confs = [r["score"] for r in res.values() if r["segs"]]
    print(
        f"\nupdated {updated} spans; multi-segment (loops/edits): {n_multi}; "
        f"path conf median={np.median(confs) if confs else 0:.2f}"
    )
    print(f"rewrote {timeline_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
