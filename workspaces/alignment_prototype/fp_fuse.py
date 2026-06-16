#!/usr/bin/env python3
"""Fusion eval: does vote-gated fingerprinting recover the wrong-CONTENT cases?

The fiber decomposition showed most decode error (~41pp) is the matched filter
landing on the WRONG content, not a wrong repeat. Fingerprinting (fp_probe)
localizes by exact transient coincidence and carries a confidence signal (vote
count). This pits, per single-line BB12 span:

  - decode : path_decode's chroma matched-filter placement
  - fp     : the landmark fingerprint offset
  - fused  : use fp when its votes >= --votes, else decode

scored strict (<2s) AND fiber-aware (HuBERT fibers — credit a same-fiber pick).
If fused > decode, vote-gated fingerprint override is a real recovery channel.

Single-line spans only (linear + odd-ratio): both methods emit one offset, so
the comparison is clean. Multi-seg/loop fusion is a later step.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.fp_fuse \
        --eval [--votes 40] [--stems regular,instrumental,acappella] [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.fp_probe import fp_offset
from workspaces.alignment_prototype.path_decode import (
    FPS,
    _ensure_feat,
    _stretch_band,
    decode_path,
    trajectory_acc,
)
from workspaces.alignment_prototype.refine_ref_offsets import (
    SR,
    _MIX_SOURCE,
    _STEM_FILE,
    find_aligning_dir,
)


def _job(args: tuple) -> dict:
    import librosa

    (
        idx,
        mix_npy,
        mix_audio,
        s0,
        full_span,
        fp_win,
        ref_npy,
        ref_audio,
        stretches,
        fp_stretches,
    ) = args
    # decode sees the FULL span (exactly as path_decode) -> real segments
    M = np.ascontiguousarray(
        np.load(mix_npy, mmap_mode="r")[:, int(s0 * FPS) : int((s0 + full_span) * FPS)],
        dtype=np.float32,
    )
    R = np.ascontiguousarray(np.load(ref_npy, mmap_mode="r"), dtype=np.float32)
    segs, _ = decode_path(M, R, tuple(stretches), 0.15)
    # fp probes the first fp_win seconds (its offset = ref @ span start)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mw, _ = librosa.load(mix_audio, sr=SR, mono=True, offset=s0, duration=fp_win)
        ry, _ = librosa.load(ref_audio, sr=SR, mono=True)
    fp_off, votes, _ = fp_offset(mw, ry, tuple(fp_stretches))
    return {"idx": idx, "segs": segs, "fp": round(fp_off, 2), "votes": votes}


def _single_seg(ref_start, span, ratio):
    return [(0.0, ref_start, ref_start + span * ratio)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", action="store_true")
    p.add_argument(
        "--gt", type=Path, default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml"
    )
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--stems", default="regular,instrumental,acappella")
    p.add_argument("--votes", type=int, default=40, help="fp override vote gate")
    p.add_argument("--max-win-s", type=float, default=15.0)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)
    if not args.eval:
        p.error("only --eval is wired")
    want = {s.strip() for s in args.stems.split(",") if s.strip()}

    import yaml
    from workspaces.alignment_prototype.mert_store import load_bb12_mert
    from workspaces.alignment_prototype.ref_fibers import compute_fibers
    from core.result import Err, Ok

    match load_bb12_mert(args.set_id):
        case Err(msg):
            sys.exit(f"grid load failed: {msg}")
        case Ok((_sid, mix_series, ref_series)):
            pass

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    # mix-channel chroma (decode feature) + mix audio path per stem
    mix_npy, mix_audio = {}, {}
    for stem, (fname, _) in _MIX_SOURCE.items():
        f = set_dir / fname
        if f.is_file():
            mix_npy[stem] = str(_ensure_feat(f, f"{args.set_id}_{stem}", "chroma", 9))
            mix_audio[stem] = str(f)

    rows = [
        r
        for r in yaml.safe_load(args.gt.read_text())["tracks"]
        if r.get("track_id")
        and (r.get("claimed_stem") or "regular") in want
        and r.get("ref_source") != "online_candidate"
        and not r.get("is_loop")
        and not r.get("ref_segments")  # single-line spans only
    ]
    jobs, meta = [], []
    for i, r in enumerate(rows):
        tr = by_tid.get(str(r["track_id"])) or by_tid.get(r.get("recording_id"))
        if not tr:
            continue
        stem = r.get("claimed_stem") or "regular"
        if stem not in mix_npy:
            stem = "regular"
        sk = _STEM_FILE.get(stem)
        ref = (tr.get("stems") or {}).get(sk) if sk else tr.get("local_path")
        if not ref or not Path(ref).is_file():
            ref = tr.get("local_path")
        if not ref or not Path(ref).is_file():
            continue
        s0 = float(r["set_start_s"])
        full_span = float(r["set_end_s"]) - s0
        if full_span < 4:
            continue
        fp_win = min(args.max_win_s, full_span)
        ref_npy = str(_ensure_feat(ref, ref, "chroma", 9))
        # decode searches the grid stretch band; fp tries a small symmetric band

        class _T:  # _stretch_band wants attrs
            pass

        tt = _T()
        tt.recording_id = r.get("recording_id") or r["track_id"]
        tt.set_start_s = s0
        stretches = _stretch_band(tt, mix_series, ref_series)
        jobs.append(
            (
                i,
                mix_npy[stem],
                mix_audio[stem],
                s0,
                full_span,
                fp_win,
                ref_npy,
                ref,
                stretches,
                (0.98, 1.0, 1.02),
            )
        )
        meta.append((r, stem, full_span))

    print(f"fusing {len(jobs)} single-line spans (vote gate {args.votes})…")
    res = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, rr in enumerate(ex.map(_job, jobs, chunksize=1)):
            res[rr["idx"]] = rr
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(jobs)}")

    # HuBERT fibers per ref (for fiber-aware scoring)
    fib_cache: dict[str, tuple] = {}

    def fibers_for(ref_audio: str):
        if ref_audio not in fib_cache:
            hf = np.load(_ensure_feat(ref_audio, ref_audio, "hubert", 9))
            fib_cache[ref_audio] = compute_fibers(hf, FPS, audio_path=ref_audio)
        return fib_cache[ref_audio]

    agg = {"decode": [], "fp": [], "fused": []}
    fagg = {"decode": [], "fp": [], "fused": []}
    by_stem: dict[str, dict] = {}
    for (i, *_rest), (r, stem, sp) in zip(jobs, meta):
        rr = res[i]
        # _rest = [mix_npy, mix_audio, s0, full_span, fp_win, ref_npy, ref_audio, ...]
        ref = _rest[6]  # ref_audio
        ratio = float(r.get("tempo_ratio") or 1.0)
        fib = fibers_for(ref)
        use_fp = rr["votes"] >= args.votes
        fp_seg = _single_seg(rr["fp"], sp, ratio)
        cand = {
            "decode": rr["segs"],  # real decode segments
            "fp": fp_seg,
            "fused": fp_seg if use_fp else rr["segs"],
        }
        for name, segs in cand.items():
            acc, _, facc = trajectory_acc(segs, r, fiber=fib)
            agg[name].append(acc)
            fagg[name].append(facc)
        by_stem.setdefault(stem, {"n": 0, "d": [], "f": []})
        by_stem[stem]["n"] += 1
        by_stem[stem]["d"].append(fagg["decode"][-1])
        by_stem[stem]["f"].append(fagg["fused"][-1])

    n = len(agg["decode"])
    print(f"\n=== fusion (n={n}, vote gate {args.votes}) — fiber-aware traj-acc ===")
    for name in ("decode", "fp", "fused"):
        s = 100 * np.mean(agg[name])
        f = 100 * np.mean(fagg[name])
        print(f"  {name:7} strict {s:4.0f}%   fiber-aware {f:4.0f}%")
    over = sum(1 for (i, *_), _ in zip(jobs, meta) if res[i]["votes"] >= args.votes)
    print(f"  fp overrode decode on {over}/{n} spans (votes >= {args.votes})")
    print("  by stem (fiber-aware decode -> fused):")
    for stem, d in by_stem.items():
        print(
            f"    {stem:12} n={d['n']:3}  {100 * np.mean(d['d']):3.0f}% -> "
            f"{100 * np.mean(d['f']):3.0f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
