"""Lever A — convert the lyrics ANCHOR into warped SEGMENTS.

The lyrics channel places acappellas well (set_start ~2-4s, ref_start ~8.5s) but
emits a SCALAR, so production acappella trajectory-acc is ~3% — the placement win
never reaches the segment metric. Fix: run `path_decode.decode_path` on the
lyrics-PLACED mix window (HuBERT-routed), so the good anchor produces a warped
ref path.

This isolated eval answers: does decoding at the LYRICS placement recover the
segment accuracy of decoding at GT placement?

  - GT-placement decode  = the decoder CEILING for acappella (oracle, ~33% chroma
    / measured here on HuBERT).
  - lyrics-placement decode = what we'd actually ship.

Honest scoring trick: GT is used ONLY to score, never to place. The decode runs on
the window at the lyrics set_start; the returned mix-relative segments are shifted
by (lyrics_ss - gt_ss) so `trajectory_acc` samples them on GT's mix-time grid. If
lyrics≈GT placement, the windows coincide and the comparison is fair; any residual
is the real (shippable) placement error, which is the point.

Read-only; imports path_decode + lyrics_align; writes only neuro/out/. No edits to
infer.py / path_decode.py / joint_ref_decode.py.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.neuro.lyrics_segment_eval \
        [--window-s 12] [--hop-s 2] [--lam 0.15] [--no-fibers]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import yaml as _yaml

from core.result import Err, Ok
from workspaces.alignment_prototype.dataset import load_set
from workspaces.alignment_prototype.lyrics_align import (
    _bigram_times,
    _norm,
    _slot_order,
    candidate_diagonals,
    load_cached,
    monotonic_decode,
    transcribe_words,
)
from workspaces.alignment_prototype.mert_store import load_bb12_mert
from workspaces.alignment_prototype.path_decode import (
    FPS,
    _ensure_feat,
    _span_class,
    _stretch_band,
    decode_path,
    find_aligning_dir,
    trajectory_acc,
)

GT_YAML = _REPO / "labeling" / "fixtures" / "bb12_ground_truth.yaml"
HUBERT_LAYER = 9


def _lyrics_anchors(set_dir: Path, gt_tracks: list[dict], by_tid: dict) -> dict:
    """slot_label -> (set_start, ref_start) from the lyrics monotonic decode."""
    manifest_dur = max((float(t.get("set_end_s") or 0) for t in gt_tracks), default=0.0)
    max_slot = (
        max(
            (_slot_order(t["slot_label"])[0] for t in gt_tracks if t.get("slot_label")),
            default=1,
        )
        or 1
    )
    mix_bt = _bigram_times(_norm(transcribe_words(set_dir / "mix_vocals.flac")))
    aca = [t for t in gt_tracks if (t.get("claimed_stem") == "acappella")]
    aca.sort(key=lambda t: _slot_order(t["slot_label"]))
    spans, slots = [], []
    for s in aca:
        voc = (by_tid.get(s["track_id"], {}).get("stems") or {}).get("vocals")
        if not voc or not Path(voc).is_file():
            continue
        cw = load_cached(voc)
        if not cw:
            continue
        cands = candidate_diagonals(_norm(cw), mix_bt)
        if not cands:
            continue
        epos = _slot_order(s["slot_label"])[0] / max_slot * manifest_dur
        spans.append((cands, epos))
        slots.append(s["slot_label"])
    chosen = monotonic_decode(spans)
    return {sl: (ss, rs) for sl, (ss, rs) in zip(slots, chosen) if ss is not None}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window-s", type=float, default=12.0)
    p.add_argument("--hop-s", type=float, default=2.0)
    p.add_argument("--lam", type=float, default=0.15)
    p.add_argument("--no-fibers", action="store_true")
    args = p.parse_args(argv)

    match load_set(GT_YAML):
        case Err(msg):
            sys.exit(f"GT load failed: {msg}")
        case Ok((gt, targets)):
            pass
    match load_bb12_mert(gt.set_id):
        case Err(msg):
            sys.exit(f"grid load failed: {msg}")
        case Ok((_sid, mix_series, ref_series)):
            pass

    raw_tracks = _yaml.safe_load(GT_YAML.read_text()).get("tracks", [])
    raw = {
        (str(r.get("slot_label")), round(float(r.get("set_start_s", -1)), 2)): r
        for r in raw_tracks
    }

    set_dir = find_aligning_dir(gt.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    print("lyrics anchors (cached)…", file=sys.stderr)
    anchors = _lyrics_anchors(set_dir, raw_tracks, by_tid)
    print(f"  {len(anchors)} acappella spans have a lyrics anchor", file=sys.stderr)

    mix_vocals = set_dir / "mix_vocals.flac"
    print("hubert(mix_vocals)…", file=sys.stderr)
    mnpy = _ensure_feat(mix_vocals, f"{gt.set_id}_acappella", "hubert", HUBERT_LAYER)
    M = np.load(mnpy, mmap_mode="r")

    fiber_cache: dict[str, tuple] = {}

    def fibers_for(ref_npy: str, ref_audio: str):
        if args.no_fibers:
            return None
        if ref_audio not in fiber_cache:
            from workspaces.alignment_prototype.ref_fibers import compute_fibers

            fiber_cache[ref_audio] = compute_fibers(
                np.load(ref_npy), FPS, audio_path=ref_audio
            )
        return fiber_cache[ref_audio]

    wlen = int(args.window_s * FPS)
    hop = int(args.hop_s * FPS)
    spans_todo = [
        t
        for t in targets
        if (t.claimed_stem or "regular") == "acappella"
        and t.slot_label != "mix"
        and t.slot_label in anchors
    ]
    rows = []  # (class, gt_acc, lyr_acc, set_start_err, slot, label)
    for k, t in enumerate(spans_todo):
        print(f"  decode {k + 1}/{len(spans_todo)} {t.slot_label}…", file=sys.stderr)
        row = raw.get((t.slot_label, round(t.set_start_s, 2)))
        if row is None:
            continue
        track = by_tid.get(t.recording_id)
        voc = (track.get("stems") or {}).get("vocals") if track else None
        if not voc or not Path(voc).is_file():
            continue
        ref_npy = _ensure_feat(voc, voc, "hubert", HUBERT_LAYER)
        R = np.ascontiguousarray(np.load(ref_npy), dtype=np.float32)
        # tight, octave-free stretch band: acappella warp is near-linear (≤0.12s);
        # octaves only tripled cost and triggered the (now-fixed) decode_path bug.
        e = next(
            (s for s in _stretch_band(t, mix_series, ref_series) if 0.85 <= s <= 1.2),
            1.0,
        )
        stretches = tuple(round(e * f, 4) for f in (0.98, 1.0, 1.02))
        n = int(max(0.0, t.set_end_s - t.set_start_s) * FPS)
        if n < 4:
            continue
        fib = fibers_for(str(ref_npy), voc)

        def _decode(a0: int):
            a0 = max(0, min(a0, M.shape[1] - 4))
            Mw = np.ascontiguousarray(M[:, a0 : a0 + n], dtype=np.float32)
            segs, _ = decode_path(Mw, R, stretches, args.lam, wlen, hop, args.lam)
            return segs

        # GT placement (ceiling)
        gt_segs = _decode(int(t.set_start_s * FPS))
        # lyrics placement (shippable); shift segs onto GT's mix grid for scoring
        lyr_ss, _lyr_rs = anchors[t.slot_label]
        lyr_segs = _decode(int(lyr_ss * FPS))
        _, _, gt_facc = trajectory_acc(gt_segs, row, fiber=fib)
        shift = lyr_ss - t.set_start_s
        lyr_shifted = [(ms + shift, rs, re) for (ms, rs, re) in lyr_segs]
        _, _, lyr_facc = trajectory_acc(lyr_shifted, row, fiber=fib)

        rows.append(
            (
                _span_class(row),
                gt_facc,
                lyr_facc,
                abs(shift),
                t.slot_label,
                (t.label or "")[:28],
            )
        )

    if not rows:
        print("no acappella spans scored", file=sys.stderr)
        return 1

    def rep(name, sel):
        if not sel:
            return
        g = np.array([r[1] for r in sel])
        l = np.array([r[2] for r in sel])
        print(
            f"  {name:14} n={len(sel):3d}  GT-place {g.mean() * 100:5.0f}%   "
            f"lyrics-place {l.mean() * 100:5.0f}%   (Δ {(l.mean() - g.mean()) * 100:+5.0f})"
        )

    print(
        f"\n=== Lever A: lyrics-anchored segment decode (HuBERT, fiber-aware) — "
        f"{len(rows)} acappella spans ==="
    )
    print("  trajectory-acc: GT-placement (ceiling) vs lyrics-placement (shippable)")
    rep("ALL", rows)
    for cls in ("linear", "multiseg", "loop", "oddratio"):
        rep(cls, [r for r in rows if r[0] == cls])
    sse = np.array([r[3] for r in rows])
    print(
        f"\n  lyrics set_start err vs GT: median {np.median(sse):.1f}s  "
        f"p90 {np.percentile(sse, 90):.1f}s"
    )

    # worst spans (lyrics-place) for inspection
    worst = sorted(rows, key=lambda r: r[2])[:6]
    print("\n  worst lyrics-place spans:")
    for cls, g, l, se, sl, lab in worst:
        print(
            f"    {sl:6} {cls:9} GT {g * 100:3.0f}% lyr {l * 100:3.0f}% "
            f"(ss_err {se:4.1f}s)  {lab}"
        )

    out = {
        "n": len(rows),
        "all_gt": float(np.mean([r[1] for r in rows])),
        "all_lyr": float(np.mean([r[2] for r in rows])),
        "set_start_err_median": float(np.median(sse)),
    }
    out_path = Path(__file__).resolve().parent / "out" / "lever_a_lyrics_segments.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
