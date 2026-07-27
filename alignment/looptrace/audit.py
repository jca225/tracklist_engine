#!/usr/bin/env python3
"""Phase 1 — ill-posedness audit of acappella repeats (CLONE vs DISTINCT TAKE).

For each reference acappella used by the GT acappella spans of a set:

  1. Self-alignment (`selfsim.repeat_pairs`): log-mel lag-diagonal scan finds
     every repeated region as a (region, lag) pair. NOT the HuBERT fibers —
     audit-v1 tried them and they under-detect (26 pairs over 19 tracks;
     HuBERT is blind to melodic repeats, see ref_fibers.compute_fibers_fp).
  2. Each pair is verified sample-accurately (`selfsim.verify_pair`, waveform
     xcorr around the coarse lag) and classified:
       CLONE          residual 1-r^2 at the digital-copy floor — production
                      copy-paste; content CANNOT tell the instances apart,
       DISTINCT TAKE  same content, measurably different rendition (breaths,
                      timing jitter, ad-libs, harmony stacks) — addressable
                      with instance-sensitive features.
  3. The GT trajectory of every eval-population span is decomposed second by
     second: clone-tied / distinct-tied / unique. The clone fraction is the
     strictly-unwinnable part; it bounds every later phase.

Clone equivalence maps: each clone pair maps t -> t + exact_lag on its region
(both directions); `clone_images` composes them transitively — used by
looptrace.eval to credit predictions landing on a clone-mate of GT.

Calibration: random non-repeat window pairs give the "unrelated" residual
distribution; the CLONE threshold must sit far below it. Thresholds live in
config.py, tuned on fixtures, never on the answer keys.

Usage:
    venvs/audio/bin/python -m alignment.looptrace.audit \
        --gt labeling/fixtures/bb12_ground_truth.yaml \
        [--out .../out/audit_<set>.json] [--track-id <tid>]  # single-track debug
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

from alignment.looptrace import selfsim  # noqa: E402
from alignment.looptrace.config import (  # noqa: E402
    AUDIT_V3,
    AuditConfig,
)
from alignment.path_decode import (  # noqa: E402
    _gt_pieces,
    _ref_at,
    _span_class,
    find_aligning_dir,
)
from alignment.refine_ref_offsets import (  # noqa: E402
    ref_audio_for,
)


# --- per-track audit --------------------------------------------------------
def audit_track(ref_path: str, cfg: AuditConfig = AUDIT_V3) -> dict:
    """Repeat map + CLONE/DISTINCT pair classification for one recording."""
    y = selfsim.load_audio(ref_path)
    pairs = selfsim.repeat_pairs(
        y,
        min_lag_s=cfg.min_lag_s,
        min_repeat_s=cfg.min_repeat_s,
        diag_thresh=cfg.diag_thresh,
        silence_ratio=cfg.silence_ratio,
        gap_close_s=cfg.gap_close_s,
        min_voiced_frac=cfg.min_voiced_frac,
    )
    verified = []
    for p in pairs:
        r, lag = selfsim.verify_pair(y, p, pad_s=cfg.verify_pad_s)
        rdb = selfsim.residual_db(r)
        verified.append(
            {
                "a0": p.a0,
                "a1": p.a1,
                "lag_s": round(lag, 4),
                "diag_sim": p.diag_sim,
                "r": round(r, 4),
                "residual_db": round(rdb, 2),
                "class": "clone" if rdb <= cfg.clone_residual_db else "distinct",
            }
        )
    verified = _dedup_verified(verified)

    # calibration: random window pairs (overwhelmingly non-repeat content)
    rng = np.random.default_rng(cfg.seed)
    dur = len(y) / selfsim.SR
    calib = []
    w = cfg.min_repeat_s
    if dur > 6 * w:
        for _ in range(cfg.calib_pairs):
            t1 = float(rng.uniform(0, dur - w))
            t2 = float(rng.uniform(0, dur - w))
            if abs(t2 - t1) < 2 * w:
                continue
            fake = selfsim.RepeatPair(t1, t1 + w, t2 - t1, 0.0)
            r, _ = selfsim.verify_pair(y, fake, pad_s=cfg.verify_pad_s)
            calib.append(round(selfsim.residual_db(r), 2))

    return {
        "ref_path": ref_path,
        "config": cfg.version,
        "duration_s": round(dur, 2),
        "pairs": verified,
        "calibration_nonrepeat_residual_db": calib,
    }


def _dedup_verified(pairs: list[dict], lag_tol: float = 0.35) -> list[dict]:
    """The coarse-lag prefilter dedups on mel-bin lags; after waveform
    verification, adjacent detections of one ridge resolve to the SAME exact
    lag — keep only the best-r pair among (overlapping region, same exact
    lag) duplicates."""
    kept: list[dict] = []
    for p in sorted(pairs, key=lambda q: -q["r"]):
        dup = any(
            abs(p["lag_s"] - k["lag_s"]) <= lag_tol
            and min(p["a1"], k["a1"]) - max(p["a0"], k["a0"]) > 0
            for k in kept
        )
        if not dup:
            kept.append(p)
    return sorted(kept, key=lambda q: (q["a0"], q["lag_s"]))


# --- clone equivalence maps --------------------------------------------------
def clone_images(
    t: float,
    pairs: list[dict],
    *,
    classes: frozenset[str] = frozenset({"clone"}),
    max_hops: int = 4,
    tol: float = 0.25,
) -> list[float]:
    """All song-times equivalent to `t` (including t) under the pairs of the
    given classes, by transitively applying each pair's t -> t + lag map on
    its region. classes={"clone"} = strictly-indistinguishable images;
    classes={"clone","distinct"} = any-repeat images (fiber-aware analog,
    but with exact lag maps)."""
    out = [float(t)]
    frontier = [float(t)]
    for _ in range(max_hops):
        nxt = []
        for x in frontier:
            for p in pairs:
                if p["class"] not in classes:
                    continue
                for img in (
                    x + p["lag_s"] if p["a0"] <= x <= p["a1"] else None,
                    x - p["lag_s"]
                    if p["a0"] + p["lag_s"] <= x <= p["a1"] + p["lag_s"]
                    else None,
                ):
                    if img is not None and all(abs(img - o) > tol for o in out):
                        out.append(img)
                        nxt.append(img)
        if not nxt:
            break
        frontier = nxt
    return out


def _covered(t: float, p: dict) -> bool:
    """Is song-time t on either side of pair p's repeat relation?"""
    return (p["a0"] <= t <= p["a1"]) or (
        p["a0"] + p["lag_s"] <= t <= p["a1"] + p["lag_s"]
    )


# --- span-second decomposition ----------------------------------------------
def decompose_span(row: dict, track_audit: dict, step_s: float = 1.0) -> dict:
    """Per-second GT-side decomposition: clone / distinct / unique seconds.

    clone    = the GT ref second has a digital-copy twin elsewhere in the song
               (unwinnable by content; only a prior can pick the instance),
    distinct = repeated but as a distinguishable take (addressable),
    unique   = not repeated (fully addressable)."""
    gt = _gt_pieces(row)
    s0, s1 = float(row["set_start_s"]), float(row["set_end_s"])
    pairs = track_audit["pairs"]
    counts = {"clone": 0, "distinct": 0, "unique": 0}
    for t in np.arange(s0, s1, step_s):
        g = float(_ref_at(gt, float(t)))
        if any(p["class"] == "clone" and _covered(g, p) for p in pairs):
            counts["clone"] += 1
        elif any(p["class"] == "distinct" and _covered(g, p) for p in pairs):
            counts["distinct"] += 1
        else:
            counts["unique"] += 1
    total = max(1, sum(counts.values()))
    return {
        "slot_label": row.get("slot_label"),
        "span_class": _span_class(row),
        "n_seconds": total,
        **counts,
        "frac_clone": round(counts["clone"] / total, 3),
        "frac_distinct": round(counts["distinct"] / total, 3),
    }


# --- eval population ---------------------------------------------------------
def eval_population(gt_yaml: Path, stems: set[str]) -> tuple[str, list[tuple]]:
    """The spans the frozen baseline decodes (mirrors path_decode --eval
    filtering: wanted stem, not online_candidate, not the mix row), each with
    its manifest track. Returns (set_id, [(row, track), ...])."""
    import yaml

    doc = yaml.safe_load(gt_yaml.read_text())
    set_id = doc["set_id"]
    set_dir = find_aligning_dir(set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)
    picked = []
    for row in doc.get("tracks", []):
        if row.get("slot_label") == "mix":
            continue
        if (row.get("claimed_stem") or "regular") not in stems:
            continue
        if row.get("ref_source") == "online_candidate":
            continue
        track = by_tid.get(row.get("track_id"))
        if track is None:
            continue
        picked.append((row, track))
    return set_id, picked


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gt", type=Path, default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml"
    )
    p.add_argument("--stems", default="acappella")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--track-id", default=None, help="audit a single track (debug)")
    args = p.parse_args(argv)
    cfg = AUDIT_V3
    stems = {s.strip() for s in args.stems.split(",") if s.strip()}

    set_id, picked = eval_population(args.gt, stems)
    if args.track_id:
        picked = [(r, t) for r, t in picked if r.get("track_id") == args.track_id]

    ref_paths: dict[str, str] = {}
    for row, track in picked:
        rp = ref_audio_for(row, track)
        if rp is not None and Path(rp).is_file():
            ref_paths[row["track_id"]] = str(rp)

    audits: dict[str, dict] = {}
    by_path: dict[str, dict] = {}
    for tid, rp in sorted(ref_paths.items()):
        if rp not in by_path:
            print(f"audit {tid}  {Path(rp).name} …", file=sys.stderr)
            by_path[rp] = audit_track(rp, cfg)
        audits[tid] = by_path[rp]

    spans = []
    for row, _track in picked:
        ta = audits.get(row["track_id"])
        if ta is None:
            continue
        d = decompose_span(row, ta)
        d["track_id"] = row["track_id"]
        spans.append(d)

    out = {
        "set_id": set_id,
        "gt": str(args.gt),
        "config": vars(cfg),
        "tracks": audits,
        "spans": spans,
    }
    dest = args.out or (Path(__file__).parent / "out" / f"audit_{set_id}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")

    # summary
    n_tracks = len(by_path)
    all_pairs = [p_ for a in by_path.values() for p_ in a["pairs"]]
    n_clone = sum(1 for p_ in all_pairs if p_["class"] == "clone")
    calib = [
        c for a in by_path.values() for c in a["calibration_nonrepeat_residual_db"]
    ]
    print(f"\n=== ill-posedness audit — {set_id} ({cfg.version}) ===")
    print(
        f"  tracks {n_tracks}  repeat-pairs {len(all_pairs)} "
        f"(CLONE {n_clone} / DISTINCT {len(all_pairs) - n_clone})"
    )
    if calib:
        print(
            f"  non-repeat calibration residual_db: median {np.median(calib):.1f} "
            f"min {min(calib):.1f}  (clone threshold {cfg.clone_residual_db})"
        )
    for cls in ("linear", "multiseg", "loop", "oddratio"):
        sel = [s for s in spans if s["span_class"] == cls]
        if not sel:
            continue
        sec = sum(s["n_seconds"] for s in sel)
        cl = sum(s["clone"] for s in sel)
        di = sum(s["distinct"] for s in sel)
        print(
            f"  {cls:9} n={len(sel):3}  seconds={sec:5}  "
            f"clone {100 * cl / sec:3.0f}%  distinct {100 * di / sec:3.0f}%  "
            f"unique {100 * (sec - cl - di) / sec:3.0f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
