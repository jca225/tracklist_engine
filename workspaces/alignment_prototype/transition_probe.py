#!/usr/bin/env python3
"""Probe: do regular/instrumental placement errors concentrate in TRANSITION
zones — where two bed layers (regular/instrumental) overlap and crossfade?

Two parts:
  A) structure (GT only): per bed span, what fraction of its set-time overlaps
     ANOTHER bed span (regular/instrumental)? That is the crossfade region.
  B) localization (decode): run path_decode per bed span, sample mix times,
     and split per-sample placement error into IN-overlap vs OUT-of-overlap.
     If the hypothesis holds, error is much higher inside overlap zones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

GT = _REPO / "labeling/fixtures/bb12_ground_truth.yaml"
BEDS = {"regular", "instrumental"}


def _stem(r):
    return (r.get("claimed_stem") or "regular").strip()


def _iv(r):
    return float(r["set_start_s"]), float(r["set_end_s"])


def overlap_len(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def main():
    rows = [
        r
        for r in yaml.safe_load(GT.read_text())["tracks"]
        if r.get("slot_label") != "mix"
    ]
    beds = [r for r in rows if _stem(r) in BEDS]

    print(f"=== A. structure: {len(beds)} bed spans (regular/instrumental) ===")
    bed_overlap_frac = []
    for r in beds:
        a = _iv(r)
        dur = a[1] - a[0]
        if dur <= 0:
            continue
        # union of overlap with OTHER bed spans
        mask_lo, mask_hi = a
        others = [_iv(o) for o in beds if o is not r]
        # measure covered seconds of `a` overlapped by any other bed
        pts = np.linspace(a[0], a[1], max(2, int(dur)))
        covered = np.zeros(pts.size, bool)
        for o in others:
            covered |= (pts >= o[0]) & (pts <= o[1])
        frac = covered.mean()
        bed_overlap_frac.append((r.get("slot_label"), _stem(r), dur, frac))

    fr = np.array([f for *_, f in bed_overlap_frac])
    print(
        f"  bed span overlaps another bed: mean {100 * fr.mean():.0f}%  "
        f"median {100 * np.median(fr):.0f}%  "
        f">50% overlapped: {100 * (fr > 0.5).mean():.0f}% of spans"
    )
    # fraction of total bed set-time that is multi-bed
    print("  per-span overlap fraction (slot, stem, dur, overlap%):")
    for slot, st, dur, f in sorted(bed_overlap_frac, key=lambda x: -x[3])[:12]:
        print(f"    {str(slot):6} {st:12} {dur:6.0f}s  {100 * f:3.0f}%")

    print("\n=== B. localization: per-sample placement error vs overlap ===")
    localize(beds)
    return bed_overlap_frac


MUTE_THR = 0.05  # matches labeling/als_io.py


def _gain_at(curve, t):
    """Interpolate a (set_time, gain) curve at t; unity if no curve."""
    if not curve:
        return 1.0
    if t <= curve[0][0]:
        return curve[0][1]
    if t >= curve[-1][0]:
        return curve[-1][1]
    for (x0, g0), (x1, g1) in zip(curve, curve[1:]):
        if x0 <= t <= x1:
            return g0 if x1 == x0 else g0 + (t - x0) / (x1 - x0) * (g1 - g0)
    return curve[-1][1]


def _curve(r):
    return [(float(x), float(g)) for x, g in (r.get("gain_curve") or [])]


def _audible_at(r, curve, t):
    """Is this bed above the mute floor at set-time t? Uses the real fader
    curve when present; else falls back to the clip extent."""
    s0, s1 = _iv(r)
    if not (s0 <= t <= s1):
        return False
    return _gain_at(curve, t) > MUTE_THR if curve else True


def _bed_intervals(beds):
    return [(o, _stem(o), o.get("slot_label"), _curve(o)) for o in beds]


def _dominance_window(self_curve, others, a, n):
    """Longest contiguous span-frame run [f0,f1) where THIS bed is the loudest
    bed (own gain >= every other bed's gain, and above mute). a = span start
    frame, n = span frames. Beds with no curve play at unity."""
    from workspaces.alignment_prototype.path_decode import FPS

    dom = np.zeros(n, bool)
    for f in range(n):
        t = (a + f) / FPS
        sg = _gain_at(self_curve, t) if self_curve else 1.0
        if sg <= MUTE_THR:
            continue
        # a competitor counts only while its own clip is playing; no curve = unity
        og = 0.0
        for o, c in others:
            os0, os1 = _iv(o)
            if os0 <= t <= os1:
                og = max(og, _gain_at(c, t) if c else 1.0)
        dom[f] = sg >= og
    best = (0, 0)
    f = 0
    while f < n:
        if dom[f]:
            g = f
            while g < n and dom[g]:
                g += 1
            if g - f > best[1] - best[0]:
                best = (f, g)
            f = g
        else:
            f += 1
    # buried fraction = audible frames where self is NOT the loudest bed. Only a
    # heavily-buried bed needs the override; a mostly-dominant bed is better
    # served by the robust full-span Viterbi (overriding it regresses).
    aud = np.array(
        [
            (_gain_at(self_curve, (a + f) / FPS) if self_curve else 1.0) > MUTE_THR
            for f in range(n)
        ]
    )
    buried_frac = float((aud & ~dom).sum()) / max(1, int(aud.sum()))
    return best[0], best[1], buried_frac


def localize(beds):
    """Run path_decode per bed span; split per-sample error IN vs OUT overlap."""
    import json

    from workspaces.alignment_prototype.dataset import load_set
    from workspaces.alignment_prototype.mert_store import load_bb12_mert
    from workspaces.alignment_prototype.path_decode import (
        FPS,
        _ensure_feat,
        _gt_pieces,
        _ref_at,
        _pieces,
        _stretch_band,
        decode_path,
        find_aligning_dir,
    )
    from workspaces.alignment_prototype.refine_ref_offsets import (
        _MIX_SOURCE,
        _STEM_FILE,
        detect_offset,
    )
    from core.result import Err, Ok

    match load_set(GT):
        case Err(m):
            sys.exit(m)
        case Ok((gt, targets)):
            pass
    match load_bb12_mert(gt.set_id):
        case Ok((_s, mix_series, ref_series)):
            pass
        case Err(m):
            sys.exit(m)

    raw = {
        (str(r.get("slot_label")), round(float(r.get("set_start_s", -1)), 2)): r
        for r in beds
    }
    set_dir = find_aligning_dir(gt.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    mix_npy = {}
    for stem, (fname, _) in _MIX_SOURCE.items():
        if stem not in {"regular", "instrumental"}:
            continue
        f = set_dir / fname
        if f.is_file():
            print(f"  chroma({fname}) …", file=sys.stderr)
            mix_npy[stem] = _ensure_feat(f, f"{gt.set_id}_{stem}", "chroma", 0)

    all_iv = _bed_intervals(beds)
    # abs ref error (s) per sample, single-bed baseline vs dominance-window
    in_err = {"base": [], "dom": []}
    out_err = {"base": [], "dom": []}
    per_span = []
    for t in targets:
        stem = t.claimed_stem or "regular"
        if stem not in {"regular", "instrumental"}:
            continue
        row = raw.get((t.slot_label, round(t.set_start_s, 2)))
        if row is None or row.get("ref_source") == "online_candidate":
            continue
        track = by_tid.get(t.recording_id)
        if not track:
            continue
        rp = None
        sk = _STEM_FILE.get(stem)
        if sk:
            sp = (track.get("stems") or {}).get(sk)
            if sp and Path(sp).is_file():
                rp = sp
        rp = rp or track.get("local_path")
        if not rp or not Path(rp).is_file():
            continue
        mnpy = mix_npy.get(stem) or mix_npy.get("regular")
        if mnpy is None:
            continue
        ref_npy = _ensure_feat(rp, rp, "chroma", 0)
        a = int(t.set_start_s * FPS)
        n = int(max(0.0, t.set_end_s - t.set_start_s) * FPS)
        if n < 8:
            continue
        M = np.ascontiguousarray(
            np.load(mnpy, mmap_mode="r")[:, a : a + n], dtype=np.float32
        )
        R = np.ascontiguousarray(np.load(ref_npy, mmap_mode="r"), dtype=np.float32)
        stretches = _stretch_band(t, mix_series, ref_series)
        wlen, hop = int(12 * FPS), int(2 * FPS)
        s0, s1 = float(row["set_start_s"]), float(row["set_end_s"])
        slope = float(row.get("tempo_ratio") or 1.0)
        gtp = _gt_pieces(row)
        self_curve = _curve(row)
        others = [(o, c) for o, _st, sl, c in all_iv if sl != t.slot_label]
        ts = np.arange(s0, s1, 1.0)

        def score(segs):
            if not segs:
                return None
            prp = _pieces([(s0 + ms, rs, re) for (ms, rs, re) in segs], s0, s1, slope)
            ein, eout = [], []
            for tt in ts:
                if self_curve and _gain_at(self_curve, tt) <= MUTE_THR:
                    continue  # don't score faded-down frames
                err = abs(_ref_at(prp, tt) - _ref_at(gtp, tt))
                covered = any(_audible_at(o, c, tt) for o, c in others)
                (ein if covered else eout).append(err)
            return ein, eout

        # DOMINANCE decode: find the longest window where THIS bed is the
        # loudest bed, decode its offset there (clean signal), then propagate
        # one linear diagonal across the whole span at the bed's tempo. In a
        # crossfade beds alternate dominance, so each bed is visible somewhere.
        f0, f1, buried_frac = _dominance_window(self_curve, others, a, n)
        dom_segs = None
        # only override the full-span decode when the bed is heavily buried
        # (>40% of audible frames dominated by another bed) AND there's a clean
        # window >=8s to decode from.
        if buried_frac > 0.4 and f1 - f0 >= int(8 * FPS) and (f1 - f0) < n:
            win = np.ascontiguousarray(M[:, f0:f1])
            ref_ws, pk, _st = detect_offset(win, R, tuple(stretches))
            ref_span0 = ref_ws - (f0 / FPS) * slope  # back to span start
            dom_segs = [(0.0, ref_span0, ref_span0 + (n / FPS) * slope)]

        base = score(decode_path(M, R, stretches, 0.15, wlen, hop)[0])
        # dominance falls back to baseline when there's no clean window
        dom = score(dom_segs) if dom_segs else base
        if base is None or dom is None:
            continue
        in_err["base"].extend(base[0])
        out_err["base"].extend(base[1])
        in_err["dom"].extend(dom[0])
        out_err["dom"].extend(dom[1])
        if base[0]:  # only overlap-touching spans are interesting here
            per_span.append(
                (
                    t.slot_label,
                    stem,
                    np.mean([e < 2 for e in base[0]]),
                    np.mean([e < 2 for e in dom[0]]) if dom[0] else float("nan"),
                    len(base[0]),
                    dom_segs is not None,
                )
            )

    def rep(tag, ie, oe):
        ie, oe = np.array(ie), np.array(oe)
        print(
            f"  [{tag:11}] IN-overlap n={ie.size:5} exact<2s {100 * (ie < 2).mean():3.0f}%"
            f" med {np.median(ie):4.1f}s  |  OUT n={oe.size:5} exact<2s "
            f"{100 * (oe < 2).mean():3.0f}% med {np.median(oe):4.1f}s"
        )

    rep("baseline", in_err["base"], out_err["base"])
    rep("dominance", in_err["dom"], out_err["dom"])
    print("\n  per-span IN-overlap exact<2s  (slot stem | base dom | n | had-window):")
    for slot, st, b, d, ni, had in per_span:
        delta = d - b
        flag = (
            "  <-- better" if delta > 0.05 else "  <-- worse" if delta < -0.05 else ""
        )
        print(
            f"    {str(slot):6} {st:12} | {100 * b:3.0f}% {100 * d:3.0f}% | {ni:3} "
            f"| {'dom' if had else 'fallback'}{flag}"
        )


if __name__ == "__main__":
    main()
