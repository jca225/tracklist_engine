#!/usr/bin/env python3
"""Per-span diff: which acappella spans does HuBERT rescue over MFCC?

Runs the v7 vocal-verification scoring (similarity_probe) twice — once per
feature — over the *same* spans, distractor pool, and matched-filter search,
then classifies each span by how the true candidate's rank changes:

    rescued    MFCC rank > 1  ->  HuBERT rank == 1
    lost       MFCC rank == 1 ->  HuBERT rank > 1
    both-right / both-wrong    (no flip)

Hypothesis (phonetic embeddings are transposition/reverb tolerant): rescued
spans should skew toward key-shifted / heavily stretched plays. The GT yaml
has no explicit transpose field, so we surface tempo_ratio (GT) and the
stretch the matcher picked for the true candidate as manipulation proxies.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.hubert_vs_mfcc_diff \
        --set-id 1fsnxchk [--hubert-layer 9] [--n-distractors 15]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    STRETCHES, detect_offset, find_aligning_dir, ref_audio_for,
)
from workspaces.section_hsmm.similarity_probe import FPS, _feat  # noqa: E402


def _score_span(mix, win_slice, cands, by_tid, feature, layer, stretches):
    """Return (rank, margin, true_peak, best_dist_peak, true_stretch) or None."""
    a, n = win_slice
    a = min(a, max(0, mix.shape[1] - n))
    win = mix[:, a:a + n]
    if win.shape[1] < 8:
        return None
    peaks: dict[str, float] = {}
    true_stretch = None
    tid = cands[0]
    for c in cands:
        rp = ref_audio_for({"claimed_stem": "acappella"}, by_tid[c])
        if rp is None:
            continue
        rf = _feat(rp, f"ref_{c}_voc", feature, layer)
        if rf.shape[1] <= win.shape[1]:
            continue
        _, peak, stretch = detect_offset(win, rf, stretches)
        peaks[c] = peak
        if c == tid:
            true_stretch = stretch
    if tid not in peaks or len(peaks) < 3:
        return None
    tp = peaks[tid]
    others = sorted((v for k, v in peaks.items() if k != tid), reverse=True)
    rank = 1 + sum(1 for v in others if v > tp)
    return rank, tp - others[0], tp, others[0], true_stretch


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--gt", type=Path,
                   default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml")
    p.add_argument("--n-distractors", type=int, default=15)
    p.add_argument("--hubert-layer", type=int, default=9)
    p.add_argument("--max-win-s", type=float, default=15.0)
    args = p.parse_args(argv)

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    rows = [r for r in yaml.safe_load(args.gt.read_text())["tracks"]
            if (r.get("claimed_stem") == "acappella") and r.get("track_id")
            and not r.get("is_loop") and not r.get("ref_segments")]
    pool = [t for t in by_tid if ref_audio_for({"claimed_stem": "acappella"},
                                               by_tid[t]) is not None]

    print("loading mix_vocals features (mfcc + hubert) …", file=sys.stderr)
    mix_m = _feat(set_dir / "mix_vocals.flac", f"{args.set_id}_mix_vocals", "mfcc")
    mix_h = _feat(set_dir / "mix_vocals.flac", f"{args.set_id}_mix_vocals",
                  "hubert", args.hubert_layer)

    cls = {"rescued": [], "lost": [], "both-right": [], "both-wrong": []}
    for r in rows:
        tid = str(r["track_id"])
        if tid not in pool:
            continue
        dur_s = min(float(r["set_end_s"]) - float(r["set_start_s"]), args.max_win_s)
        a = int(float(r["set_start_s"]) * FPS)
        n = int(dur_s * FPS)
        cands = [tid] + [t for t in pool if t != tid][:args.n_distractors]
        sm = _score_span(mix_m, (a, n), cands, by_tid, "mfcc", 0, STRETCHES)
        sh = _score_span(mix_h, (a, n), cands, by_tid, "hubert", args.hubert_layer, STRETCHES)
        if sm is None or sh is None:
            continue
        rm, rh = sm[0], sh[0]
        if rm > 1 and rh == 1:
            key = "rescued"
        elif rm == 1 and rh > 1:
            key = "lost"
        elif rm == 1 and rh == 1:
            key = "both-right"
        else:
            key = "both-wrong"
        t = by_tid.get(tid, {})
        name = t.get("title") or t.get("name") or t.get("filename") or tid
        cls[key].append({
            "slot": r.get("slot_label", "?"), "name": name,
            "dur_s": round(dur_s, 1),
            "tempo_ratio": r.get("tempo_ratio"),
            "mfcc": f"rk{rm} m{sm[1]:+.2f} str{sm[4]:.3f}",
            "hub": f"rk{rh} m{sh[1]:+.2f} str{sh[4]:.3f}",
        })

    n_tot = sum(len(v) for v in cls.values())
    print(f"\n=== HuBERT(L{args.hubert_layer}) vs MFCC per-span diff "
          f"({args.set_id}, {n_tot} scorable acappella spans) ===")
    for key in ("rescued", "lost", "both-right", "both-wrong"):
        print(f"  {key:11s}: {len(cls[key])}")
    for key in ("rescued", "lost"):
        if not cls[key]:
            continue
        print(f"\n--- {key.upper()} ---")
        for s in cls[key]:
            tr = f" tempo_ratio={s['tempo_ratio']}" if s["tempo_ratio"] else ""
            print(f"  [{s['slot']}] {s['name'][:48]:48s} {s['dur_s']:5.1f}s{tr}")
            print(f"        MFCC {s['mfcc']:28s} | HuBERT {s['hub']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
