#!/usr/bin/env python3
"""Order-constrained aligner placement (doc Deliverable B), with the two levers:

  LEVER 1 — local BPM. The matched-filter stretch grid is derived from the mix's
  actual local-tempo range (the beat grid stored as per-measure mix_start in the
  MERT npz), not a single global BPM — so a source can lock to whatever local
  tempo its true position has.

  LEVER 2 — tracklist order. From-scratch argmax is hopeless (self-similar
  reprises). The tracklist gives ORDER, so we place all spans jointly by a
  monotonic DP that maximizes total matched-filter score subject to
  non-decreasing mix positions. This is the prior the per-span eval lacked.

Reports, on the same spans: INDEPENDENT argmax (no order) vs ORDERED decode,
against held-out GT with the doc's tolerances (±bars, pitch class).

    venvs/audio/bin/python -m workspaces.source_detection.align_decode \\
        --audit out/1fsnxchk_audit_fast.json --set-id 1fsnxchk [--limit 40]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.source_detection import config, features  # noqa: E402
from workspaces.source_detection.matcher import _best_over_rot_stretch  # noqa: E402

FRAME_S = config.FRAME_S
POOL = int(round(1.0 / FRAME_S))         # ~1 s position bins
REF_OFFSETS_S = (0.0, 12.0)
BPM_CLAMP = (85.0, 165.0)


def local_bpm_values(set_id: str) -> np.ndarray:
    m = features.load_mert_npz(set_id)
    if m is None:
        return np.array([128.0])
    ms = np.asarray(m["mix_start"], dtype=float)
    dur = np.diff(ms)
    bpm = 4 * 60.0 / dur[(dur > 0.15) & (dur < 8)]      # ~4 beats / measure
    return np.clip(bpm, *BPM_CLAMP)


def stretch_grid(src_bpm: float | None, bpm_vals: np.ndarray, k: int = 7) -> list[float]:
    if not src_bpm or src_bpm <= 0:
        return list(config.FALLBACK_STRETCHES)
    vals = np.percentile(bpm_vals, np.linspace(8, 95, k))

    def fold(s: float) -> float:
        while s > 1.7:
            s /= 2
        while s < 0.6:
            s *= 2
        return s
    return sorted({round(fold(b / src_bpm), 3) for b in vals}) or [1.0]


def span_curve(srcc, mixc, stretches, cfg):
    """Best matched-filter score + rotation per mix frame, over ref offsets."""
    tmpl_len = int(round(cfg.template_s / FRAME_S))
    best = None
    for off_s in REF_OFFSETS_S:
        off = int(round(off_s / FRAME_S))
        base = srcc[:, off:off + tmpl_len]
        if base.shape[1] < tmpl_len // 2:
            continue
        res = _best_over_rot_stretch(base, mixc, stretches, cfg)
        if res is None:
            continue
        score, rot, _s, _m = res
        if best is None:
            best = [score.copy(), rot.copy()]
        else:
            n = min(best[0].size, score.size)
            m = score[:n] > best[0][:n]
            best[0][:n][m] = score[:n][m]
            best[1][:n][m] = rot[:n][m]
    return best


def to_bins(score, rot, n_bins):
    """Max-pool frame curve to 1 s bins; keep rotation at each bin's argmax."""
    L = (score.size // POOL) * POOL
    if L < POOL:
        return None, None
    s = score[:L].reshape(-1, POOL)
    a = s.argmax(1)
    bs = s.max(1)
    br = rot[:L].reshape(-1, POOL)[np.arange(s.shape[0]), a]
    out_s = np.full(n_bins, -9.0); out_r = np.zeros(n_bins, int)
    nb = min(s.shape[0], n_bins)
    out_s[:nb] = bs[:nb]; out_r[:nb] = br[:nb]
    return out_s, out_r


def _mix_path(mix_dir, stem):
    f = mix_dir / ("mix_vocals.flac" if stem == "acappella" else "mix_instrumental.flac")
    return f if f.is_file() else mix_dir / "mix_instrumental.flac"


def _slot_key(s: str):
    m = re.match(r"(\d+)", s or "")
    return (int(m.group(1)) if m else 9999, s or "")


def monotonic_decode(bin_scores: list[np.ndarray]) -> list[int]:
    """DP: pick a bin per span, non-decreasing, maximizing total score."""
    n = len(bin_scores); P = bin_scores[0].size
    dp = bin_scores[0].copy()
    ptr = []
    for i in range(1, n):
        run = -9e9; bi = 0
        pref = np.empty(P); parg = np.empty(P, int)
        for p in range(P):                      # running max/argmax over q<=p
            if dp[p] > run:
                run = dp[p]; bi = p
            pref[p] = run; parg[p] = bi
        dp = bin_scores[i] + pref
        ptr.append(parg)
    placements = [0] * n
    p = int(dp.argmax()); placements[-1] = p
    for i in range(n - 1, 0, -1):
        p = int(ptr[i - 1][p]); placements[i - 1] = p
    return placements


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--set-id", required=True)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args(argv)

    audit = json.loads(Path(args.audit).read_text())
    mix_dir = next(iter(sorted((Path.home() / "aligning").glob(f"{args.set_id}__*"))), None)
    cfg = config.Config()
    bpm_vals = local_bpm_values(args.set_id)
    mix_dur = int(features.chroma_of(mix_dir / "mix_instrumental.flac").shape[1] * FRAME_S) + 1
    bar_s = 4 * 60.0 / float(np.median(bpm_vals))

    spans = [f for f in audit["facts"]
             if f.get("src_path") and Path(f["src_path"]).is_file()
             and (f["ref_end_s"] - f["ref_start_s"]) > 1.0
             and f.get("audible_frac", 1.0) >= 0.5            # skip volume-muted spans
             and "mix" not in f["song"].strip().lower()]
    spans.sort(key=lambda f: (_slot_key(f["slot"]), f["mix_start_s"]))
    if args.limit:
        spans = spans[: args.limit]
    print(f"local-BPM grid from {len(bpm_vals)} measures (med {np.median(bpm_vals):.0f}); "
          f"1 bar≈{bar_s:.2f}s; ordered decode over {len(spans)} spans in slot order.\n")

    bin_scores, bin_rots, kept = [], [], []
    for i, f in enumerate(spans):
        mixc = features.chroma_of(_mix_path(mix_dir, f["claimed_stem"]))
        srcc = features.chroma_of(Path(f["src_path"]))
        grid = stretch_grid(features.bpm_of(Path(f["src_path"])), bpm_vals)
        res = span_curve(srcc, mixc, grid, cfg)
        if res is None:
            continue
        bs, br = to_bins(res[0], res[1], mix_dur)
        if bs is None:
            continue
        bin_scores.append(bs); bin_rots.append(br); kept.append(f)
        if (i + 1) % 10 == 0:
            print(f"  scored {i + 1}/{len(spans)}")

    placements = monotonic_decode(bin_scores)
    indep = [int(bs.argmax()) for bs in bin_scores]

    def report(name, picks):
        errs, pcs = [], []
        for f, p, br in zip(kept, picks, bin_rots):
            errs.append(abs(p * POOL * FRAME_S - f["mix_start_s"]))
            pcs.append((int(br[p]) % 12) == (f["pitch_coarse"] % 12))
        e = np.array(errs)
        print(f"  {name:24} median={np.median(e):6.1f}s  p90={np.percentile(e,90):6.1f}s  "
              f"hit±1bar={100*np.mean(e<=bar_s):3.0f}%  hit±2bar={100*np.mean(e<=2*bar_s):3.0f}%  "
              f"hit±5s={100*np.mean(e<=5):3.0f}%  key={100*np.mean(pcs):3.0f}%")
        return e

    print(f"\n=========== ORDERED DECODE vs INDEPENDENT (n={len(kept)}) ===========")
    report("independent argmax", indep)
    eo = report("ordered (monotonic DP)", placements)
    print("=" * 64)
    print("\nordered-decode worst spans:")
    order = sorted(range(len(kept)), key=lambda i: -abs(placements[i]*POOL*FRAME_S - kept[i]["mix_start_s"]))
    for i in order[:8]:
        f = kept[i]
        print(f"  err={abs(placements[i]*POOL*FRAME_S-f['mix_start_s']):6.0f}s  "
              f"placed={placements[i]*POOL*FRAME_S:6.0f}s gt={f['mix_start_s']:6.0f}s  "
              f"{f['song'][:38]:38} [{f['claimed_stem']}]")

    out = config.OUT_ROOT / f"{args.set_id}_align_decode.json"
    out.write_text(json.dumps({
        "set_id": args.set_id, "bar_s": bar_s, "n": len(kept),
        "ordered": [{"song": f["song"], "gt": f["mix_start_s"],
                     "placed": placements[i] * POOL * FRAME_S}
                    for i, f in enumerate(kept)]}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
