#!/usr/bin/env python3
"""First aligner-placement eval against the doc's success criterion.

Per docs/alignment_objective.md the aligner is GIVEN the tracklist — identity is
known; stem discovery is ingest, not the aligner. So this measures Deliverable B
directly: for each KNOWN source span, place it in the mix and recover its key and
warp, then score against held-out GT with the documented tolerances:

  * placement   — |detected mix start − GT| in seconds and BARS (B-level: ±N bars)
  * key (B2)    — best chroma rotation == GT pitch class?
  * warp (B1)   — detected stretch vs GT tempo_ratio

Two placements are reported per span to separate the *signal* from the
*self-similarity* problem:

  * GLOBAL   — argmax of the matched filter over the whole mix (no prior). This is
               where a naive detector lands; it often hits a different reprise.
  * CORRIDOR — argmax within ±BAND s of the GT position. This is what the
               tracklist-ORDER prior (monotonic decode) would buy — the gap
               between the two quantifies how much that prior is worth.

Identity is given (the eval reads the source the labeling used); this is the
aligner sub-problem, not from-scratch detection over the 20k corpus.

    venvs/audio/bin/python -m workspaces.source_detection.align_eval \\
        --audit out/1fsnxchk_audit_fast.json --set-id 1fsnxchk [--limit 30] [--band 45]
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

from workspaces.source_detection import config, features  # noqa: E402
from workspaces.source_detection.matcher import _best_over_rot_stretch, _stretch_set  # noqa: E402

FRAME_S = config.FRAME_S
REF_OFFSETS_S = (0.0, 12.0, 24.0)   # try a few source windows (ref offset is an output too)


def _mix_path(mix_dir: Path, claimed_stem: str) -> Path | None:
    name = {"acappella": "mix_vocals.flac"}.get(claimed_stem, "mix_instrumental.flac")
    f = mix_dir / name
    if not f.is_file():
        f = mix_dir / "mix_instrumental.flac"
    return f if f.is_file() else None


def place_span(fact, mix_dir, mix_bpm, cfg, band_s):
    """Return (global, corridor) placements, each {pos_s, score, rot}."""
    src = fact.get("src_path")
    if not src or not Path(src).is_file():
        return None
    mp = _mix_path(mix_dir, fact["claimed_stem"])
    if mp is None:
        return None
    mixc = features.chroma_of(mp)
    srcc = features.chroma_of(Path(src))
    if srcc.shape[1] < 8 or mixc.shape[1] < 8:
        return None

    src_bpm = features.bpm_of(Path(src))
    stretches = _stretch_set(src_bpm, mix_bpm)
    tmpl_len = int(round(cfg.template_s / FRAME_S))
    gt_frame = int(round(fact["mix_start_s"] / FRAME_S))
    lo = max(0, gt_frame - int(band_s / FRAME_S))
    hi = gt_frame + int(band_s / FRAME_S)

    g = {"score": -9.0, "pos_s": 0.0, "rot": 0}
    c = {"score": -9.0, "pos_s": 0.0, "rot": 0}
    for off_s in REF_OFFSETS_S:
        off = int(round(off_s / FRAME_S))
        base = srcc[:, off:off + tmpl_len]
        if base.shape[1] < tmpl_len // 2:
            continue
        res = _best_over_rot_stretch(base, mixc, stretches, cfg)
        if res is None:
            continue
        score, rot, _str, _m = res
        k = int(score.argmax())
        if score[k] > g["score"]:
            g = {"score": float(score[k]), "pos_s": k * FRAME_S, "rot": int(rot[k])}
        hh = min(hi, score.size)
        if lo < hh:
            j = lo + int(score[lo:hh].argmax())
            if score[j] > c["score"]:
                c = {"score": float(score[j]), "pos_s": j * FRAME_S, "rot": int(rot[j])}
    return g, c


def _pc_ok(rot: int, gt_pitch: int) -> bool:
    return (rot % 12) == (gt_pitch % 12)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--set-id", required=True)
    ap.add_argument("--limit", type=int, default=30, help="0 = all spans")
    ap.add_argument("--band", type=float, default=45.0, help="corridor half-width (s)")
    args = ap.parse_args(argv)

    audit = json.loads(Path(args.audit).read_text())
    mix_dir = next(iter(sorted((Path.home() / "aligning").glob(f"{args.set_id}__*"))), None)
    if mix_dir is None:
        sys.exit(f"no ~/aligning folder for {args.set_id}")
    cfg = config.Config()
    mix_bpm = features.bpm_of(mix_dir / "mix_instrumental.flac")
    bar_s = (4 * 60.0 / mix_bpm) if mix_bpm else 1.875
    print(f"mix bpm≈{mix_bpm:.1f} → 1 bar≈{bar_s:.2f}s; corridor=±{args.band:.0f}s; "
          f"identity GIVEN (tracklist). placement vs held-out GT.\n")

    # eligible spans: real external sources with audio + a finite GT position
    spans = [f for f in audit["facts"]
             if f.get("src_path") and Path(f["src_path"]).is_file()
             and (f["ref_end_s"] - f["ref_start_s"]) > 1.0
             and f.get("audible_frac", 1.0) >= 0.5            # skip volume-muted spans
             and "mix" not in f["song"].strip().lower()]
    if args.limit:
        spans = spans[: args.limit]
    print(f"evaluating {len(spans)} known-identity spans…\n")

    rows = []
    for i, f in enumerate(spans):
        res = place_span(f, mix_dir, mix_bpm, cfg, args.band)
        if res is None:
            continue
        g, c = res
        rows.append({
            "song": f["song"], "stem": f["claimed_stem"], "gt": f["mix_start_s"],
            "g_err": abs(g["pos_s"] - f["mix_start_s"]), "g_pc": _pc_ok(g["rot"], f["pitch_coarse"]),
            "c_err": abs(c["pos_s"] - f["mix_start_s"]), "c_pc": _pc_ok(c["rot"], f["pitch_coarse"]),
            "g_score": g["score"], "c_score": c["score"],
        })
        if (i + 1) % 10 == 0:
            print(f"  …{i + 1}/{len(spans)}")

    if not rows:
        sys.exit("no spans evaluated")
    ge = np.array([r["g_err"] for r in rows]); ce = np.array([r["c_err"] for r in rows])

    def hit(arr, thr): return 100.0 * np.mean(arr <= thr)
    print(f"\n================ ALIGNER PLACEMENT EVAL (n={len(rows)}) ================")
    print(f"{'':12} {'GLOBAL (no prior)':>22} {'CORRIDOR (±'+str(int(args.band))+'s prior)':>26}")
    print(f"{'median err':12} {np.median(ge):18.1f} s {np.median(ce):22.1f} s")
    print(f"{'p90 err':12} {np.percentile(ge,90):18.1f} s {np.percentile(ce,90):22.1f} s")
    for thr, lbl in [(bar_s, '±1 bar'), (2*bar_s, '±2 bar'), (2.0, '±2 s'), (5.0, '±5 s')]:
        print(f"{'hit '+lbl:12} {hit(ge,thr):17.0f} % {hit(ce,thr):21.0f} %")
    gpc = 100.0 * np.mean([r["g_pc"] for r in rows])
    cpc = 100.0 * np.mean([r["c_pc"] for r in rows])
    print(f"{'key pc-acc':12} {gpc:17.0f} % {cpc:21.0f} %   (B2: correct pitch class)")
    print("=" * 62)
    print("\nworst global placements (self-similarity casualties):")
    for r in sorted(rows, key=lambda x: -x["g_err"])[:8]:
        print(f"  g_err={r['g_err']:6.0f}s  c_err={r['c_err']:5.0f}s  {r['song'][:44]:44} [{r['stem']}]")

    out = config.OUT_ROOT / f"{args.set_id}_align_eval.json"
    out.write_text(json.dumps({"set_id": args.set_id, "bar_s": bar_s, "band_s": args.band,
                               "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
