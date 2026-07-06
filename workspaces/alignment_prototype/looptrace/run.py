#!/usr/bin/env python3
"""Phase 3 — the looptrace decoder: loop-collapse -> landmark point cloud ->
fixed-slope Hough -> segment-cover DP -> expanded segment list.

Per span: detect DJ loops/replays (Phase 2, structural test against the
song's Phase-1 repeat map), collapse them, hash the collapsed mix audio and
the reference (cached), match to a raw (t_mix, t_song) point cloud, pick
the slope by total Hough support over the beat-grid band, decode the
segment cover, and expand back to full mix time.

CLI (oracle placement, mirrors the frozen path_decode --eval population):
    venvs/audio/bin/python -m workspaces.alignment_prototype.looptrace.run \
        --gt labeling/fixtures/bb12_ground_truth.yaml \
        [--audit .../out/audit_<set>.json] [--dump .../out/lt_pred_<set>.json]

The dump is schema-compatible with `looptrace.eval` and scored with the
frozen `path_decode.trajectory_acc` for the legacy comparison.
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

from workspaces.alignment_prototype.looptrace import (  # noqa: E402
    landmarks,
    loops,
    segments,
    selfsim,
)
from workspaces.alignment_prototype.looptrace.config import (  # noqa: E402
    LM_V1,
    LOOP_V2,
    SEG_V1,
)


def decode_span(
    y: np.ndarray,
    ref_hashes: dict,
    slope_candidates: list[float],
    *,
    song_lags_s: list[float] | None = None,
    song_pairs: list[dict] | None = None,
    ref_path: str | None = None,
    ref_mel: tuple | None = None,  # (mel, hz) of the reference, for hybrid
    belief_window: tuple[float, float] | None = None,  # local coords (lo, hi)
    discrim: bool = False,
    hybrid: bool = False,
    lm_cfg=LM_V1,
    seg_cfg=SEG_V1,
    loop_cfg=LOOP_V2,
) -> tuple[list[tuple[float, float, float]], dict]:
    """Segment list [(mix_start_s, song_start_s, song_end_s)] for one span
    (span-relative mix times), plus diagnostics."""
    center = slope_candidates[len(slope_candidates) // 2]
    det = loops.detect_loops(y, loop_cfg, song_lags_s=song_lags_s, slope=center)
    cs = loops.collapse(y, det) if det else None
    y_dec = loops.collapsed_audio(y, cs) if cs else y
    mix_h = landmarks.audio_points(y_dec, lm_cfg)
    pts = landmarks.match_points(mix_h, ref_hashes, lm_cfg)
    weights = None
    if cs is not None and len(pts):
        weights = np.array(
            [cs.weight_at_collapsed(float(t)) for t in pts[:, 0]], dtype=np.float64
        )
    # rank slopes by histogram peakiness, then let the top few COMPETE on
    # decode quality — peakiness alone lost to noise peaks on weak-evidence
    # spans. Raw path evidence is biased toward WRONG slopes (the true
    # matches fragment across many short diagonals which the DP chains), so
    # each segment pays an MDL-style charge of `seg_cost_s` seconds of the
    # decode's own average evidence: fragmented covers stop winning.
    ranked = sorted(
        slope_candidates,
        key=lambda s: -segments.total_support(pts, s, seg_cfg),
    )
    dur_dec = len(y_dec) / selfsim.SR
    grid = np.arange(0.0, max(dur_dec, seg_cfg.grid_step_s), seg_cfg.grid_step_s)
    hyb = None
    if hybrid:
        if ref_mel is None and ref_path is not None:
            from workspaces.alignment_prototype.looptrace.residual import mel_cached

            ref_mel = mel_cached(ref_path)
        if ref_mel is not None:
            mix_mel, mix_hz = selfsim.mel_features(y_dec)
            hyb = (mix_mel, mix_hz, ref_mel[0], ref_mel[1])
    best_slope, segs, best_ev, best_adj = ranked[0], [], 0.0, -1.0
    for s in ranked[: seg_cfg.slope_top_k]:
        diags = segments.hough_diagonals(pts, s, seg_cfg)
        bonus = None
        if hyb is not None and len(diags) >= 2:
            bonus = segments.mel_emission(
                hyb[0], hyb[1], hyb[2], hyb[3], diags, s, grid, seg_cfg
            )
        sg, _dp_ev = segments.cover_dp(
            pts, s, dur_dec, seg_cfg, weights=weights, diagonals=diags, bonus=bonus
        )
        ev = segments.path_inlier_evidence(
            pts, sg, s, dur_dec, seg_cfg, weights=weights
        )
        adj = ev * max(0.0, 1.0 - len(sg) * seg_cfg.seg_cost_s / max(dur_dec, 1e-6))
        if adj > best_adj:
            best_slope, segs, best_ev, best_adj = s, sg, ev, adj
    if belief_window is not None and segs:
        # placement-quality signal: fraction of the decoded path's INLIER
        # EVIDENCE lying outside the believed span window. Evidence-weighted
        # (segment-TIME fraction measured uninformative: real padded windows
        # tile fully with weak-match segments, NULL rarely wins).
        lo, hi = belief_window
        if cs is not None:
            lo, hi = cs.to_collapsed(lo), cs.to_collapsed(hi)
        m_in = (pts[:, 0] >= lo) & (pts[:, 0] <= hi)
        w_all = weights if weights is not None else np.ones(len(pts))
        ev_in = segments.path_inlier_evidence(
            pts[m_in], segs, best_slope, dur_dec, seg_cfg, weights=w_all[m_in]
        )
        ev_tot = segments.path_inlier_evidence(
            pts, segs, best_slope, dur_dec, seg_cfg, weights=w_all
        )
        ev_out_frac = 1.0 - (ev_in / ev_tot) if ev_tot > 0 else 1.0
    else:
        ev_out_frac = None
    if discrim and segs and song_pairs and ref_path:
        from workspaces.alignment_prototype.looptrace.discrim import (
            reselect_instances,
        )

        segs = reselect_instances(segs, pts, best_slope, song_pairs, ref_path, dur_dec)
    if cs is not None:
        segs = loops.expand_segments(segs, cs)
    meta = {
        "n_points": int(len(pts)),
        "slope": best_slope,
        "evidence": round(best_ev, 1),
        # router signal: above-background evidence per decoded second —
        # high = the landmark cloud clearly explains this span
        "evidence_rate": round(best_ev / max(dur_dec, 1e-6), 2),
        "ev_out_frac": None if ev_out_frac is None else round(ev_out_frac, 3),
        "loops": [
            {
                "source_start_s": lo.source_start_s,
                "source_end_s": lo.source_end_s,
                "period_s": lo.period_s,
                "n_iter": lo.n_iter,
            }
            for lo in det
        ],
    }
    return segs, meta


def outside_frac(
    segs: list[tuple[float, float, float]], dur_s: float, lo: float, hi: float
) -> float:
    """Fraction of decoded-segment TIME lying outside the believed span
    window [lo, hi] (local coords). Placement-quality signal: a ratio
    within one decode, so it should survive the absolute point-density
    shift between fixtures and real separated vocals that broke
    evidence_rate twice (router, adaptive retry)."""
    tot = out = 0.0
    for i, (m0, _s0, _s1) in enumerate(segs):
        m1 = segs[i + 1][0] if i + 1 < len(segs) else dur_s
        m1 = max(m1, m0)
        tot += m1 - m0
        out += max(0.0, min(m1, lo) - m0) + max(0.0, m1 - max(m0, hi))
    return out / tot if tot > 0 else 0.0


def _slope_band(row: dict, mix_series, ref_series, t) -> list[float]:
    """Slope candidates: the beat-grid band (legacy _stretch_band,
    octave-folded) UNIONED with a canonical near-1 band. The grid's bar
    ratio is wrong on some refs (measured: true-slope peakiness 856 vs the
    grid band's best 292 on BB12 slot 073) — the union keeps the true slope
    in the candidate set without trusting either source alone."""
    from workspaces.alignment_prototype.path_decode import _stretch_band

    band = set(_stretch_band(t, mix_series, ref_series))
    for f in (0.94, 0.97, 1.0, 1.03, 1.06):
        for o in (0.5, 1.0, 2.0):
            band.add(round(f * o, 4))
    return sorted(band)


def main(argv: list[str] | None = None) -> int:
    from workspaces.alignment_prototype.dataset import load_set
    from workspaces.alignment_prototype.looptrace.audit import eval_population
    from workspaces.alignment_prototype.mert_store import load_bb12_mert
    from workspaces.alignment_prototype.path_decode import (
        _span_class,
        find_aligning_dir,
        trajectory_acc,
    )
    from workspaces.alignment_prototype.refine_ref_offsets import ref_audio_for

    from core.result import Err, Ok

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gt", type=Path, default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml"
    )
    p.add_argument("--stems", default="acappella")
    p.add_argument("--audit", type=Path, default=None)
    p.add_argument("--dump", type=Path, default=None)
    p.add_argument("--slot", default=None, help="single slot_label (debug)")
    p.add_argument(
        "--jitter-s",
        type=float,
        default=0.0,
        help="simulate placement error: shift each span's window center by "
        "±this (alternating sign per span, deterministic)",
    )
    p.add_argument(
        "--pad-s",
        type=float,
        default=0.0,
        help="SELF-PLACEMENT: widen the decoded mix window by this much on "
        "each side of the placement prior — the Hough diagonal's extent "
        "finds where the song actually plays; NULL absorbs the rest",
    )
    p.add_argument(
        "--hybrid",
        action="store_true",
        help="enable the hybrid mel-consistency DP emission (measured flat "
        "overall: big oddratio wins, small erosion elsewhere — see NOTES.md)",
    )
    p.add_argument(
        "--discrim",
        action="store_true",
        help="enable Phase-4 discriminability instance re-selection "
        "(DEFAULT OFF: measured to REGRESS on both sets — see NOTES.md)",
    )
    args = p.parse_args(argv)

    stems = {s.strip() for s in args.stems.split(",") if s.strip()}
    set_id, picked = eval_population(args.gt, stems)
    if args.slot:
        picked = [(r, t) for r, t in picked if str(r.get("slot_label")) == args.slot]
    set_dir = find_aligning_dir(set_id)
    mix_path = set_dir / "mix_vocals.flac"
    audit = json.loads(args.audit.read_text()) if args.audit else None

    match load_set(args.gt):
        case Err(msg):
            sys.exit(f"GT load failed: {msg}")
        case Ok((_gt, targets)):
            pass
    match load_bb12_mert(set_id):
        case Err(msg):
            sys.exit(f"grid load failed: {msg}")
        case Ok((_sid, mix_series, ref_series)):
            pass
    tgt_by_key = {
        (str(t.slot_label), round(t.set_start_s, 2)): t
        for t in targets
        if t.slot_label != "mix"
    }

    payload = []
    accs = []
    for row, track in picked:
        rp = ref_audio_for(row, track)
        if rp is None or not Path(rp).is_file():
            continue
        s0, s1 = float(row["set_start_s"]), float(row["set_end_s"])
        t = tgt_by_key.get((str(row.get("slot_label")), round(s0, 2)))
        if t is None:
            continue
        jit = args.jitter_s * (1 if len(payload) % 2 == 0 else -1)
        off = max(0.0, s0 + jit - args.pad_s)
        y = selfsim.load_audio(
            str(mix_path), offset_s=off, duration_s=(s1 + jit + args.pad_s) - off
        )
        ref_h = landmarks.ref_points_cached(str(rp))
        song_lags, song_pairs = None, None
        if audit is not None:
            ta = audit["tracks"].get(row.get("track_id"))
            song_lags = [q["lag_s"] for q in ta["pairs"]] if ta else []
            song_pairs = ta["pairs"] if ta else []
        slopes = _slope_band(row, mix_series, ref_series, t)
        segs, meta = decode_span(
            y,
            ref_h,
            list(slopes),
            song_lags_s=song_lags,
            song_pairs=song_pairs,
            ref_path=str(rp),
            belief_window=((s0 + jit) - off, (s1 + jit) - off)
            if args.pad_s > 0
            else None,
            discrim=args.discrim,
            hybrid=args.hybrid,
        )
        if args.pad_s > 0:
            dur_loc = (s1 + jit + args.pad_s) - off
            meta["out_frac"] = round(
                outside_frac(segs, dur_loc, (s0 + jit) - off, (s1 + jit) - off), 3
            )
        if off < s0:  # widened window -> re-express relative to the span
            segs = [(round(m - (s0 - off), 3), r0, r1) for m, r0, r1 in segs]
        acc, _n, _f = trajectory_acc(segs, row)
        accs.append((_span_class(row), acc))
        print(
            f"  {row.get('slot_label')} {_span_class(row):8} n_pts={meta['n_points']:5} "
            f"slope={meta['slope']:.3f} segs={len(segs)} loops={len(meta['loops'])} "
            f"traj<2s={100 * acc:3.0f}%",
            file=sys.stderr,
        )
        payload.append(
            {
                "slot_label": row.get("slot_label"),
                "track_id": row.get("track_id"),
                "set_start_s": s0,
                "set_end_s": s1,
                "claimed_stem": row.get("claimed_stem") or "regular",
                "span_class": _span_class(row),
                "ref_audio": str(rp),
                "segs": [list(s) for s in segs],
                "meta": meta,
            }
        )

    dest = args.dump or (Path(__file__).parent / "out" / f"lt_pred_{set_id}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {"gt": str(args.gt), "decoder": "looptrace", "spans": payload}, indent=1
        )
    )
    print(f"dumped {len(payload)} spans -> {dest}")
    print(f"\n=== looptrace.run — {set_id} trajectory accuracy (frozen metric) ===")
    for cls in ("ALL", "linear", "multiseg", "loop", "oddratio"):
        sel = [a for c, a in accs if cls == "ALL" or c == cls]
        if sel:
            print(
                f"  {cls:9} n={len(sel):3}  traj-acc(<2s) mean {100 * np.mean(sel):3.0f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
