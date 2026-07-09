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
from workspaces.alignment_prototype.stem_resolve import resolve_stem

OUT_DIR = Path(__file__).resolve().parent / "out"

# stretch band fallback when the MERT beat grid is unavailable (no octave anchor)
_FALLBACK_STRETCHES = (0.5, 0.51, 0.96, 0.98, 1.0, 1.02, 1.04, 1.96, 2.0, 2.04)


def _feature_for(stem: str) -> str:
    """Axis-routed matched-filter feature: vocals → HuBERT, else chroma."""
    return "hubert" if stem == "acappella" else "chroma"


def _ref_audio_for(span: dict, track: dict, set_dir: Path | None = None) -> str | None:
    """Stem-routed reference audio (vocals/instrumental stem or full track).

    Disk is truth: the manifest ``stems`` field is written once at pull time
    and drifts stale (measured 2026-07-09: 44 BB11 + 13 BB12 acappella spans
    silently lost their looptrace decode to this — stems on disk since June,
    absent from the manifest). ``resolve_stem`` globs the on-disk slot dirs
    when the manifest hint misses."""
    sk = _STEM_FILE.get(span.get("claimed_stem") or "regular")
    if sk:
        sp = resolve_stem(set_dir, span.get("slot_label"), track, sk)
        if sp is not None:
            return str(sp)
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
    p.add_argument(
        "--decoder",
        choices=["legacy", "looptrace"],
        default="legacy",
        help="looptrace = landmark Hough + segment-cover DP per stem channel "
        "(acappella: mix_vocals vs ref vocal stem; regular: full mix vs full "
        "ref; instrumental: mix_instrumental vs ref instrumental stem); "
        "see workspaces/alignment_prototype/looptrace/",
    )
    p.add_argument(
        "--lt-stems",
        default="acappella,instrumental",
        help="which claimed_stem values route through looptrace when "
        "--decoder looptrace (others keep the legacy matched-filter path). "
        "regular is EXCLUDED by default: whole-stem looptrace on full-mix "
        "regular regresses BB12 -13pp (linear -26; loop/multiseg gain but "
        "linear is the grid's strongest cell) — see NOTES.md 'Regular / "
        "instrumental span routing (2026-07-08)'. Pass regular explicitly "
        "only for the loop-gated experiment.",
    )
    p.add_argument(
        "--audit",
        type=Path,
        action="append",
        default=None,
        help="looptrace Phase-1 audit JSON (enables the loop structural "
        "test); repeatable — per-stem audit files merge on track_id. "
        "Full-mix spans whose track has NO audit entry skip loop-collapse "
        "(bar-scale EDM self-repeats misread as DJ edits otherwise)",
    )
    p.add_argument(
        "--timeline",
        type=Path,
        default=None,
        help="input timeline JSON (default: out/<set-id>_predicted_timeline.json)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the updated timeline here instead of rewriting in place",
    )
    args = p.parse_args(argv)

    timeline_path = args.timeline or (
        OUT_DIR / f"{args.set_id}_predicted_timeline.json"
    )
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

    # looptrace branch: spans of the routed stems decode via the landmark
    # Hough + segment-cover DP (validated vs the frozen oracle-placement
    # baseline: acappella multiseg 35/21 vs 27/12; regular/instrumental at
    # parity-or-better with the loop guard, big loop-class wins); other
    # stems keep the legacy path below.
    lt_res: dict[int, list] = {}
    lt_status: dict[int, str] = {}  # idx -> why looptrace produced no segments
    audit = None
    if args.audit:
        audit = {"tracks": {}}
        for f in args.audit:
            audit["tracks"].update(json.loads(f.read_text()).get("tracks", {}))
    lt_stems = {s.strip() for s in args.lt_stems.split(",") if s.strip()}
    if args.decoder == "looptrace":
        from workspaces.alignment_prototype.looptrace import (
            landmarks as lt_landmarks,
        )
        from workspaces.alignment_prototype.looptrace.run import decode_span, mix_slice

        n_retry = 0
        n_weave = 0
        for idx, s in enumerate(spans):
            stem = s.get("claimed_stem") or "regular"
            if stem not in lt_stems:
                continue
            t = by_tid.get(s["recording_id"])
            ref_path = _ref_audio_for(s, t or {}, set_dir)
            mix_path = set_dir / _MIX_SOURCE.get(stem, _MIX_SOURCE["regular"])[0]
            if ref_path is None or not mix_path.is_file():
                lt_status[idx] = (
                    "skip-no-ref-audio" if t is not None else "skip-manifest-miss"
                )
                continue
            s0, s1 = float(s["set_start_s"]), float(s["set_end_s"])
            if s1 - s0 < 1.0:
                lt_status[idx] = "skip-too-short"
                continue
            ref_h = lt_landmarks.ref_points_cached(str(ref_path))
            song_lags = song_pairs = None
            ta = None
            if audit is not None:
                ta = audit["tracks"].get(s["recording_id"]) or audit["tracks"].get(
                    s.get("track_id", "")
                )
                song_lags = [q["lag_s"] for q in ta["pairs"]] if ta else []
                song_pairs = ta["pairs"] if ta else []
            # loop-collapse guard: full-mix (regular/instrumental) content
            # self-repeats at bar scale, so without THIS track's Phase-1
            # repeat map the detector misreads song structure as DJ edits
            # (measured: BB12 regular oracle 60% -> 47%). Vocal spans keep
            # the shipped behavior (sparse content, misfires rare).
            enable_loops = stem == "acappella" or ta is not None
            if mix_series is not None and ref_series is not None:
                shim = SimpleNamespace(recording_id=s["recording_id"], set_start_s=s0)
                slopes = set(_stretch_band(shim, mix_series, ref_series))
            else:
                slopes = set(_FALLBACK_STRETCHES)
            for f in (0.94, 0.97, 1.0, 1.03, 1.06):
                for o in (0.5, 1.0, 2.0):
                    slopes.add(round(f * o, 4))
            # self-placement gate (ACAPPELLA only — validated there; regular/
            # instrumental placement comes from the fp diagonal and is far
            # tighter, gate untested): decode the PADDED window first; if the
            # decoded evidence concentrates OUTSIDE the believed span window
            # (ev_out_frac >= gate), placement was wrong — keep the
            # self-placed padded decode; otherwise decode tight (validated:
            # placed spans keep tight quality 43%, 15s-misplaced 6%->17%).
            from workspaces.alignment_prototype.looptrace.config import SEG_V1

            segs = None
            if stem == "acappella":
                pad = SEG_V1.gate_pad_s
                off = max(0.0, s0 - pad)
                y_pad = mix_slice(
                    str(mix_path), offset_s=off, duration_s=(s1 + pad) - off
                )
                segs_p, meta_p = decode_span(
                    y_pad,
                    ref_h,
                    sorted(slopes),
                    song_lags_s=song_lags,
                    song_pairs=song_pairs,
                    ref_path=str(ref_path),
                    belief_window=(s0 - off, s1 - off),
                    enable_loops=enable_loops,
                )
                of = meta_p.get("ev_out_frac")
                if segs_p and of is not None and of >= SEG_V1.gate_out_frac:
                    segs = [(round(m - (s0 - off), 3), r0, r1) for m, r0, r1 in segs_p]
                    n_retry += 1
            # long-weave window growth (instrumental only — oracle-window
            # bound shows the window is the wall for 90-300 s instrumental
            # weaves (0.21->0.58) but does NOTHING for acappella; see
            # SegmentConfig.weave_pads_s). Grow the decode window while the
            # evidence keeps landing OUTSIDE the believed span AND the
            # evidence rate holds up (within-span guard against the
            # measured widen-admits-rivals regressions).
            if segs is None and stem == "instrumental":
                prev_rate = None
                for pad in SEG_V1.weave_pads_s:
                    off = max(0.0, s0 - pad)
                    y_pad = mix_slice(
                        str(mix_path), offset_s=off, duration_s=(s1 + pad) - off
                    )
                    segs_p, meta_p = decode_span(
                        y_pad,
                        ref_h,
                        sorted(slopes),
                        song_lags_s=song_lags,
                        song_pairs=song_pairs,
                        ref_path=str(ref_path),
                        belief_window=(s0 - off, s1 - off),
                        enable_loops=enable_loops,
                    )
                    of = meta_p.get("ev_out_frac")
                    rate = float(meta_p.get("evidence_rate") or 0.0)
                    if not segs_p or of is None or of < SEG_V1.gate_out_frac:
                        break  # content contained -> tight decode is right
                    if (
                        prev_rate is not None
                        and rate < SEG_V1.weave_rate_margin * prev_rate
                    ):
                        break  # wider window diluted evidence: keep last rung
                    segs = [(round(m - (s0 - off), 3), r0, r1) for m, r0, r1 in segs_p]
                    prev_rate = rate
                    n_weave += 1
            if segs is None:
                y = mix_slice(str(mix_path), offset_s=s0, duration_s=s1 - s0)
                segs, _meta = decode_span(
                    y,
                    ref_h,
                    sorted(slopes),
                    song_lags_s=song_lags,
                    song_pairs=song_pairs,
                    ref_path=str(ref_path),
                    enable_loops=enable_loops,
                )
            if segs:
                lt_res[idx] = segs
            else:
                # decode ran and produced NOTHING — do not let the span
                # silently degrade to the legacy path without a trace
                # (north-star law 5: diagnostics as values).
                lt_status[idx] = "looptrace-empty"
        n_status = {}
        for v in lt_status.values():
            n_status[v] = n_status.get(v, 0) + 1
        print(
            f"looptrace decoded {len(lt_res)} spans (stems={sorted(lt_stems)}, "
            f"{n_retry} self-placement retries, {n_weave} weave window growths); "
            f"fallbacks: {n_status or 'none'}",
            file=sys.stderr,
        )

    wlen, hop = int(args.window_s * FPS), int(args.hop_s * FPS)
    lam_back = args.lam if args.lam_back is None else args.lam_back
    jobs, skipped = [], 0
    for idx, s in enumerate(spans):
        if idx in lt_res:
            continue
        t = by_tid.get(s["recording_id"])
        if t is None:
            skipped += 1
            continue
        stem = s.get("claimed_stem") or "regular"
        ref_path = _ref_audio_for(s, t, set_dir)
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
                0.0,  # fwd_slope: flat (warp-off) — path_decode._job unpacks 12
                0.0,  # back_slope: flat — neutral default matches decode_path
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
        if idx in lt_status:
            s["ref_decode_status"] = lt_status[idx]
        raw = lt_res.get(idx) or (res.get(idx) or {}).get("segs")
        if not raw:
            continue
        s0 = s["set_start_s"]
        segs = [
            {
                "mix_start_s": round(s0 + ms, 2),
                "ref_start_s": round(rs, 2),
                "ref_end_s": round(re, 2),
            }
            for (ms, rs, re) in raw
        ]
        s["ref_segments"] = segs
        if idx in lt_res:
            s["ref_decoder"] = "looptrace"
            s["ref_decode_status"] = "looptrace"
        else:
            s["ref_path_conf"] = res[idx]["score"]
            s.setdefault("ref_decode_status", "legacy")
        s.setdefault("ref_start_detect", s["ref_start_s"])
        s["ref_start_s"] = segs[0]["ref_start_s"]
        s["ref_end_s"] = segs[-1]["ref_end_s"]
        updated += 1
        if len(segs) > 1:
            n_multi += 1

    # --- fiber hypothesis set (B3, additive): every decoded span carries the
    # instance set of the repeat class its ref_start lands in, plus the
    # membership/ambiguity signals. This is the W2 span-posterior hypothesis
    # set: when ambiguous=True, content alone cannot fix WHICH instance played
    # and downstream consumers (fusion, review, offboard labeler) must treat
    # the exact instance as decode-prior, not audio evidence.
    fiber_soft_cache: dict[str, tuple] = {}

    def _fiber_soft(ref_path: str):
        if ref_path not in fiber_soft_cache:
            from workspaces.alignment_prototype.fibers import compute_fibers_soft

            hf = np.load(_ensure_feat(ref_path, ref_path, "hubert", args.hubert_layer))
            fiber_soft_cache[ref_path] = compute_fibers_soft(
                hf, FPS, audio_path=ref_path
            )
        return fiber_soft_cache[ref_path]

    n_amb = 0
    for s in spans:
        if not s.get("ref_segments"):
            continue
        t = by_tid.get(s["recording_id"])
        ref_path = _ref_audio_for(s, t, set_dir) if t is not None else None
        if ref_path is None:
            continue
        try:
            labels, hz, mu, conf = _fiber_soft(str(ref_path))
        except Exception as exc:  # loud, never silent (kernel law 5)
            s["fiber_status"] = f"unavailable: {exc}"
            continue
        from workspaces.alignment_prototype.fibers import (
            fiber_ambiguity,
            fiber_intervals,
        )

        rs = float(s["ref_segments"][0]["ref_start_s"])
        amb = fiber_ambiguity(labels, hz, rs, mu=mu)
        s["fiber_ambiguity"] = amb
        fid = amb["fiber_id"]
        if fid >= 0:
            s["fiber_instances"] = [
                {"start_s": round(a, 2), "end_s": round(b, 2)}
                for a, b, lab in fiber_intervals(labels, hz, min_len_s=3.0)
                if lab == fid
            ]
            if amb["ambiguous"]:
                n_amb += 1
    print(f"fiber hypothesis sets: {n_amb} spans instance-ambiguous")

    timeline["ref_decode"] = (
        "path_decode jump-Viterbi, feature-routed (joint_ref_decode)"
        if args.decoder == "legacy"
        else f"looptrace landmark Hough+DP ({','.join(sorted(lt_stems))}) "
        "+ path_decode (others)"
    )
    dest = args.out or timeline_path
    dest.write_text(json.dumps(timeline, indent=2))
    confs = [r["score"] for r in res.values() if r["segs"]]
    print(
        f"\nupdated {updated} spans ({len(lt_res)} looptrace); "
        f"multi-segment (loops/edits): {n_multi}; "
        f"path conf median={np.median(confs) if confs else 0:.2f}"
    )
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
