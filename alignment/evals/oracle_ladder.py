"""Acappella oracle→e2e gap decomposition ("oracle ladder").

Attributes the acappella trajectory gap between end-to-end and oracle-placement
to {routing, identity, placement} by re-decoding the SAME acappella population
under progressively more oracle inputs, decoder held fixed at looptrace, scored
against a fixed GT-acappella-row denominator. Design + decision rule:
docs/superpowers/specs/2026-07-18-acappella-oracle-ladder-design.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

RUNGS: tuple[str, ...] = ("R0", "R1", "R2", "R3")

# stable per-GT-row key stamped onto each rung span (collision-free; slot_labels
# are NOT — GT uses flat numeric slots '012', the timeline uses w-layers '12w1',
# and a few GT slots repeat). summarize joins decoded spans back to GT by this.
GT_IDX_KEY = "_gt_idx"

# GT<->timeline overlap tolerance (seconds), mirroring score_timeline_vs_gt.
_OVERLAP_TOL_S = 5.0


def _nearest_same_recording(g: dict, r0_spans: list[dict]) -> dict | None:
    """Predicted span whose recording_id == GT track_id, nearest set_start.

    Mirrors build_span_table's nearest-same-recording pairing (no overlap
    requirement): identity is 'was this recording predicted anywhere', and the
    nearest occurrence is the one this GT row is scored against.
    """
    tid = str(g["track_id"])
    cands = [s for s in r0_spans if str(s.get("recording_id")) == tid]
    if not cands:
        return None
    gs = float(g["set_start_s"])
    return min(cands, key=lambda s: abs(float(s["set_start_s"]) - gs))


def _time_overlap_span(g: dict, r0_spans: list[dict]) -> dict | None:
    """Predicted span (ANY recording) overlapping the GT row's window, nearest
    set_start — the aligner's *coverage* of this moment, used to give a
    mis-identified GT row a predicted placement at the identity-oracle rung."""
    g0, g1 = float(g["set_start_s"]), float(g["set_end_s"])
    cands = [
        s
        for s in r0_spans
        if float(s["set_start_s"]) < g1 + _OVERLAP_TOL_S
        and float(s["set_end_s"]) > g0 - _OVERLAP_TOL_S
    ]
    if not cands:
        return None
    return min(cands, key=lambda s: abs(float(s["set_start_s"]) - g0))


def build_rung_timeline(
    rung: str, r0_spans: list[dict], gt_acap_rows: list[dict]
) -> list[dict]:
    """Acappella input-span list for *rung*, oracle-substituted.

    Join is by recording_id (+ time), never slot. Each output span is stamped
    with ``GT_IDX_KEY`` = the GT row's index so ``summarize`` can pair back
    without depending on slot conventions.

    - R0: predicted span for each correctly-identified GT row (stale routing kept).
    - R1: + GT stem (acappella routing) on that predicted span.
    - R2: + GT recording_id. Mis-identified rows are recovered here using the
      time-overlap coverage span's predicted placement; never-covered rows stay
      absent (recovered only at R3).
    - R3: ALL GT rows, GT recording + GT placement (full oracle).
    """
    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung!r} (expected one of {RUNGS})")
    out: list[dict] = []
    for i, g in enumerate(gt_acap_rows):
        span: dict | None = None
        if rung == "R3":
            span = {
                "slot_label": str(g["slot_label"]),
                "recording_id": str(g["track_id"]),
                "claimed_stem": "acappella",
                "set_start_s": float(g["set_start_s"]),
                "set_end_s": float(g["set_end_s"]),
                "ref_start_s": 0.0,
                "ref_end_s": 0.0,
            }
        else:
            same_rec = _nearest_same_recording(g, r0_spans)
            if rung in ("R0", "R1"):
                if same_rec is not None:
                    span = dict(same_rec)
                    if rung == "R1":
                        span["claimed_stem"] = "acappella"
            else:  # R2 identity oracle
                src = same_rec or _time_overlap_span(g, r0_spans)
                if src is not None:
                    span = dict(src)
                    span["recording_id"] = str(g["track_id"])
                    span["claimed_stem"] = "acappella"
        if span is None:
            continue  # GT row scores 0 downstream (fixed denominator)
        span.setdefault("ref_start_s", 0.0)
        span[GT_IDX_KEY] = i
        out.append(span)
    return out


def summarize_acappella(
    decoded_spans: list[dict],
    gt_acap_rows: list[dict],
    score_fn: Callable[[dict, dict], tuple[float, float]],
) -> dict:
    """Mean strict/fiber over the FIXED GT-acappella-row denominator.

    Decoded spans are paired to GT rows by the stamped ``GT_IDX_KEY``. A GT row
    with no decoded span contributes 0 to both sums — never dropped (that is the
    inflation trap of predicted-centric scoring). ``score_fn(span, gt_row) ->
    (strict, fiber)`` is injected so this is testable without audio.
    """
    dec_by_idx = {s[GT_IDX_KEY]: s for s in decoded_spans if GT_IDX_KEY in s}
    n = len(gt_acap_rows)
    n_scored = 0
    strict_sum = 0.0
    fiber_sum = 0.0
    for i, g in enumerate(gt_acap_rows):
        span = dec_by_idx.get(i)
        if span is None:
            continue  # missing decode -> contributes 0 to both sums
        st, fb = score_fn(span, g)
        strict_sum += float(st)
        fiber_sum += float(fb)
        n_scored += 1
    return {
        "n": n,
        "n_scored": n_scored,
        "strict": strict_sum / n if n else 0.0,
        "fiber": fiber_sum / n if n else 0.0,
    }


# --------------------------------------------------------------------------
# Runner (integration): re-decode each rung via joint_ref_decode (looptrace
# fixed), score against the fixed GT-acappella denominator. Heavy decode
# imports stay lazy so the unit tests above collect cheaply.
# --------------------------------------------------------------------------


def _load_gt_acap(set_id: str, gt_path: Path) -> list[dict]:
    import yaml

    from alignment.score_timeline_vs_gt import _REPO

    doc = yaml.safe_load(gt_path.read_text())
    if doc.get("set_id") != set_id:
        raise ValueError(f"GT set_id {doc.get('set_id')} != {set_id}")
    rows = [
        r
        for r in doc["tracks"]
        if (r.get("claimed_stem") == "acappella")
        and str(r.get("slot_label")) != "mix"
        and r.get("track_id")
    ]
    id_map_path = _REPO / "labeling" / "fixtures" / "id_maps" / f"{set_id}.json"
    if id_map_path.exists():
        id_map = json.loads(id_map_path.read_text())
        for r in rows:
            tid = str(r.get("track_id") or "")
            if tid in id_map and id_map[tid] != tid:
                r["track_id"] = id_map[tid]
    return rows


def _by_tid(set_dir: Path) -> dict[str, dict]:
    """recording/track id -> {track_id, stems, local_path} via the contract
    loader (by_track_id keys by both track_id AND recording_id). Mirrors the
    dict shape score_timeline_vs_gt._resolve_ref_audio expects."""
    from core.contracts.manifest import MANIFEST_FILENAME, load_manifest

    manifest = load_manifest(set_dir / MANIFEST_FILENAME)
    return {
        k: {"track_id": r.track_id, "stems": r.stems, "local_path": r.local_path}
        for k, r in manifest.by_track_id().items()
    }


def real_score_fn(
    set_id: str, by_tid: dict, fibers: bool
) -> Callable[[dict, dict], tuple[float, float]]:
    """(strict, fiber) for a decoded span vs its KNOWN GT row, via trajectory_acc."""
    import numpy as np

    from alignment import score_timeline_vs_gt as sc
    from alignment.path_decode import (
        FPS,
        _ensure_feat,
        trajectory_acc,
    )

    cache: dict[str, tuple] = {}

    def _fib(ref_audio: str | None):
        if not fibers or ref_audio is None:
            return None
        if ref_audio not in cache:
            from alignment.ref_fibers import compute_fibers

            hf = np.load(_ensure_feat(ref_audio, ref_audio, "hubert", 9))
            cache[ref_audio] = compute_fibers(hf, FPS, audio_path=ref_audio)
        return cache[ref_audio]

    def score_fn(span: dict, gt_row: dict) -> tuple[float, float]:
        ref_audio = sc._resolve_ref_audio(
            span, by_tid.get(span["recording_id"]), stem="acappella"
        )
        fib = _fib(ref_audio)
        pred = sc._pred_segs_from_span(span, anchor_s=float(gt_row["set_start_s"]))
        strict, _n, facc = trajectory_acc(pred, gt_row, fiber=fib)
        return float(strict), float(facc if fibers else strict)

    return score_fn


def run_ladder(
    set_id: str,
    gt_path: Path,
    r0_timeline: Path,
    out_dir: Path,
    fibers: bool = True,
) -> dict[str, dict]:
    from alignment import joint_ref_decode as jrd
    from alignment.refine_ref_offsets import find_aligning_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_acap = _load_gt_acap(set_id, gt_path)
    set_dir = find_aligning_dir(set_id)
    by_tid = _by_tid(set_dir)
    score_fn = real_score_fn(set_id, by_tid, fibers)
    r0 = json.loads(Path(r0_timeline).read_text())
    r0_spans = r0["spans"]

    table: dict[str, dict] = {}
    for rung in RUNGS:
        spans = build_rung_timeline(rung, r0_spans, gt_acap)
        if rung == "R0":
            decoded = spans  # already decoded in the _lt timeline
        else:
            in_path = out_dir / f"{rung}_in.json"
            out_path = out_dir / f"{rung}_out.json"
            in_path.write_text(json.dumps({"spans": spans}))
            rc = jrd.main(
                [
                    "--set-id",
                    set_id,
                    "--decoder",
                    "looptrace",
                    "--timeline",
                    str(in_path),
                    "--out",
                    str(out_path),
                    "--workers",
                    "8",
                ]
            )
            if rc != 0:
                raise RuntimeError(f"joint_ref_decode failed for {set_id} {rung}")
            decoded = json.loads(out_path.read_text())["spans"]
        table[rung] = summarize_acappella(decoded, gt_acap, score_fn)
        print(
            f"  {set_id} {rung}: strict={table[rung]['strict']:.3f} "
            f"fiber={table[rung]['fiber']:.3f} "
            f"(n={table[rung]['n']}, scored={table[rung]['n_scored']})"
        )
    (out_dir / "ladder.json").write_text(json.dumps(table, indent=2))
    return table


def _print_attribution(set_id: str, t: dict[str, dict]) -> None:
    f = {k: t[k]["fiber"] for k in RUNGS}
    print(f"\n[{set_id}] fiber-aware attribution (pp of the R0->R3 gap):")
    print(f"  routing   (R1-R0): {100 * (f['R1'] - f['R0']):+.1f}")
    print(f"  identity  (R2-R1): {100 * (f['R2'] - f['R1']):+.1f}")
    print(f"  placement (R3-R2): {100 * (f['R3'] - f['R2']):+.1f}")
    print(
        f"  instance headroom  strict->fiber @R0: "
        f"{100 * (t['R0']['fiber'] - t['R0']['strict']):+.1f}  "
        f"@R3: {100 * (t['R3']['fiber'] - t['R3']['strict']):+.1f}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--r0-timeline", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-fibers", action="store_true")
    a = p.parse_args(argv)
    out_dir = a.out_dir or (
        Path(__file__).resolve().parent.parent / "out" / "oracle_ladder" / a.set_id
    )
    t = run_ladder(a.set_id, a.gt, a.r0_timeline, out_dir, fibers=not a.no_fibers)
    _print_attribution(a.set_id, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
