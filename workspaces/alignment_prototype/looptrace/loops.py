#!/usr/bin/env python3
"""Phase 2 — DJ-edit repeat detection (loops + replays) and loop-collapse.

A DJ loop/replay repeats mix audio the SONG cannot produce on its own. Two
signals identify it inside a span of mix_vocals:

  1. self-match evidence: mel lag-diagonal proposal + waveform verification.
     On the ref audio a digital copy verifies at the CLONE floor, but the
     mix's vocal stem passes through Roformer separation under DIFFERENT
     instrumental beds per iteration, which degrades waveform identity
     (a confirmed real loop measured only -6.7 dB), so the mix-side residual
     floor is a noise-rejection gate, not the discriminator;
  2. the STRUCTURAL test (the discriminator): the span self-repeats at lag L
     (mix seconds). In song seconds that is L*slope. If the SONG has no
     repeat at that lag (Phase-1 audit repeat map), the song cannot have
     produced it — it must be a DJ edit. If the song does repeat at that
     lag, the DJ is just playing the song's own structure straight, and
     collapsing it would force a wrong same-source constraint.

Shapes: a button LOOP tiles contiguously (lag <= matched region length); a
REPLAY jumps back after a long stretch (lag > region). Both are one object:
source region [a0,a1] plus n_iter images at multiples of `period`.

Collapse: drop every image interval, keep the source with n_iter x evidence
weight; `CollapsedSpan` maps times both ways and `expand_segments`
replicates decoded segments over all iterations — every iteration maps to
the SAME source region (the Phase-2 hard constraint).

CLI (Phase-2 gate): detect on every eval-population acappella span, compare
with the GT answer key:
    venvs/audio/bin/python -m workspaces.alignment_prototype.looptrace.loops \
        --gt labeling/fixtures/bb12_ground_truth.yaml \
        [--audit .../out/audit_<set>.json]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from workspaces.alignment_prototype.looptrace import selfsim
from workspaces.alignment_prototype.looptrace.config import LOOP_V2, LoopConfig


@dataclass(frozen=True)
class Loop:
    source_start_s: float  # span-relative matched source region
    source_end_s: float
    period_s: float  # image k lives at source + k*period
    n_iter: int  # total iterations incl. the source
    residual_db: float  # waveform verification residual (evidence quality)
    mel_sim: float  # mel diagonal similarity of the proposal

    @property
    def end_s(self) -> float:
        """End of the last image, span-relative."""
        return self.source_end_s + (self.n_iter - 1) * self.period_s

    def images(self) -> list[tuple[float, float]]:
        return [
            (
                self.source_start_s + k * self.period_s,
                self.source_end_s + k * self.period_s,
            )
            for k in range(1, self.n_iter)
        ]


@dataclass(frozen=True)
class CollapsedSpan:
    """Span timeline with every loop/replay image interval removed."""

    duration_s: float
    keep: tuple[tuple[float, float], ...]
    weight: tuple[float, ...]
    loops: tuple[Loop, ...]

    def to_original(self, t_collapsed: float) -> float:
        acc = 0.0
        for (o0, o1), _w in zip(self.keep, self.weight):
            if t_collapsed <= acc + (o1 - o0):
                return o0 + (t_collapsed - acc)
            acc += o1 - o0
        return self.keep[-1][1]

    def to_collapsed(self, t_original: float) -> float:
        """Original span time -> collapsed time; image intervals fold onto
        their source region."""
        for lo in self.loops:
            for k in range(1, lo.n_iter):
                i0 = lo.source_start_s + k * lo.period_s
                i1 = lo.source_end_s + k * lo.period_s
                if i0 <= t_original < i1:
                    t_original = t_original - k * lo.period_s
                    break
        acc = 0.0
        for o0, o1 in self.keep:
            if t_original < o0:
                return acc
            if t_original <= o1:
                return acc + (t_original - o0)
            acc += o1 - o0
        return acc

    @property
    def collapsed_duration_s(self) -> float:
        return float(sum(o1 - o0 for o0, o1 in self.keep))

    def weight_at_collapsed(self, t_collapsed: float) -> float:
        """Evidence multiplier at a collapsed time (n_iter on a loop's
        canonical iteration, else 1)."""
        acc = 0.0
        for (o0, o1), w in zip(self.keep, self.weight):
            if t_collapsed <= acc + (o1 - o0):
                return float(w)
            acc += o1 - o0
        return 1.0


def detect_loops(
    y: np.ndarray,
    cfg: LoopConfig = LOOP_V2,
    *,
    song_lags_s: list[float] | None = None,
    slope: float = 1.0,
) -> list[Loop]:
    """Find DJ loops/replays in one mix span (audio at selfsim.SR).

    song_lags_s: the song's own repeat lags (song seconds) from the Phase-1
    audit (`[p["lag_s"] for p in track_audit["pairs"]]`). slope = song
    seconds per mix second (tempo ratio). When given, a mix self-repeat
    whose song-lag matches song structure is REJECTED (played straight, not
    a DJ edit). When None the structural test is skipped (fixtures)."""
    dur = len(y) / selfsim.SR
    pairs = selfsim.repeat_pairs(
        y,
        min_lag_s=cfg.min_lag_s,
        min_repeat_s=cfg.min_repeat_s,
        diag_thresh=cfg.diag_thresh,
        silence_ratio=cfg.silence_ratio,
        smooth_s=cfg.smooth_s,
        gap_close_s=cfg.gap_close_s,
        min_voiced_frac=cfg.min_voiced_frac,
    )
    pairs = _merge_same_lag(pairs, cfg)
    cands: list[Loop] = []
    for p in pairs:
        if not (cfg.min_lag_s <= p.lag_s <= min(cfg.max_lag_s, dur)):
            continue
        # structural test: is this lag the song's own repeat structure?
        if song_lags_s is not None:
            song_lag = p.lag_s * slope
            tol = max(cfg.song_lag_tol_s, cfg.song_lag_tol_frac * song_lag)
            if any(abs(song_lag - sl) <= tol for sl in song_lags_s):
                continue
        # verify on the FULL (gap-merged) region with trimmed edges — for a
        # tiling loop the region spans several iterations, all consistent
        # at lag=period, so the long window keeps r stable under bleed
        # (a one-period core was measured too noisy at -6 dB bleed).
        region = p.a1 - p.a0
        c0 = p.a0 + cfg.core_trim_frac * region
        c1 = p.a1 - cfg.core_trim_frac * region
        if c1 - c0 < 0.5:
            continue
        r, lag = selfsim.verify_pair(
            y, selfsim.RepeatPair(c0, c1, p.lag_s, p.diag_sim), pad_s=cfg.verify_pad_s
        )
        rdb = selfsim.residual_db(r)
        if rdb > cfg.min_residual_db or lag <= 0:
            continue
        if region >= lag * (1.0 - cfg.core_trim_frac):
            # contiguous tiling loop: source = first iteration; the merged
            # region covers iterations 1..N-1 (matching 2..N)
            src0, src1 = p.a0, p.a0 + lag
            n_iter = int(round(region / lag)) + 1
        else:  # replay: source = the matched region, one image
            src0, src1 = p.a0, p.a1
            n_iter = 2
        if n_iter < cfg.min_iters:
            continue
        cands.append(
            Loop(
                round(src0, 3),
                round(src1, 3),
                round(lag, 4),
                n_iter,
                round(rdb, 2),
                p.diag_sim,
            )
        )
    return _merge_loops(cands, cfg)


def _merge_same_lag(
    pairs: list[selfsim.RepeatPair], cfg: LoopConfig
) -> list[selfsim.RepeatPair]:
    """Separation bleed fragments one repeat's mel run into several short
    same-lag proposals — merge regions at the same lag across gaps up to
    `fragment_gap_s` so coverage (and hence iteration counting) is whole."""
    out: list[selfsim.RepeatPair] = []
    for p in sorted(pairs, key=lambda q: (q.lag_s, q.a0)):
        if (
            out
            and abs(p.lag_s - out[-1].lag_s) <= 0.35
            and p.a0 - out[-1].a1 <= cfg.fragment_gap_s
        ):
            out[-1] = selfsim.RepeatPair(
                out[-1].a0,
                max(out[-1].a1, p.a1),
                out[-1].lag_s,
                max(out[-1].diag_sim, p.diag_sim),
            )
        else:
            out.append(p)
    return out


def _merge_loops(cands: list[Loop], cfg: LoopConfig) -> list[Loop]:
    """A loop also self-matches at period multiples (iter1 vs iter3): among
    overlapping candidates with near-multiple periods keep the shortest
    period; among same-period overlaps keep the widest matched region."""
    out: list[Loop] = []
    for c in sorted(cands, key=lambda q: (q.period_s, -(q.end_s - q.source_start_s))):
        dup = False
        for k in out:
            if min(c.end_s, k.end_s) - max(c.source_start_s, k.source_start_s) <= 0:
                continue
            ratio = c.period_s / k.period_s
            if abs(ratio - round(ratio)) < cfg.period_multiple_tol:
                dup = True
                break
        if not dup:
            out.append(c)
    return sorted(out, key=lambda q: q.source_start_s)


def collapse(y: np.ndarray, loops: list[Loop]) -> CollapsedSpan:
    """Timeline with every image interval removed (clipped to the span)."""
    dur = len(y) / selfsim.SR
    drops: list[tuple[float, float]] = []
    for lo in loops:
        for i0, i1 in lo.images():
            i0, i1 = max(0.0, min(i0, dur)), max(0.0, min(i1, dur))
            if i1 - i0 > 1e-6:
                drops.append((i0, i1))
    drops.sort()
    merged: list[list[float]] = []
    for d0, d1 in drops:
        if merged and d0 <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], d1)
        else:
            merged.append([d0, d1])
    raw_keep: list[tuple[float, float]] = []
    t = 0.0
    for d0, d1 in merged:
        if d0 > t + 1e-6:
            raw_keep.append((t, d0))
        t = d1
    if t < dur - 1e-6:
        raw_keep.append((t, dur))
    # split keep intervals at loop-source boundaries so the n_iter evidence
    # weight applies to exactly the source sub-interval
    marks = sorted({b for lo in loops for b in (lo.source_start_s, lo.source_end_s)})
    keep: list[tuple[float, float]] = []
    for o0, o1 in raw_keep:
        cuts = [o0] + [m for m in marks if o0 < m < o1] + [o1]
        for c0, c1 in zip(cuts, cuts[1:]):
            if c1 - c0 > 1e-6:
                keep.append((c0, c1))
    weight = [
        1.0
        + sum(
            (lo.n_iter - 1)
            for lo in loops
            if lo.source_start_s - 1e-6 <= (o0 + o1) / 2 <= lo.source_end_s + 1e-6
        )
        for (o0, o1) in keep
    ]
    return CollapsedSpan(dur, tuple(keep), tuple(weight), tuple(loops))


def collapsed_audio(y: np.ndarray, cs: CollapsedSpan) -> np.ndarray:
    parts = [y[int(o0 * selfsim.SR) : int(o1 * selfsim.SR)] for o0, o1 in cs.keep]
    return np.concatenate(parts) if parts else y[:0]


def expand_segments(
    segments: list[tuple[float, float, float]], cs: CollapsedSpan
) -> list[tuple[float, float, float]]:
    """Map decoded segments (mix_start_collapsed, song_start, song_end) back
    to original span time.

    A decoded segment may CROSS keep-interval boundaries (a straight
    diagonal in collapsed time legitimately spans pre-loop + source + post
    material): split it at those boundaries, shift each piece to original
    time, and replicate any piece inside a loop's source region over all
    its images — every iteration maps to the SAME source region (the
    hard constraint). Segment ends follow the next-start convention, so
    ends are reconstructed before splitting and re-implied after (pieces
    and replicas emit contiguously)."""
    if not cs.loops:
        return sorted((round(m, 3), s0, s1) for m, s0, s1 in segments)
    # collapsed-time boundaries of the keep intervals AND of every loop's
    # source region — a decoded segment must be cut at source boundaries or
    # a piece merely CONTAINING the source is never replicated
    bounds_set: set[float] = {0.0}
    acc = 0.0
    for o0, o1 in cs.keep:
        acc += o1 - o0
        bounds_set.add(acc)
    for lo in cs.loops:
        bounds_set.add(cs.to_collapsed(lo.source_start_s))
        bounds_set.add(cs.to_collapsed(lo.source_end_s - 1e-9) + 1e-9)
    bounds = sorted(bounds_set)
    segs = sorted(segments)
    out: list[tuple[float, float, float]] = []
    for i, (mc0, s0, s1) in enumerate(segs):
        mc1 = segs[i + 1][0] if i + 1 < len(segs) else cs.collapsed_duration_s
        mc1 = max(mc1, mc0 + 1e-6)
        slope = (s1 - s0) / (mc1 - mc0)
        cuts = [mc0] + [b for b in bounds if mc0 < b < mc1] + [mc1]
        for c0, c1 in zip(cuts, cuts[1:]):
            if c1 - c0 < 1e-6:
                continue
            p0 = s0 + (c0 - mc0) * slope
            p1 = s0 + (c1 - mc0) * slope
            mo = cs.to_original((c0 + c1) / 2) - (c1 - c0) / 2
            out.append((round(mo, 3), round(p0, 3), round(p1, 3)))
            for lo in cs.loops:
                if lo.source_start_s - 1e-6 <= mo < lo.source_end_s:
                    for k in range(1, lo.n_iter):
                        img = mo + k * lo.period_s
                        if img < cs.duration_s - 1e-6:
                            out.append((round(img, 3), round(p0, 3), round(p1, 3)))
                    break
    return sorted(out)


def gt_loops_from_row(row: dict, tol_s: float = 0.75) -> list[tuple[float, float, int]]:
    """(span-relative mix_start, period, n_iter) repeats implied by GT: runs
    of consecutive ref_segments restarting at (nearly) the same ref position
    — the annotator's encoding of a loop/replay."""
    segs = sorted(row.get("ref_segments") or [], key=lambda s: float(s["mix_start_s"]))
    if len(segs) < 2:
        return []
    s0 = float(row["set_start_s"])
    out: list[tuple[float, float, int]] = []
    i = 0
    while i < len(segs) - 1:
        n = 1
        while i + n < len(segs):
            a, b = segs[i + n - 1], segs[i + n]
            if abs(float(b["ref_start_s"]) - float(a["ref_start_s"])) >= tol_s:
                break
            n += 1
        if n >= 2:
            period = float(segs[i + 1]["mix_start_s"]) - float(segs[i]["mix_start_s"])
            out.append((float(segs[i]["mix_start_s"]) - s0, period, n))
            i += n
        else:
            i += 1
    return out


def main(argv: list[str] | None = None) -> int:
    """Phase-2 gate runner: detect loops on every eval-population acappella
    span of a set's mix_vocals and tabulate against the GT answer key."""
    import argparse
    import json as _json
    import sys
    from pathlib import Path

    from workspaces.alignment_prototype.looptrace.audit import eval_population
    from workspaces.alignment_prototype.path_decode import find_aligning_dir

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--stems", default="acappella")
    p.add_argument(
        "--audit",
        type=Path,
        default=None,
        help="Phase-1 audit JSON: enables the song-structure exclusion",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    stems = {s.strip() for s in args.stems.split(",") if s.strip()}
    set_id, picked = eval_population(args.gt, stems)
    set_dir = find_aligning_dir(set_id)
    mix_path = set_dir / "mix_vocals.flac"
    audit = _json.loads(args.audit.read_text()) if args.audit else None

    rows = []
    for row, _track in picked:
        s0, s1 = float(row["set_start_s"]), float(row["set_end_s"])
        y = selfsim.load_audio(str(mix_path), offset_s=s0, duration_s=s1 - s0)
        song_lags = None
        if audit is not None:
            ta = audit["tracks"].get(row.get("track_id"))
            song_lags = [q["lag_s"] for q in ta["pairs"]] if ta else []
        slope = float(row.get("tempo_ratio") or 1.0)
        det = detect_loops(y, song_lags_s=song_lags, slope=slope)
        gt = gt_loops_from_row(row)
        rows.append(
            {
                "slot_label": row.get("slot_label"),
                "is_loop_gt": bool(row.get("is_loop")),
                "gt_loops": [
                    {"start_s": round(m, 2), "period_s": round(pd, 3), "n_iter": n}
                    for m, pd, n in gt
                ],
                "detected": [
                    {
                        "start_s": round(d.source_start_s, 2),
                        "period_s": round(d.period_s, 3),
                        "n_iter": d.n_iter,
                        "residual_db": d.residual_db,
                        "mel_sim": d.mel_sim,
                    }
                    for d in det
                ],
            }
        )
        flag = "LOOP" if (gt or row.get("is_loop")) else "    "
        print(
            f"  {row.get('slot_label')} {flag} gt={len(gt)} det={len(det)} "
            f"{[(d['period_s'], d['n_iter']) for d in rows[-1]['detected']]}",
            file=sys.stderr,
        )
    dest = args.out or (Path(__file__).parent / "out" / f"loops_{set_id}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_json.dumps({"set_id": set_id, "spans": rows}, indent=1))
    print(f"wrote {dest}")
    n_gt = sum(1 for r in rows if r["gt_loops"])
    n_det = sum(1 for r in rows if r["detected"])
    hit = sum(1 for r in rows if r["gt_loops"] and r["detected"])
    print(
        f"=== loops — {set_id}: spans={len(rows)} gt-loop spans={n_gt} "
        f"detected-loop spans={n_det} overlap={hit} ==="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
